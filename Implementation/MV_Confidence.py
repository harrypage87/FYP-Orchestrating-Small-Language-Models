import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import gc
from datetime import datetime
import json
import multiprocessing
from multiprocessing import Process, Queue
from collections import defaultdict

'''
Independent Confidence Scoring Benchmark
Each solution is evaluated independently in its own prompt.
The judge is asked to assess whether the solution is correct (binary yes/no).
The solution with the highest aggregate confidence score wins.
This avoids presenting all solutions together, eliminating positional bias entirely.

Model order: first=A, second=B, third=C
'''
MODELS = {
    "CodeGemma 7B Instruct": "google/codegemma-7b-it",
    "CodeLlama 7B Instruct": "codellama/CodeLlama-7b-Instruct-hf",
    "DeepSeek Coder 7B Instruct": "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
}

DATASET_PATH = "/home/demouser/Desktop/121336311/McEval DataSet/McEval_Generation_Tasks.csv"


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    return df.head(50)


# ─── Test execution (unchanged from pairwise) ────────────────────────────────

def _execute_test_in_process(code_to_test: str, test_code: str, entry_point: str, result_queue: Queue):
    result = {'passed': False, 'error': None, 'code_tested': code_to_test}
    try:
        namespace = {}
        exec("from typing import List, Dict, Tuple, Optional, Union, Any", namespace)
        try:
            exec(code_to_test, namespace)
        except SyntaxError as e:
            result['error'] = f"Syntax error: {str(e)}"
            result_queue.put(result)
            return
        except Exception as e:
            result['error'] = f"Execution error: {str(e)}"
            result_queue.put(result)
            return
        if entry_point not in namespace:
            result['error'] = f"Function '{entry_point}' not found"
            result_queue.put(result)
            return
        try:
            exec(test_code, namespace)
        except AssertionError as e:
            result['error'] = f"Test assertion failed: {str(e) or 'Assertion failed (no message)'}"
            result_queue.put(result)
            return
        except Exception as e:
            result['error'] = f"Test error: {str(e)}"
            result_queue.put(result)
            return
        result['passed'] = True
        result_queue.put(result)
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
        result_queue.put(result)


def run_unit_test(code: str, test_code: str, entry_point: str, timeout: int = 40) -> Dict:
    result = {'passed': False, 'error': None, 'code_tested': None}
    try:
        code_to_test = code.strip()

        solution_match = re.search(r'<solution>(.*?)</solution>', code_to_test, re.DOTALL)
        if solution_match:
            code_to_test = solution_match.group(1).strip()
        else:
            if '</solution>' in code_to_test:
                code_to_test = code_to_test.split('</solution>')[0].strip()
            else:
                result['error'] = "No <solution> tags found in model output"
                return result

        if '<end_of_turn>' in code_to_test:
            code_to_test = code_to_test.split('<end_of_turn>')[0].strip()
        if code_to_test.startswith('>>>'):
            code_to_test = code_to_test[3:].strip()
        if code_to_test.startswith('```python'):
            code_to_test = code_to_test[9:].strip()
        elif code_to_test.startswith('```'):
            code_to_test = code_to_test[3:].strip()
        if code_to_test.endswith('```'):
            code_to_test = code_to_test[:-3].strip()

        code_to_test = re.sub(r'```python\s*\n?', '\n', code_to_test)
        code_to_test = re.sub(r'```\s*\n?', '\n', code_to_test)
        lines = [l for l in code_to_test.split('\n') if l.strip() not in ['```', '```python', '```py']]
        code_to_test = '\n'.join(lines)

        result['code_tested'] = code_to_test

        if not code_to_test.strip():
            result['error'] = "Generated code is empty after extraction"
            return result

        result_queue = Queue()
        process = Process(target=_execute_test_in_process, args=(code_to_test, test_code, entry_point, result_queue))
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            result['error'] = f"Infinite loop detected: exceeded {timeout}s timeout"
            return result

        if not result_queue.empty():
            result = result_queue.get()
        else:
            result['error'] = "Test process terminated unexpectedly"

    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
    return result


# ─── Model loading / cleanup ──────────────────────────────────────────────────

def load_model_safe(model_name: str, model_path: str):
    import os
    os.environ["ACCELERATE_USE_FSDP"] = "false"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None, None


def cleanup_model(model, tokenizer):
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Generation ───────────────────────────────────────────────────────────────

def generate_code(tokenizer, model, device, prompt, max_tokens, temperature):
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            prompt_length = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=1.0 if temperature == 0 else temperature,
                do_sample=False if temperature == 0 else True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            )
            generated_tokens = outputs[0][prompt_length:]
            if len(generated_tokens) == 0:
                return ""
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            del inputs, outputs
            torch.cuda.empty_cache()
            return result.strip()
    except Exception as e:
        st.error(f"Generation exception: {str(e)}")
        return f"# Generation error: {e}"


def create_generation_prompt(task_prompt: str, model_name: str) -> str:
    if "CodeGemma" in model_name:
        return f"""<start_of_turn>user
{task_prompt}

Please provide a complete solution to the task. Encapsulate your solution within <solution> tags.

IMPORTANT: Do NOT add any explanations or commentary after the </solution> tag. End your response immediately after </solution>.

Example response format:
<solution>
from typing import List

def function_name(params):
    # Your implementation
    return result
</solution>
<end_of_turn>
<start_of_turn>model
"""
    elif "DeepSeek" in model_name:
        return f"""<instruction>
<task>
{task_prompt}
</task>
Please provide a complete solution to the task described within the <task> tag.
Encapsulate your solution within <solution> tags. Include any necessary imports inside the <solution> tags.

IMPORTANT: Do NOT add any explanations or commentary after the </solution> tag. End your response immediately after </solution>.

Example response format:
>>> <solution>
from typing import List

def function_name(params):
    # Your implementation
    return result
</solution>
</instruction>
Your response:"""
    else:
        # Code Llama
        return f"""[INST] {task_prompt}

Write a complete Python solution. Wrap your code in <solution> tags.

Example:
<solution>
def function_name(params):
    # implementation
    return result
</solution>
[/INST]

<solution>"""


# ─── Independent Confidence Scoring ───────────────────────────────────────────

def _clean_solution_body(raw: str) -> str:
    """Strip <solution> wrapper tags so we expose only the code body."""
    s = raw
    if '</solution>' in s:
        s = s.split('</solution>')[0]
    if '<solution>' in s:
        s = s.split('<solution>')[-1]
    return s.strip()


def create_confidence_prompt(task: str, solution_code: str, model_name: str = "") -> str:
    """
    Build an independent assessment prompt for a single solution.
    The judge evaluates only this solution without seeing any alternatives.
    Simplified prompt that makes clear solutions may contain errors.
    """
    core = (
        f"You are evaluating a solution to a programming task. "
        f"The solution may be correct or may contain errors.\n\n"
        f"Task:\n{task}\n\n"
        f"Proposed solution:\n{solution_code}\n\n"
        f"Is this solution correct? Respond with ONLY 'CORRECT' or 'INCORRECT' wrapped in <assessment> tags.\n"
        f"Example: <assessment>CORRECT</assessment> or <assessment>INCORRECT</assessment>"
    )

    if "CodeGemma" in model_name:
        return f"<start_of_turn>user\n{core}\n<end_of_turn>\n<start_of_turn>model\n"
    elif "DeepSeek" in model_name:
        return f"<instruction>\n{core}\n</instruction>\nYour response:"
    else:
        # Code Llama
        return f"[INST] {core} [/INST]"


def extract_confidence_assessment(response_text: str) -> Optional[str]:
    """
    Extract CORRECT or INCORRECT from the assessment response.
    Returns 'CORRECT', 'INCORRECT', or None if unparseable.
    """
    # Try <assessment>CORRECT/INCORRECT</assessment>
    match = re.search(r'<assessment>\s*(CORRECT|INCORRECT)\s*</assessment>', response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: bare CORRECT or INCORRECT anywhere in output
    if re.search(r'\bCORRECT\b', response_text, re.IGNORECASE):
        return 'CORRECT'
    if re.search(r'\bINCORRECT\b', response_text, re.IGNORECASE):
        return 'INCORRECT'

    return None


def run_confidence_scoring(
    task: str,
    solutions: Dict[str, str],          # {model_name: raw_output}
    solution_tags: List[str],            # ['solutionA', 'solutionB', 'solutionC']
    models_loaded: Dict,
    temperature_score: float,
) -> Dict:
    """
    For each solution, ask every model to independently assess if it's correct.
    Returns per-solution assessment details and aggregate confidence scores.
    """
    # Pre-clean solution bodies once
    clean_bodies = {}
    tag_to_model = {}
    for idx, (model_name, raw) in enumerate(solutions.items()):
        tag = solution_tags[idx]
        clean_bodies[tag] = _clean_solution_body(raw)
        tag_to_model[tag] = model_name

    solution_assessments = {}   # tag -> {model_name: {'raw': ..., 'assessment': ...}}
    confidence_scores = defaultdict(int)   # tag -> total CORRECT votes

    for tag in solution_tags:
        if tag not in clean_bodies:
            continue

        solution_assessments[tag] = {}

        for model_name, (tokenizer, model, device) in models_loaded.items():
            # Build a model-specific prompt for each scorer
            prompt = create_confidence_prompt(
                task,
                clean_bodies[tag],
                model_name=model_name,
            )
            raw_assessment = _generate_confidence_assessment(tokenizer, model, device, prompt, temperature_score)
            assessment = extract_confidence_assessment(raw_assessment)

            solution_assessments[tag][model_name] = {
                'raw': raw_assessment,
                'assessment': assessment,
            }

            if assessment == 'CORRECT':
                confidence_scores[tag] += 1

    return {
        'solution_assessments': solution_assessments,
        'confidence_scores': dict(confidence_scores),
        'tag_to_model': tag_to_model,
    }


def _generate_single_assessment(tokenizer, model, device, prompt: str, temperature: float) -> str:
    """Single inference call for a confidence assessment (max_tokens=50 is sufficient)."""
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            prompt_length = inputs["input_ids"].shape[1]
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=1.0 if temperature == 0 else temperature,
                do_sample=False if temperature == 0 else True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            )
            generated_tokens = outputs[0][prompt_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            del inputs, outputs
            torch.cuda.empty_cache()
            return result.strip()
    except Exception as e:
        return f"Assessment error: {e}"


def _generate_confidence_assessment(tokenizer, model, device, prompt: str, temperature: float) -> str:
    """
    Wraps _generate_single_assessment with one retry on empty output.
    """
    for attempt in range(2):
        result = _generate_single_assessment(tokenizer, model, device, prompt, temperature)
        if result.strip():
            return result
    return ""


def select_winner_from_confidence(confidence_scores: Dict[str, int], solution_tags: List[str]) -> Optional[str]:
    """
    Pick the solution tag with the highest confidence score (most CORRECT assessments).
    Ties are broken by tag order (A > B > C) — simple but transparent.
    Returns None if confidence_scores is empty.
    """
    if not confidence_scores:
        return None
    max_score = max(confidence_scores.values())
    # Among tied tags, prefer earlier letter
    for tag in solution_tags:
        if confidence_scores.get(tag, 0) == max_score:
            return tag
    return None


# ─── Main workflow ─────────────────────────────────────────────────────────────

def run_confidence_workflow(
    task_prompt: str,
    test_code: str,
    entry_point: str,
    task_id: str,
    level: str,
    temperature_gen: float,
    temperature_score: float,
    max_tokens: int,
    timeout: int,
    models_loaded: Dict,
) -> Dict:

    solution_tags = ['solutionA', 'solutionB', 'solutionC']

    result = {
        'task_id': task_id,
        'level': level,
        'entry_point': entry_point,
        'prompt': task_prompt,
        'solutions': {},                # model_name -> raw output
        'all_test_results': {},         # model_name -> {passed, error, code_tested}
        'solution_assessments': {},     # tag -> per-model assessments
        'confidence_scores': {},        # solution_tag -> int (CORRECT assessments)
        'winner': None,                 # winning solution tag
        'passed': False,
        'error': None,
        'code_tested': None,
        'num_passing_solutions': 0,
        'scoring_correct': None,
    }

    try:
        # ── Phase 1: Generation ────────────────────────────────────────────────
        for model_name, (tokenizer, model, device) in models_loaded.items():
            gen_prompt = create_generation_prompt(task_prompt, model_name)
            solution = generate_code(tokenizer, model, device, gen_prompt, max_tokens, temperature_gen)
            if solution and not solution.startswith("#"):
                result['solutions'][model_name] = solution

        if len(result['solutions']) == 0:
            result['error'] = "No solutions generated"
            return result

        # ── Phase 2: Test ALL solutions independently ──────────────────────────
        for model_name, solution in result['solutions'].items():
            test_res = run_unit_test(solution, test_code, entry_point, timeout)
            result['all_test_results'][model_name] = {
                'passed': test_res['passed'],
                'error': test_res['error'],
                'code_tested': test_res['code_tested'],
            }

        passing_models = [m for m, r in result['all_test_results'].items() if r['passed']]
        result['num_passing_solutions'] = len(passing_models)

        # ── Phase 3: Independent confidence scoring ────────────────────────────
        available_tags = solution_tags[:len(result['solutions'])]
        confidence_data = run_confidence_scoring(
            task=task_prompt,
            solutions=result['solutions'],
            solution_tags=available_tags,
            models_loaded=models_loaded,
            temperature_score=temperature_score,
        )

        result['solution_assessments'] = confidence_data['solution_assessments']
        result['confidence_scores'] = confidence_data['confidence_scores']
        tag_to_model = confidence_data['tag_to_model']

        # ── Phase 4: Select winner ─────────────────────────────────────────────
        winner_tag = select_winner_from_confidence(result['confidence_scores'], available_tags)
        result['winner'] = winner_tag

        if winner_tag and winner_tag in tag_to_model:
            winner_model = tag_to_model[winner_tag]
            if winner_model in result['all_test_results']:
                winner_test = result['all_test_results'][winner_model]
                result['passed'] = winner_test['passed']
                result['error'] = winner_test['error']
                result['code_tested'] = winner_test['code_tested']

                # Did scoring pick a passing solution when one existed?
                if passing_models:
                    result['scoring_correct'] = winner_model in passing_models
                else:
                    result['scoring_correct'] = None  # No correct option existed
        else:
            result['error'] = "No valid confidence winner determined"

    except Exception as e:
        result['error'] = f"Workflow error: {str(e)}"

    return result


# ─── Benchmark runner ──────────────────────────────────────────────────────────

def benchmark_confidence(
    dataset: pd.DataFrame,
    temperature_gen: float,
    temperature_score: float,
    max_tokens: int,
    timeout: int,
) -> List[Dict]:

    results = []
    total_tasks = len(dataset)

    st.write("### Loading All Models")
    models_loaded = {}
    for model_name, model_path in MODELS.items():
        tokenizer, model, device = load_model_safe(model_name, model_path)
        if tokenizer and model:
            models_loaded[model_name] = (tokenizer, model, device)
            st.success(f"✅ {model_name} loaded")
        else:
            st.error(f"❌ Failed to load {model_name}")
            return []

    progress_bar = st.progress(0)
    status_text = st.empty()
    current_task_ph = st.empty()
    completed_container = st.container()

    for idx, row in dataset.iterrows():
        status_text.text(f"Processing task {idx + 1}/{total_tasks}: {row['task_id']} ({row['level']})")
        with current_task_ph.container():
            st.markdown(f"### ⏳ Task {idx + 1}/{total_tasks}: {row['task_id']} ({row['level']})")

        result = run_confidence_workflow(
            task_prompt=row['prompt'],
            test_code=row['test'],
            entry_point=row['entry_point'],
            task_id=row['task_id'],
            level=row['level'],
            temperature_gen=temperature_gen,
            temperature_score=temperature_score,
            max_tokens=max_tokens,
            timeout=timeout,
            models_loaded=models_loaded,
        )

        status_icon = "✅" if result['passed'] else "❌"

        with completed_container:
            with st.expander(
                f"{status_icon} Task {idx + 1}: {row['task_id']} ({row['level']}) — {'PASS' if result['passed'] else 'FAIL'}",
                expanded=False,
            ):
                # ── Generated solutions ────────────────────────────────────────
                st.markdown("**Generated Solutions:**")
                cols = st.columns(3)
                model_names = list(MODELS.keys())
                solutions = result.get('solutions', {})
                for col_idx, model_name in enumerate(model_names):
                    with cols[col_idx]:
                        st.markdown(f"**{model_name}:**")
                        if model_name in solutions and solutions[model_name].strip():
                            with st.expander("View code", expanded=False):
                                st.code(solutions[model_name], language="python")
                        else:
                            st.error("No output generated")

                # ── Unit test results ──────────────────────────────────────────
                st.markdown("---")
                st.markdown("**Unit Test Results (all solutions):**")
                test_cols = st.columns(3)
                all_test_res = result.get('all_test_results', {})
                for col_idx, model_name in enumerate(model_names):
                    with test_cols[col_idx]:
                        if model_name in all_test_res:
                            tr = all_test_res[model_name]
                            if tr['passed']:
                                st.success(f"✅ PASS")
                            else:
                                st.error(f"❌ FAIL")
                                if tr['error']:
                                    st.caption(tr['error'][:120])
                        else:
                            st.warning("Not tested")

                # ── Confidence assessments ─────────────────────────────────────
                st.markdown("---")
                st.markdown("**Independent Confidence Assessments:**")

                solution_assessments = result.get('solution_assessments', {})
                confidence_scores = result.get('confidence_scores', {})

                for tag in ['solutionA', 'solutionB', 'solutionC']:
                    if tag not in solution_assessments:
                        continue

                    with st.expander(f"Solution {tag[-1].upper()}", expanded=True):
                        assessments = solution_assessments[tag]
                        assess_cols = st.columns(len(assessments))
                        for a_idx, (scorer_name, a_data) in enumerate(assessments.items()):
                            with assess_cols[a_idx]:
                                short_name = scorer_name.split()[0]  # e.g. "DeepSeek"
                                st.markdown(f"**{short_name}**")
                                assessment = a_data.get('assessment')
                                if assessment == 'CORRECT':
                                    st.success("✅ CORRECT")
                                elif assessment == 'INCORRECT':
                                    st.error("❌ INCORRECT")
                                else:
                                    st.warning("⚠️ Invalid")
                                with st.expander("Raw", expanded=False):
                                    st.text(a_data.get('raw', '')[:300])

                # ── Aggregate confidence scores ────────────────────────────────
                st.markdown("---")
                st.markdown("**Aggregate Confidence Scores:**")
                sc_cols = st.columns(3)
                for col_idx, tag in enumerate(['solutionA', 'solutionB', 'solutionC']):
                    with sc_cols[col_idx]:
                        score = confidence_scores.get(tag, 0)
                        max_possible = len(MODELS)   # 3 scorers
                        st.metric(f"Solution {tag[-1].upper()}", f"{score}/{max_possible}")

                # ── Winner and scoring correctness ─────────────────────────────
                st.markdown("---")
                winner_tag = result.get('winner')
                if winner_tag:
                    st.success(f"🏆 Confidence Winner: Solution {winner_tag[-1].upper()}")
                else:
                    st.warning("No winner determined")

                col1, col2 = st.columns(2)
                with col1:
                    num_passing = result.get('num_passing_solutions', 0)
                    st.metric("Passing Solutions", f"{num_passing}/3")
                    if result['passed']:
                        st.success("✅ Winner passed tests!")
                    else:
                        st.error(f"❌ Winner failed: {result.get('error', 'Unknown')[:80]}")

                with col2:
                    sc = result.get('scoring_correct')
                    if sc is True:
                        st.success("✅ Confidence scoring chose a passing solution")
                    elif sc is False:
                        st.error(f"❌ Scoring chose wrong ({num_passing} passing option(s) existed)")
                    else:
                        st.warning("⚠️ No passing solutions available for this task")

                # ── Winning code ───────────────────────────────────────────────
                if result.get('code_tested'):
                    st.markdown("**📝 Winning Code (as tested):**")
                    st.code(result['code_tested'], language="python")

        current_task_ph.empty()
        results.append(result)
        progress_bar.progress((idx + 1) / total_tasks)

    for model_name, (tokenizer, model, device) in models_loaded.items():
        cleanup_model(model, tokenizer)

    status_text.text("✅ Confidence scoring benchmark complete!")
    return results


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_results(results: List[Dict]) -> Dict:
    df = pd.DataFrame(results)

    analysis = {
        'total_tasks': len(df),
        'total_passed': int(df['passed'].sum()),
        'total_failed': int((~df['passed']).sum()),
        'overall_accuracy': float(df['passed'].mean() * 100),
        'by_difficulty': {},
    }

    # Scoring accuracy (tasks where at least one solution existed)
    scorable = df[df['scoring_correct'].notna()]
    if len(scorable) > 0:
        correct = int(scorable['scoring_correct'].sum())
        total = len(scorable)
        analysis['scoring_accuracy'] = correct / total * 100
        analysis['scoring_correct_count'] = correct
        analysis['scoring_total_count'] = total

        # Missed better option
        missed = df[(df['scoring_correct'] == False) & (df['num_passing_solutions'] > 0)]
        analysis['scoring_chose_wrong_when_correct_available'] = len(missed)

        # Passing solution distribution
        dist = df['num_passing_solutions'].value_counts().to_dict()
        analysis['passing_solutions_distribution'] = {
            '0_passing': dist.get(0, 0),
            '1_passing': dist.get(1, 0),
            '2_passing': dist.get(2, 0),
            '3_passing': dist.get(3, 0),
        }

        # Aggregate confidence score distribution across all tasks
        all_confidence_scores = {'solutionA': 0, 'solutionB': 0, 'solutionC': 0}
        for r in results:
            for tag, score in r.get('confidence_scores', {}).items():
                if tag in all_confidence_scores:
                    all_confidence_scores[tag] += score
        analysis['aggregate_confidence_distribution'] = all_confidence_scores

    # By difficulty
    for level in ['easy', 'middle', 'hard']:
        level_df = df[df['level'] == level]
        if len(level_df) > 0:
            analysis['by_difficulty'][level] = {
                'total': len(level_df),
                'passed': int(level_df['passed'].sum()),
                'failed': int((~level_df['passed']).sum()),
                'accuracy': float(level_df['passed'].mean() * 100),
            }

    return analysis


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Independent Confidence Scoring Benchmark", layout="wide")

    st.title("🎯 Independent Confidence Scoring Benchmark")
    st.markdown(
        "**Workflow:** All models generate → each solution evaluated independently → "
        "solution with highest confidence score selected"
    )
    st.markdown(
        "**Why independent scoring?** Presenting all solutions together triggers positional bias. "
        "Independent assessment eliminates this entirely—no solution is compared directly to others."
    )
    st.markdown(f"**Assessments per task:** {len(MODELS)} solutions × {len(MODELS)} scorers = {len(MODELS) * len(MODELS)} total assessments")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configuration")

    st.sidebar.markdown("### Generation Phase")
    temperature_gen = st.sidebar.slider("Temperature (Generation)", 0.0, 1.0, 0.3, 0.1)

    st.sidebar.markdown("### Scoring Phase")
    temperature_score = st.sidebar.slider("Temperature (Scoring)", 0.0, 1.0, 0.0, 0.1)
    st.sidebar.caption(
        "Temperature 0.0 = deterministic scoring (recommended). "
        "Higher values introduce sampling noise into assessments."
    )

    max_tokens = st.sidebar.slider("Max Tokens (Generation)", 256, 2048, 1024, 128)
    timeout = st.sidebar.slider("Test Timeout (seconds)", 10, 120, 40, 5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 GPU Memory")
    if torch.cuda.is_available():
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            reserved_mem = torch.cuda.memory_reserved(0) / 1e9
            allocated_mem = torch.cuda.memory_allocated(0) / 1e9
            st.sidebar.text(f"Total:     {total_mem:.1f} GB")
            st.sidebar.text(f"Reserved:  {reserved_mem:.1f} GB")
            st.sidebar.text(f"Allocated: {allocated_mem:.1f} GB")
        except Exception:
            st.sidebar.text("GPU info unavailable")

    # ── Dataset ────────────────────────────────────────────────────────────────
    st.header("📊 Dataset")
    try:
        dataset = load_dataset()
        st.success(f"Loaded {len(dataset)} tasks from McEval")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Easy", len(dataset[dataset['level'] == 'easy']))
        with col2:
            st.metric("Middle", len(dataset[dataset['level'] == 'middle']))
        with col3:
            st.metric("Hard", len(dataset[dataset['level'] == 'hard']))
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    # ── Run ────────────────────────────────────────────────────────────────────
    if st.button("🚀 Run Confidence Scoring Benchmark", type="primary"):
        st.header("🔬 Running Benchmark")

        results = benchmark_confidence(
            dataset=dataset,
            temperature_gen=temperature_gen,
            temperature_score=temperature_score,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        if not results:
            return

        analysis = analyze_results(results)

        # Overall accuracy
        st.header("📊 Overall Performance")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tasks", analysis['total_tasks'])
        with col2:
            st.metric("Passed", analysis['total_passed'])
        with col3:
            st.metric("Failed", analysis['total_failed'])
        with col4:
            st.metric("Accuracy", f"{analysis['overall_accuracy']:.1f}%")

        # Scoring accuracy
        if 'scoring_accuracy' in analysis:
            st.header("🎯 Confidence Scoring Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Scoring Accuracy", f"{analysis['scoring_accuracy']:.1f}%",
                          help="Of tasks where ≥1 correct solution existed, how often did confidence scoring pick it?")
            with col2:
                st.metric("Correct Selections",
                          f"{analysis['scoring_correct_count']}/{analysis['scoring_total_count']}")
            with col3:
                st.metric("Missed Better Options",
                          analysis.get('scoring_chose_wrong_when_correct_available', 0))

            # Aggregate confidence distribution — key bias check
            if 'aggregate_confidence_distribution' in analysis:
                st.markdown("**Aggregate confidence score distribution across all tasks (bias check):**")
                st.caption(
                    "In three-way voting, Solution B received only 11.8% of votes due to middle-position bias. "
                    "Independent scoring should produce roughly equal distributions if unbiased."
                )
                acd = analysis['aggregate_confidence_distribution']
                total_scores = sum(acd.values()) or 1
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Solution A score", acd.get('solutionA', 0),
                              delta=f"{acd.get('solutionA', 0)/total_scores*100:.1f}%")
                with sc2:
                    st.metric("Solution B score", acd.get('solutionB', 0),
                              delta=f"{acd.get('solutionB', 0)/total_scores*100:.1f}%")
                with sc3:
                    st.metric("Solution C score", acd.get('solutionC', 0),
                              delta=f"{acd.get('solutionC', 0)/total_scores*100:.1f}%")

            # Passing solution distribution
            if 'passing_solutions_distribution' in analysis:
                st.markdown("**Passing solutions per task:**")
                dist = analysis['passing_solutions_distribution']
                dc1, dc2, dc3, dc4 = st.columns(4)
                with dc1:
                    st.metric("0/3 pass", dist['0_passing'])
                with dc2:
                    st.metric("1/3 pass", dist['1_passing'])
                with dc3:
                    st.metric("2/3 pass", dist['2_passing'])
                with dc4:
                    st.metric("3/3 pass", dist['3_passing'])

        # By difficulty
        st.header("📈 Performance by Difficulty")
        for level in ['easy', 'middle', 'hard']:
            if level in analysis['by_difficulty']:
                stats = analysis['by_difficulty'][level]
                with st.expander(f"{level.upper()} — {stats['accuracy']:.1f}%", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Total", stats['total'])
                    with c2:
                        st.metric("Passed", stats['passed'])
                    with c3:
                        st.metric("Failed", stats['failed'])

        # Export
        st.header("💾 Export Results")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        def _serialise(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: _serialise(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_serialise(i) for i in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'item'):
                return obj.item()
            try:
                if pd.isna(obj):
                    return None
            except Exception:
                pass
            return obj

        export_data = {
            'workflow': 'Independent Confidence Scoring (Generate → Independent Assessment → Aggregate Scores)',
            'models': list(MODELS.keys()),
            'timestamp': timestamp,
            'temperature_gen': float(temperature_gen),
            'temperature_score': float(temperature_score),
            'max_tokens': int(max_tokens),
            'timeout': int(timeout),
            'analysis': _serialise(analysis),
            'results': _serialise(results),
        }

        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"confidence_scoring_benchmark_{timestamp}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()