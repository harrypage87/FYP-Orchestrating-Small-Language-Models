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
Pairwise Voting Benchmark
Each pair of solutions (S1,S2), (S1,S3), (S2,S3) is evaluated in a separate independent prompt.
The solution with the most aggregate wins across all pairwise comparisons is selected.
This eliminates the middle-position bias observed in 3-way voting, since every
comparison is a binary choice with no "middle" candidate.

Model order: first=A, second=B, third=C
'''
MODELS = {
    "CodeGemma 7B Instruct": "google/codegemma-7b-it",
    "Code Llama 7B Instruct": "codellama/CodeLlama-7b-Instruct-hf",
    "DeepSeek Coder 7B Instruct": "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
}

DATASET_PATH = "/home/demouser/Desktop/121336311/McEval DataSet/McEval_Generation_Tasks.csv"

# Round-robin pairwise comparisons: each solution appears in first position once,
# second position once, and sits out once. This prevents any single solution
# from benefiting disproportionately from primacy bias across comparisons.
PAIRS = [
    ("solutionA", "solutionB"),  # A first, B second
    ("solutionB", "solutionC"),  # B first, C second
    ("solutionC", "solutionA"),  # C first, A second
]


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    return df.head(50)


# ─── Test execution (unchanged from original) ─────────────────────────────────

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


# ─── Model loading / cleanup (unchanged) ──────────────────────────────────────

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


# ─── Generation (unchanged) ───────────────────────────────────────────────────

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


# ─── Pairwise voting (new) ─────────────────────────────────────────────────────

def _clean_solution_body(raw: str) -> str:
    """Strip <solution> wrapper tags so we expose only the code body."""
    s = raw
    if '</solution>' in s:
        s = s.split('</solution>')[0]
    if '<solution>' in s:
        s = s.split('<solution>')[-1]
    return s.strip()


def create_pairwise_prompt(task: str, tag_a: str, code_a: str, tag_b: str, code_b: str, model_name: str = "") -> str:
    """
    Build an independent binary comparison prompt for exactly two solutions.
    No third option exists, so there is no middle position.
    Each model receives its native chat template to maximise valid response rate.
    """
    label_a = tag_a[-1].upper()
    label_b = tag_b[-1].upper()

    core = (
        f"Compare the two candidate solutions below and decide which one better implements the task.\n"
        f"Evaluate based on: correctness, completeness, edge-case handling, and code quality.\n"
        f"Respond with ONLY the tag name of the better solution, wrapped in <best> tags.\n"
        f"Example: <best>solution{label_a}</best>  OR  <best>solution{label_b}</best>\n\n"
        f"<task>\n{task}\n</task>\n\n"
        f"<solution{label_a}>\n{code_a}\n</solution{label_a}>\n\n"
        f"<solution{label_b}>\n{code_b}\n</solution{label_b}>"
    )

    if "CodeGemma" in model_name:
        return f"<start_of_turn>user\n{core}\n<end_of_turn>\n<start_of_turn>model\n"
    elif "DeepSeek" in model_name:
        return f"<instruction>\n{core}\n</instruction>\nYour response:"
    else:
        # Code Llama
        return f"[INST] {core} [/INST]"


def extract_pairwise_winner(vote_text: str, tag_a: str, tag_b: str) -> Optional[str]:
    """
    Return the winning tag (e.g. 'solutionA') from a binary vote response,
    or None if the response is unparseable.
    """
    # Try <best>solutionX</best>
    match = re.search(r'<best>\s*(solution[ABC])\s*</best>', vote_text, re.IGNORECASE)
    if match:
        candidate = "solution" + match.group(1)[-1].upper()
        if candidate in (tag_a, tag_b):
            return candidate

    # Fallback: bare solutionX anywhere in output
    match = re.search(r'solution([ABC])', vote_text, re.IGNORECASE)
    if match:
        candidate = "solution" + match.group(1).upper()
        if candidate in (tag_a, tag_b):
            return candidate

    return None


def run_pairwise_votes(
    task: str,
    solutions: Dict[str, str],          # {model_name: raw_output}
    solution_tags: List[str],            # ['solutionA', 'solutionB', 'solutionC']
    models_loaded: Dict,
    temperature_vote: float,
) -> Dict:
    """
    For every pair in PAIRS, ask every model to vote in a separate binary prompt.
    Returns per-pair vote details and aggregate win counts per solution tag.
    """
    # Pre-clean solution bodies once
    clean_bodies = {}
    tag_to_model = {}
    for idx, (model_name, raw) in enumerate(solutions.items()):
        tag = solution_tags[idx]
        clean_bodies[tag] = _clean_solution_body(raw)
        tag_to_model[tag] = model_name

    pair_results = {}   # (tag_a, tag_b) -> {model_name: {'raw': ..., 'winner': ...}}
    win_counts = defaultdict(int)   # tag -> total wins across all pairs and voters

    for tag_a, tag_b in PAIRS:
        if tag_a not in clean_bodies or tag_b not in clean_bodies:
            continue

        pair_key = f"{tag_a}_vs_{tag_b}"
        pair_results[pair_key] = {}

        for model_name, (tokenizer, model, device) in models_loaded.items():
            # Build a model-specific prompt for each voter so native chat
            # templates are respected — critical for CodeGemma validity rate.
            prompt = create_pairwise_prompt(
                task,
                tag_a, clean_bodies[tag_a],
                tag_b, clean_bodies[tag_b],
                model_name=model_name,
            )
            raw_vote = _generate_pairwise_vote(tokenizer, model, device, prompt, temperature_vote)
            winner = extract_pairwise_winner(raw_vote, tag_a, tag_b)

            pair_results[pair_key][model_name] = {
                'raw': raw_vote,
                'winner': winner,
                'prompt_order': [tag_a, tag_b],   # preserved for bias analysis
            }

            if winner:
                win_counts[winner] += 1

    return {
        'pair_results': pair_results,
        'win_counts': dict(win_counts),
        'tag_to_model': tag_to_model,
    }


def _generate_single_vote(tokenizer, model, device, prompt: str, temperature: float) -> str:
    """Single inference call for a pairwise vote (max_tokens=50 is sufficient)."""
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
        return f"Voting error: {e}"


def _generate_pairwise_vote(tokenizer, model, device, prompt: str, temperature: float) -> str:
    """
    Wraps _generate_single_vote with one retry on empty output.
    Note: with temperature=0 (deterministic), the retry will produce the same
    result as the first attempt. It serves as a safety net for intermittent
    empty outputs rather than a reliable fix for systematic failures.
    """
    for attempt in range(2):
        result = _generate_single_vote(tokenizer, model, device, prompt, temperature)
        if result.strip():
            return result
    return ""


def select_winner_from_pairwise(win_counts: Dict[str, int], solution_tags: List[str]) -> Optional[str]:
    """
    Pick the solution tag with the most aggregate pairwise wins.
    Ties are broken by tag order (A > B > C) — a simple but transparent rule.
    Returns None if win_counts is empty.
    """
    if not win_counts:
        return None
    max_wins = max(win_counts.values())
    # Among tied tags, prefer earlier letter (most conservative choice)
    for tag in solution_tags:
        if win_counts.get(tag, 0) == max_wins:
            return tag
    return None


# ─── Main workflow ─────────────────────────────────────────────────────────────

def run_pairwise_workflow(
    task_prompt: str,
    test_code: str,
    entry_point: str,
    task_id: str,
    level: str,
    temperature_gen: float,
    temperature_vote: float,
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
        'pairwise_votes': {},           # pair_key -> per-model votes
        'win_counts': {},               # solution_tag -> int
        'winner': None,                 # winning solution tag
        'passed': False,
        'error': None,
        'code_tested': None,
        'num_passing_solutions': 0,
        'voting_correct': None,
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

        # ── Phase 3: Pairwise voting ───────────────────────────────────────────
        # Only run if we have all 3 solutions; gracefully skip missing ones
        available_tags = solution_tags[:len(result['solutions'])]
        pairwise_data = run_pairwise_votes(
            task=task_prompt,
            solutions=result['solutions'],
            solution_tags=available_tags,
            models_loaded=models_loaded,
            temperature_vote=temperature_vote,
        )

        result['pairwise_votes'] = pairwise_data['pair_results']
        result['win_counts'] = pairwise_data['win_counts']
        tag_to_model = pairwise_data['tag_to_model']

        # ── Phase 4: Select winner ─────────────────────────────────────────────
        winner_tag = select_winner_from_pairwise(result['win_counts'], available_tags)
        result['winner'] = winner_tag

        if winner_tag and winner_tag in tag_to_model:
            winner_model = tag_to_model[winner_tag]
            if winner_model in result['all_test_results']:
                winner_test = result['all_test_results'][winner_model]
                result['passed'] = winner_test['passed']
                result['error'] = winner_test['error']
                result['code_tested'] = winner_test['code_tested']

                # Did voting pick a passing solution when one existed?
                if passing_models:
                    result['voting_correct'] = winner_model in passing_models
                else:
                    result['voting_correct'] = None  # No correct option existed
        else:
            result['error'] = "No valid pairwise winner determined"

    except Exception as e:
        result['error'] = f"Workflow error: {str(e)}"

    return result


# ─── Benchmark runner ──────────────────────────────────────────────────────────

def benchmark_pairwise(
    dataset: pd.DataFrame,
    temperature_gen: float,
    temperature_vote: float,
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

        result = run_pairwise_workflow(
            task_prompt=row['prompt'],
            test_code=row['test'],
            entry_point=row['entry_point'],
            task_id=row['task_id'],
            level=row['level'],
            temperature_gen=temperature_gen,
            temperature_vote=temperature_vote,
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

                # ── Pairwise vote breakdown ────────────────────────────────────
                st.markdown("---")
                st.markdown("**Pairwise Vote Results:**")

                pairwise_votes = result.get('pairwise_votes', {})
                win_counts = result.get('win_counts', {})

                for pair_key, voter_results in pairwise_votes.items():
                    # e.g. "solutionA_vs_solutionB"
                    parts = pair_key.split('_vs_')
                    tag_a, tag_b = parts[0], parts[1]
                    label = f"{tag_a[-1].upper()} vs {tag_b[-1].upper()}"

                    with st.expander(f"Pair: {label}", expanded=True):
                        vote_cols = st.columns(len(voter_results))
                        for v_idx, (voter_name, v_data) in enumerate(voter_results.items()):
                            with vote_cols[v_idx]:
                                short_name = voter_name.split()[0]  # e.g. "DeepSeek"
                                st.markdown(f"**{short_name}**")
                                winner = v_data.get('winner')
                                if winner:
                                    st.success(f"→ {winner[-1].upper()}")
                                else:
                                    st.error("Invalid")
                                with st.expander("Raw", expanded=False):
                                    st.text(v_data.get('raw', '')[:300])

                # ── Aggregate win counts ───────────────────────────────────────
                st.markdown("---")
                st.markdown("**Aggregate Win Counts:**")
                wc_cols = st.columns(3)
                for col_idx, tag in enumerate(['solutionA', 'solutionB', 'solutionC']):
                    with wc_cols[col_idx]:
                        wins = win_counts.get(tag, 0)
                        max_possible = len(PAIRS) * len(MODELS)   # 3 pairs × 3 voters = 9
                        st.metric(f"Solution {tag[-1].upper()}", f"{wins}/{max_possible}")

                # ── Winner and voting correctness ──────────────────────────────
                st.markdown("---")
                winner_tag = result.get('winner')
                if winner_tag:
                    st.success(f"🏆 Pairwise Winner: Solution {winner_tag[-1].upper()}")
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
                    vc = result.get('voting_correct')
                    if vc is True:
                        st.success("✅ Pairwise voting chose a passing solution")
                    elif vc is False:
                        st.error(f"❌ Voting chose wrong ({num_passing} passing option(s) existed)")
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

    status_text.text("✅ Pairwise benchmark complete!")
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

    # Voting accuracy (tasks where at least one solution existed)
    votable = df[df['voting_correct'].notna()]
    if len(votable) > 0:
        correct = int(votable['voting_correct'].sum())
        total = len(votable)
        analysis['voting_accuracy'] = correct / total * 100
        analysis['voting_correct_count'] = correct
        analysis['voting_total_count'] = total

        # Missed better option
        missed = df[(df['voting_correct'] == False) & (df['num_passing_solutions'] > 0)]
        analysis['voting_chose_wrong_when_correct_available'] = len(missed)

        # Passing solution distribution
        dist = df['num_passing_solutions'].value_counts().to_dict()
        analysis['passing_solutions_distribution'] = {
            '0_passing': dist.get(0, 0),
            '1_passing': dist.get(1, 0),
            '2_passing': dist.get(2, 0),
            '3_passing': dist.get(3, 0),
        }

        # Aggregate win count distribution across all tasks
        all_win_counts = {'solutionA': 0, 'solutionB': 0, 'solutionC': 0}
        for r in results:
            for tag, wins in r.get('win_counts', {}).items():
                if tag in all_win_counts:
                    all_win_counts[tag] += wins
        analysis['aggregate_win_distribution'] = all_win_counts

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
    st.set_page_config(page_title="Pairwise Voting Benchmark", layout="wide")

    st.title("⚖️ Pairwise Voting Benchmark")
    st.markdown(
        "**Workflow:** All models generate → each pair evaluated in a separate binary prompt → "
        "solution with most aggregate wins selected"
    )
    st.markdown(
        "**Why pairwise?** Three-way voting exhibits severe middle-position bias (Solution B selected "
        "only ~11% of the time). Binary comparisons eliminate the middle position entirely."
    )
    st.markdown(f"**Pairs evaluated:** {', '.join(f'{a[-1].upper()} vs {b[-1].upper()}' for a, b in PAIRS)}")
    st.markdown(f"**Votes per task:** {len(PAIRS)} pairs × {len(MODELS)} voters = {len(PAIRS) * len(MODELS)} total votes")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Configuration")

    st.sidebar.markdown("### Generation Phase")
    temperature_gen = st.sidebar.slider("Temperature (Generation)", 0.0, 1.0, 0.3, 0.1)

    st.sidebar.markdown("### Voting Phase")
    temperature_vote = st.sidebar.slider("Temperature (Voting)", 0.0, 1.0, 0.0, 0.1)
    st.sidebar.caption(
        "Temperature 0.0 = deterministic voting (recommended). "
        "Higher values introduce sampling noise into pairwise decisions."
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
    if st.button("🚀 Run Pairwise Voting Benchmark", type="primary"):
        st.header("🔬 Running Benchmark")

        results = benchmark_pairwise(
            dataset=dataset,
            temperature_gen=temperature_gen,
            temperature_vote=temperature_vote,
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

        # Voting accuracy
        if 'voting_accuracy' in analysis:
            st.header("⚖️ Pairwise Voting Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Voting Accuracy", f"{analysis['voting_accuracy']:.1f}%",
                          help="Of tasks where ≥1 correct solution existed, how often did pairwise voting pick it?")
            with col2:
                st.metric("Correct Selections",
                          f"{analysis['voting_correct_count']}/{analysis['voting_total_count']}")
            with col3:
                st.metric("Missed Better Options",
                          analysis.get('voting_chose_wrong_when_correct_available', 0))

            # Aggregate win distribution — key bias check
            if 'aggregate_win_distribution' in analysis:
                st.markdown("**Aggregate win distribution across all tasks (bias check):**")
                st.caption(
                    "In three-way voting, Solution B received only 11.8% of votes. "
                    "A fair pairwise system should produce roughly equal wins across A, B, C."
                )
                awd = analysis['aggregate_win_distribution']
                total_wins = sum(awd.values()) or 1
                wc1, wc2, wc3 = st.columns(3)
                with wc1:
                    st.metric("Solution A wins", awd.get('solutionA', 0),
                              delta=f"{awd.get('solutionA', 0)/total_wins*100:.1f}%")
                with wc2:
                    st.metric("Solution B wins", awd.get('solutionB', 0),
                              delta=f"{awd.get('solutionB', 0)/total_wins*100:.1f}%")
                with wc3:
                    st.metric("Solution C wins", awd.get('solutionC', 0),
                              delta=f"{awd.get('solutionC', 0)/total_wins*100:.1f}%")

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
            'workflow': 'Pairwise Voting (Generate → Binary Pair Votes → Aggregate Wins)',
            'models': list(MODELS.keys()),
            'pairs': [f"{a}_vs_{b}" for a, b in PAIRS],
            'timestamp': timestamp,
            'temperature_gen': float(temperature_gen),
            'temperature_vote': float(temperature_vote),
            'max_tokens': int(max_tokens),
            'timeout': int(timeout),
            'analysis': _serialise(analysis),
            'results': _serialise(results),
        }

        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"pairwise_voting_benchmark_{timestamp}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()