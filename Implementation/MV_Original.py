import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from typing import Dict, List
import gc
from datetime import datetime
import json
import multiprocessing
from multiprocessing import Process, Queue
from collections import Counter

# Model configurations
MODELS = {
    "DeepSeek Coder 7B Instruct": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "CodeGemma 7B Instruct": "google/codegemma-7b-it",
    "Code Llama 7B Instruct": "codellama/CodeLlama-7b-Instruct-hf"
}

DATASET_PATH = "/home/demouser/Desktop/121336311/McEval_Generation_Tasks.csv"

def load_dataset() -> pd.DataFrame:
    """Load the McEval benchmark dataset - first 50 tasks"""
    df = pd.read_csv(DATASET_PATH)
    return df.head(50)

def _execute_test_in_process(code_to_test: str, test_code: str, entry_point: str, result_queue: Queue):
    """Execute test in a separate process"""
    result = {
        'passed': False,
        'error': None,
        'code_tested': code_to_test
    }
    
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
            error_msg = str(e) if str(e) else "Assertion failed (no message)"
            result['error'] = f"Test assertion failed: {error_msg}"
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
    """Run unit tests with timeout - exact extraction from CodeGemma script"""
    result = {
        'passed': False,
        'error': None,
        'code_tested': None
    }
    
    try:
        code_to_test = code.strip()
        
        # Extract from <solution> tags if present
        solution_match = re.search(r'<solution>(.*?)</solution>', code_to_test, re.DOTALL)
        if solution_match:
            # ONLY take what's inside the tags, nothing after </solution>
            code_to_test = solution_match.group(1).strip()
        else:
            # Try to extract if only closing tag present (Code Llama case)
            if '</solution>' in code_to_test:
                code_to_test = code_to_test.split('</solution>')[0].strip()
            else:
                # No solution tags found at all
                result['error'] = "No <solution> tags found in model output"
                return result
        
        # Remove <end_of_turn> if model added it
        if '<end_of_turn>' in code_to_test:
            code_to_test = code_to_test.split('<end_of_turn>')[0].strip()
        
        # Remove >>> prefix if present
        if code_to_test.startswith('>>>'):
            code_to_test = code_to_test[3:].strip()
        
        # Clean markdown fences at the start
        if code_to_test.startswith('```python'):
            code_to_test = code_to_test[9:].strip()
        elif code_to_test.startswith('```'):
            code_to_test = code_to_test[3:].strip()
        
        # Clean markdown fence at the end
        if code_to_test.endswith('```'):
            code_to_test = code_to_test[:-3].strip()
        
        # Remove any remaining inline markdown fences
        code_to_test = re.sub(r'```python\s*\n?', '\n', code_to_test)
        code_to_test = re.sub(r'```\s*\n?', '\n', code_to_test)
        
        # Remove lines that are just fence markers
        lines = code_to_test.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped in ['```', '```python', '```py']:
                continue
            cleaned_lines.append(line)
        code_to_test = '\n'.join(cleaned_lines)
        
        result['code_tested'] = code_to_test
        
        # Check if code is empty
        if not code_to_test or len(code_to_test.strip()) == 0:
            result['error'] = "Generated code is empty after extraction"
            return result
        
        # Create queue and process for subprocess execution
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

def load_model_safe(model_name: str, model_path: str):
    """Load model with proper configuration"""
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
            device_map=device,  # Use device_map instead of .to()
            low_cpu_mem_usage=True,  # Changed to True for better memory handling
        )
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None, None

def cleanup_model(model, tokenizer):
    """Clean up model and free GPU memory"""
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_code(tokenizer, model, device, prompt, max_tokens, temperature):
    """Generate code from a model."""
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
            
            # Extract only generated tokens
            generated_tokens = outputs[0][prompt_length:]
            
            # Check if model generated nothing
            if len(generated_tokens) == 0:
                return ""  # Return empty string, not error
            
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            del inputs
            del outputs
            torch.cuda.empty_cache()
            
            return result.strip()
    except Exception as e:
        st.error(f"Generation exception: {str(e)}")
        return f"# Generation error: {e}"

def generate_vote(tokenizer, model, device, voting_prompt, max_tokens, temperature):
    """Generate a vote from a model."""
    try:
        with torch.no_grad():
            inputs = tokenizer(voting_prompt, return_tensors="pt").to(device)
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
            
            # Extract only generated tokens
            generated_tokens = outputs[0][prompt_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return result.strip()
    except Exception as e:
        return f"Voting error: {e}"

def create_generation_prompt(task_prompt: str, model_name: str) -> str:
    """Create prompt for code generation - model-specific formats"""
    
    if "CodeGemma" in model_name:
        # CodeGemma uses Gemma chat format
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
        # DeepSeek - exact format from solo script
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
        # Code Llama - uses [INST] format
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

def create_voting_prompt(task: str, solutions: Dict[str, str]) -> str:
    """Create voting prompt with solutions wrapped in XML tags"""
    candidate_letters = ['A', 'B', 'C']
    
    prompt = f"""<instruction>
<task>
{task}
</task>
Among the {len(solutions)} solutions below, which solution best implements the task?
Respond with ONLY the solution tag name (solutionA, solutionB, or solutionC).

Encapsulate your answer in <best> tags:
>>> <best>solutionX</best>
</instruction>

<allSolutions>
"""
    
    for idx, (model_name, solution) in enumerate(solutions.items()):
        letter = candidate_letters[idx]
        prompt += f"<solution{letter}>\n{solution}\n</solution{letter}>\n\n"
    
    prompt += "</allSolutions>\n\nYour response:"
    
    return prompt

def extract_vote(vote_text: str, valid_solutions: List[str]) -> str:
    """Extract vote from model output"""
    # Try <best>X</best>
    match = re.search(r'<best>(.*?)</best>', vote_text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip().lower()
        if extracted in [v.lower() for v in valid_solutions]:
            return extracted
    
    # Try solutionX anywhere
    match = re.search(r'solution([ABC])', vote_text, re.IGNORECASE)
    if match:
        normalized = f"solution{match.group(1).upper()}"
        if normalized in valid_solutions:
            return normalized
    
    return None

def run_voting_workflow(
    task_prompt: str,
    test_code: str,
    entry_point: str,
    task_id: str,
    level: str,
    temperature_gen: float,
    temperature_vote: float,
    max_tokens: int,
    timeout: int,
    models_loaded: Dict
) -> Dict:
    """Run voting workflow: all models generate, all models vote"""
    
    result = {
        'task_id': task_id,
        'level': level,
        'entry_point': entry_point,
        'prompt': task_prompt,
        'solutions': {},
        'votes': {},
        'winner': None,
        'passed': False,
        'error': None,
        'code_tested': None
    }
    
    try:
        # Phase 1: Generate solutions from all models
        for model_name, (tokenizer, model, device) in models_loaded.items():
            generation_prompt = create_generation_prompt(task_prompt, model_name)
            solution = generate_code(tokenizer, model, device, generation_prompt, max_tokens, temperature_gen)
            
            if solution and not solution.startswith("#"):
                result['solutions'][model_name] = solution
        
        if len(result['solutions']) == 0:
            result['error'] = "No solutions generated"
            return result
        
        # Phase 2: Create voting prompt
        voting_prompt = create_voting_prompt(task_prompt, result['solutions'])
        valid_solutions = ['solutionA', 'solutionB', 'solutionC']
        
        # Phase 3: All models vote
        for model_name, (tokenizer, model, device) in models_loaded.items():
            vote_output = generate_vote(tokenizer, model, device, voting_prompt, 50, temperature_vote)
            if vote_output and not vote_output.startswith("Voting error"):
                extracted_vote = extract_vote(vote_output, valid_solutions)
                result['votes'][model_name] = {
                    'raw': vote_output,
                    'extracted': extracted_vote
                }
        
        # Phase 4: Test ALL solutions (not just winner)
        solution_test_results = {}
        for model_name, solution in result['solutions'].items():
            test_result = run_unit_test(solution, test_code, entry_point, timeout)
            solution_test_results[model_name] = {
                'passed': test_result['passed'],
                'error': test_result['error'],
                'code_tested': test_result['code_tested']
            }
        
        result['all_test_results'] = solution_test_results
        
        # Count valid votes
        valid_votes = [v['extracted'] for v in result['votes'].values() if v['extracted']]
        
        # Phase 5: Determine voting accuracy
        if valid_votes:
            vote_counts = Counter(valid_votes)
            winner_solution_tag = vote_counts.most_common(1)[0][0]
            result['winner'] = winner_solution_tag
            
            # Map solution tag to model - use lowercase keys
            solution_map = {
                'solutiona': list(result['solutions'].keys())[0] if len(result['solutions']) > 0 else None,
                'solutionb': list(result['solutions'].keys())[1] if len(result['solutions']) > 1 else None,
                'solutionc': list(result['solutions'].keys())[2] if len(result['solutions']) > 2 else None,
            }
            
            winner_model = solution_map.get(winner_solution_tag.lower())
            
            if winner_model and winner_model in solution_test_results:
                # Check if winner passed
                result['passed'] = solution_test_results[winner_model]['passed']
                result['error'] = solution_test_results[winner_model]['error']
                result['code_tested'] = solution_test_results[winner_model]['code_tested']
                
                # Calculate voting correctness
                # Did voting pick a passing solution when one existed?
                passing_solutions = [m for m, r in solution_test_results.items() if r['passed']]
                result['num_passing_solutions'] = len(passing_solutions)
                result['voting_correct'] = winner_model in passing_solutions if passing_solutions else None
                
            else:
                result['error'] = f"Could not find winner model for {winner_solution_tag}"
        else:
            result['error'] = "No valid votes"
        
    except Exception as e:
        result['error'] = f"Workflow error: {str(e)}"
    
    return result

def benchmark_voting(
    dataset: pd.DataFrame,
    temperature_gen: float,
    temperature_vote: float,
    max_tokens: int,
    timeout: int
) -> List[Dict]:
    """Benchmark voting workflow on dataset"""
    
    results = []
    total_tasks = len(dataset)
    
    # Load all models once
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
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_task_placeholder = st.empty()
    completed_tasks_container = st.container()
    
    for idx, row in dataset.iterrows():
        status_text.text(f"Processing task {idx + 1}/{total_tasks}: {row['task_id']} ({row['level']})")
        
        with current_task_placeholder.container():
            st.markdown(f"### Task {idx + 1}/{total_tasks}: {row['task_id']} ({row['level']})")
        
        # Run workflow
        result = run_voting_workflow(
            task_prompt=row['prompt'],
            test_code=row['test'],
            entry_point=row['entry_point'],
            task_id=row['task_id'],
            level=row['level'],
            temperature_gen=temperature_gen,
            temperature_vote=temperature_vote,
            max_tokens=max_tokens,
            timeout=timeout,
            models_loaded=models_loaded
        )
        
        # Display result
        status_icon = "✅" if result['passed'] else "❌"
        with completed_tasks_container:
            with st.expander(
                f"{status_icon} Task {idx + 1}: {row['task_id']} ({row['level']}) - {'PASS' if result['passed'] else 'FAIL'}",
                expanded=False
            ):
                # Show solutions from each model - always show all 3 models
                st.markdown("**Generated Solutions:**")
                cols = st.columns(3)
                
                model_names = list(MODELS.keys())
                solutions = result.get('solutions', {})
                
                for col_idx, model_name in enumerate(model_names):
                    with cols[col_idx]:
                        st.markdown(f"**{model_name}:**")
                        if model_name in solutions:
                            solution = solutions[model_name]
                            if solution and solution.strip():
                                with st.expander("View code", expanded=False):
                                    st.code(solution, language="python")
                            else:
                                st.warning("Empty output")
                        else:
                            st.error("No output generated")
                
                # Show votes - always show all 3 models
                st.markdown("---")
                st.markdown("**Votes:**")
                vote_cols = st.columns(3)
                votes = result.get('votes', {})
                
                for col_idx, model_name in enumerate(model_names):
                    with vote_cols[col_idx]:
                        st.write(f"**{model_name}:**")
                        if model_name in votes:
                            vote_data = votes[model_name]
                            extracted = vote_data.get('extracted', 'Invalid')
                            if extracted and extracted.lower() != 'invalid':
                                st.success(extracted)
                            else:
                                st.error("Invalid vote")
                            with st.expander("Raw vote"):
                                st.text(vote_data.get('raw', ''))
                        else:
                            st.error("No vote")
                
                # Show winner and test result
                st.markdown("---")
                
                if result.get('winner'):
                    st.success(f" Winner: {result['winner']}")
                
                # Show test results for ALL solutions
                st.markdown("**Test Results for All Solutions:**")
                all_results = result.get('all_test_results', {})
                test_cols = st.columns(3)
                
                for idx, model_name in enumerate(model_names):
                    with test_cols[idx]:
                        if model_name in all_results:
                            test_res = all_results[model_name]
                            if test_res['passed']:
                                st.success(f"✅ {model_name}: PASS")
                            else:
                                st.error(f"❌ {model_name}: FAIL")
                        else:
                            st.warning(f"{model_name}: Not tested")
                
                # Show voting correctness
                st.markdown("---")
                num_passing = result.get('num_passing_solutions', 0)
                voting_correct = result.get('voting_correct')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Passing Solutions", f"{num_passing}/3")
                    
                    if result['passed']:
                        st.success("✅ Winner passed tests!")
                    else:
                        st.error(f"❌ Winner failed: {result.get('error', 'Unknown')}")
                
                with col2:
                    if voting_correct is not None:
                        if voting_correct:
                            st.success("✅ Voting chose a passing solution")
                        else:
                            if num_passing > 0:
                                st.error(f"❌ Voting chose wrong (had {num_passing} passing options)")
                            else:
                                st.warning("No passing solutions available")
                    
                # Show tested code prominently
                if result.get('code_tested'):
                    st.markdown("**Winning Code That Was Tested:**")
                    st.code(result['code_tested'], language="python")
        
        current_task_placeholder.empty()
        results.append(result)
        progress_bar.progress((idx + 1) / total_tasks)
    
    # Cleanup all models
    for model_name, (tokenizer, model, device) in models_loaded.items():
        cleanup_model(model, tokenizer)
    
    status_text.text("✅ Benchmark complete!")
    return results

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze benchmark results"""
    df = pd.DataFrame(results)
    
    # Overall accuracy
    analysis = {
        'total_tasks': len(df),
        'total_passed': df['passed'].sum(),
        'total_failed': (~df['passed']).sum(),
        'overall_accuracy': df['passed'].mean() * 100,
        'by_difficulty': {}
    }
    
    # Voting accuracy analysis
    voting_df = df[df['voting_correct'].notna()]
    if len(voting_df) > 0:
        correct_votes = voting_df['voting_correct'].sum()
        total_votes = len(voting_df)
        
        analysis['voting_accuracy'] = (correct_votes / total_votes * 100) if total_votes > 0 else 0
        analysis['voting_correct_count'] = int(correct_votes)
        analysis['voting_total_count'] = total_votes
        
        # Cases where voting failed despite having correct options
        better_available_df = df[df['voting_correct'] == False]
        had_better_choice = better_available_df[better_available_df['num_passing_solutions'] > 0]
        analysis['voting_chose_wrong_when_correct_available'] = len(had_better_choice)
        
        # Distribution of passing solutions
        passing_dist = df['num_passing_solutions'].value_counts().to_dict()
        analysis['passing_solutions_distribution'] = {
            '0_passing': passing_dist.get(0, 0),
            '1_passing': passing_dist.get(1, 0),
            '2_passing': passing_dist.get(2, 0),
            '3_passing': passing_dist.get(3, 0),
        }
    
    # By difficulty
    for level in ['easy', 'middle', 'hard']:
        level_df = df[df['level'] == level]
        if len(level_df) > 0:
            analysis['by_difficulty'][level] = {
                'total': len(level_df),
                'passed': level_df['passed'].sum(),
                'failed': (~level_df['passed']).sum(),
                'accuracy': level_df['passed'].mean() * 100
            }
    
    return analysis

def main():
    st.set_page_config(page_title="Multi-Model Voting Benchmark", layout="wide")
    
    st.title("Multi-Model Voting Benchmark")
    st.markdown("**Workflow:** All models generate solutions → All models vote → Test winning solution")
    st.markdown("**Models:** DeepSeek, CodeGemma, Code Llama")
    st.markdown("**Dataset:** McEval Generation Tasks (50 tasks)")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    st.sidebar.markdown("### Generation Phase")
    temperature_gen = st.sidebar.slider(
        "Temperature (Generation)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.1
    )
    
    st.sidebar.markdown("### Voting Phase")
    temperature_vote = st.sidebar.slider(
        "Temperature (Voting)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.1
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens (Generation)", 
        min_value=256, 
        max_value=2048, 
        value=1024,
        step=128
    )
    
    timeout = st.sidebar.slider(
        "Test Timeout (seconds)",
        min_value=10,
        max_value=120,
        value=40,
        step=5
    )
    
    # GPU info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### GPU Memory")
    if torch.cuda.is_available():
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            reserved_mem = torch.cuda.memory_reserved(0) / 1e9
            allocated_mem = torch.cuda.memory_allocated(0) / 1e9
            
            st.sidebar.text(f"Total: {total_mem:.1f} GB")
            st.sidebar.text(f"Reserved: {reserved_mem:.1f} GB")
            st.sidebar.text(f"Allocated: {allocated_mem:.1f} GB")
        except:
            st.sidebar.text("GPU info unavailable")
    
    # Load dataset
    st.header("Dataset")
    try:
        dataset = load_dataset()
        st.success(f"Loaded {len(dataset)} tasks from McEval")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            easy_count = len(dataset[dataset['level'] == 'easy'])
            st.metric("Easy", easy_count)
        with col2:
            middle_count = len(dataset[dataset['level'] == 'middle'])
            st.metric("Middle", middle_count)
        with col3:
            hard_count = len(dataset[dataset['level'] == 'hard'])
            st.metric("Hard", hard_count)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return
    
    # Run benchmark
    if st.button(" Run Voting Benchmark", type="primary"):
        st.header(" Running Benchmark")
        
        results = benchmark_voting(
            dataset=dataset,
            temperature_gen=temperature_gen,
            temperature_vote=temperature_vote,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        if results:
            analysis = analyze_results(results)
            
            # Overall statistics
            st.header("Overall Performance")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tasks", analysis['total_tasks'])
            with col2:
                st.metric("Passed", analysis['total_passed'])
            with col3:
                st.metric("Failed", analysis['total_failed'])
            with col4:
                st.metric("Accuracy", f"{analysis['overall_accuracy']:.2f}%")
            
            # Voting accuracy
            if 'voting_accuracy' in analysis:
                st.header("Voting Performance")
                st.markdown("**Did the models vote for correct solutions?**")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Voting Accuracy", f"{analysis['voting_accuracy']:.1f}%")
                with col2:
                    st.metric("Correct Votes", f"{analysis['voting_correct_count']}/{analysis['voting_total_count']}")
                with col3:
                    st.metric("Missed Better Options", analysis.get('voting_chose_wrong_when_correct_available', 0))
                
                # Solution distribution
                if 'passing_solutions_distribution' in analysis:
                    st.markdown("**Solutions passing per task:**")
                    dist = analysis['passing_solutions_distribution']
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("0/3 Pass", dist['0_passing'])
                    with cols[1]:
                        st.metric("1/3 Pass", dist['1_passing'])
                    with cols[2]:
                        st.metric("2/3 Pass", dist['2_passing'])
                    with cols[3]:
                        st.metric("3/3 Pass", dist['3_passing'])
            
            # By difficulty
            st.header("Performance by Difficulty")
            for level in ['easy', 'middle', 'hard']:
                if level in analysis['by_difficulty']:
                    stats = analysis['by_difficulty'][level]
                    with st.expander(f"{level.upper()} - {stats['accuracy']:.2f}%", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", stats['total'])
                        with col2:
                            st.metric("Passed", stats['passed'])
                        with col3:
                            st.metric("Failed", stats['failed'])
            
            # Export
            st.header("Export Results")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            def convert_to_serializable(obj):
                """Convert numpy/pandas types to native Python types"""
                import numpy as np
                
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                else:
                    return obj
            
            export_data = {
                'workflow': 'Multi-Model Voting (Generate → Vote → Test)',
                'models': list(MODELS.keys()),
                'timestamp': timestamp,
                'temperature_gen': float(temperature_gen),
                'temperature_vote': float(temperature_vote),
                'max_tokens': int(max_tokens),
                'timeout': int(timeout),
                'analysis': convert_to_serializable(analysis),
                'results': convert_to_serializable(results)
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_str,
                file_name=f"voting_benchmark_{timestamp}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()