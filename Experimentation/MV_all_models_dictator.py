import streamlit as st
import torch
import gc
import time
import pandas as pd
from itertools import combinations
from datetime import datetime
import json

st.set_page_config(page_title="SLM Permutation Tester", layout="wide")
st.title("SLM Permutation Tester")

# ================== MODELS ==================
ALL_MODELS = {
    "DeepSeek Coder 6.7B": "deepseek-ai/deepseek-coder-7b-base-v1.5",
    "CodeQwen 1.5 7B": "Qwen/CodeQwen1.5-7B",
    "CodeGemma 7B": "google/codegemma-7b",
    "StarCoder2 7B": "bigcode/starcoder2-7b",
    "CodeLlama 7B": "meta-llama/CodeLlama-7b-hf",
}

JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

JUDGE_SYSTEM_PROMPT = """You are an expert code reviewer. Compare the candidate solutions and select the best one.

Evaluate based on:
- Correctness: Does it solve the task completely?
- Completeness: Is the code finished (no "pass" stubs, no truncated code)?
- Edge cases: Does it handle edge cases properly?
- Code quality: Is it clean and efficient?

Respond in exactly this format:
REASONING: [1-2 sentences explaining which solution is best and why]
VOTE: [Letter A, B, or C]

Be concise and direct."""

# ================== TEST TASKS ==================
TEST_TASKS = {
    "Task 1: has_close_elements": {
        "prompt": """from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"""",
        "test_code": """
def check(has_close_elements):
    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True
    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True
    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False

check(has_close_elements)
"""
    },
    
    "Task 2: separate_paren_groups": {
        "prompt": """from typing import List

def separate_paren_groups(paren_string: str) -> List[str]:
    \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    \"\"\"""",
        "test_code": """
def check(separate_paren_groups):
    assert separate_paren_groups('(()()) ((())) () ((())()())') == [
        '(()())', '((()))', '()', '((())()())'
    ]
    assert separate_paren_groups('() (()) ((())) (((())))') == [
        '()', '(())', '((()))', '(((())))'
    ]
    assert separate_paren_groups('(()(())((())))') == [
        '(()(())((())))'
    ]
    assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']

check(separate_paren_groups)
"""
    },
    
    "Task 3: truncate_number": {
        "prompt": """def truncate_number(number: float) -> float:
    \"\"\" Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    \"\"\"""",
        "test_code": """
def check(truncate_number):
    assert truncate_number(3.5) == 0.5
    assert abs(truncate_number(1.33) - 0.33) < 1e-6
    assert abs(truncate_number(123.456) - 0.456) < 1e-6

check(truncate_number)
"""
    }
}

# ================== SIDEBAR ==================
st.sidebar.header("Test Configuration")


# Select which tasks to run
selected_tasks = st.sidebar.multiselect(
    "Select Tasks to Run",
    options=list(TEST_TASKS.keys()),
    default=list(TEST_TASKS.keys())
)

# Generation parameters
max_tokens = st.sidebar.slider("Max New Tokens", 50, 500, 150)
temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)

if st.sidebar.button("Clear GPU Cache"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.sidebar.success("GPU cache cleared")

st.sidebar.divider()


if st.sidebar.button("Clear HuggingFace Cache"):
    import os
    import shutil
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        try:
            size_before = 0
            for dirpath, _, filenames in os.walk(cache_dir):
                for filename in filenames:
                    try:
                        file_path = os.path.join(dirpath, filename)
                        if os.path.exists(file_path):
                            size_before += os.path.getsize(file_path)
                    except (OSError, FileNotFoundError):
                        continue
            size_before_gb = size_before / (1024**3)
            
            # Remove all cached models
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception as e:
                    # Skip files/folders that can't be removed
                    continue
            
            st.sidebar.success(f"Cleared {size_before_gb:.2f} GB from HuggingFace cache")
        except Exception as e:
            st.sidebar.error(f"Error clearing cache: {e}")
    else:
        st.sidebar.info("No HuggingFace cache found")



# Display GPU info
if torch.cuda.is_available():
    st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")
    st.sidebar.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
else:
    st.sidebar.warning("No GPU available - using CPU")

# Display disk space info
import os
import shutil
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_dir):
    try:
        cache_size = 0
        for dirpath, _, filenames in os.walk(cache_dir):
            for filename in filenames:
                try:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        cache_size += os.path.getsize(file_path)
                except (OSError, FileNotFoundError):
                    # Skip files that were deleted or are inaccessible
                    continue
        cache_size_gb = cache_size / (1024**3)
        st.sidebar.warning(f"HF Cache: {cache_size_gb:.2f} GB")
    except Exception as e:
        st.sidebar.info("HF Cache: Unable to calculate size")

# ================== HELPER FUNCTIONS ==================
def clean_output(output: str) -> str:
    """Remove trailing whitespace and empty lines."""
    lines = output.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()

def load_model_safe(model_name, is_judge=False):
    """Safely load tokenizer and model to CPU/GPU."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import os
        
        # Force disable meta device globally
        os.environ["ACCELERATE_USE_FSDP"] = "false"
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # The KEY is to explicitly set low_cpu_mem_usage=False
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # CRITICAL: This prevents meta tensors
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        # Move to device after loading
        model = model.to(device)
        
        return tokenizer, model, device
        
    except Exception as e:
        st.error(f"Failed to load {model_name}: {str(e)}")
        return None, None, None

def generate_code(tokenizer, model, device, prompt, max_new_tokens=150, temp=0.7):
    """Generate code using a model."""
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Move inputs to the same device as the model
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            
            # Get the length of the prompt
            prompt_length = inputs["input_ids"].shape[1]
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                temperature=temp,
                do_sample=True,
                repetition_penalty=1.2,  # Prevent repetition loops
                no_repeat_ngram_size=3   # Prevent repetitive patterns
            )
            
            # Decode only the new tokens (skip the prompt)
            generated_tokens = outputs[0][prompt_length:]
            text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Combine prompt with generated completion
            full_output = prompt + text
            
            return clean_output(full_output)
            
    except Exception as e:
        return f"# Generation error: {e}"

def judge_solutions(tokenizer, model, device, task, solutions):
    """Use judge model to select best solution."""
    try:
        solution_mapping = {}
        candidate_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        
        judge_prompt = f"<TASK>\n{task}\n</TASK>\n\n"
        
        for idx, (model_name, solution) in enumerate(solutions.items()):
            letter = candidate_letters[idx]
            solution_mapping[letter] = model_name
            separator = f"{'-' * 35}{letter}{'-' * 35}"
            judge_prompt += f"{separator}\n{solution}\n"
        
        judge_prompt += f"{'-' * 71}\n"
        
        with torch.no_grad():
            if hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt}
                ]
                inputs = tokenizer.apply_chat_template(
                    messages, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )
            else:
                full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{judge_prompt}"
                inputs = tokenizer(full_prompt, return_tensors="pt")
            
            # Move inputs to model device
            if isinstance(inputs, torch.Tensor):
                if hasattr(model, 'device'):
                    inputs = inputs.to(model.device)
                else:
                    inputs = inputs.to(device)
            else:
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if isinstance(inputs, dict) and "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            
            outputs = model.generate(
                inputs if isinstance(inputs, torch.Tensor) else inputs["input_ids"],
                max_new_tokens=500,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                temperature=0.3,
                do_sample=True
            )
            
            judgment = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if hasattr(tokenizer, 'apply_chat_template'):
                if "REASONING:" in judgment:
                    judgment = judgment[judgment.find("REASONING:"):]
            else:
                if "REASONING:" in judgment:
                    judgment = judgment[judgment.find("REASONING:"):]
                elif judgment.startswith(JUDGE_SYSTEM_PROMPT):
                    judgment = judgment[len(JUDGE_SYSTEM_PROMPT):].strip()
                    if judgment.startswith(judge_prompt):
                        judgment = judgment[len(judge_prompt):].strip()
            
            return judgment.strip(), solution_mapping
    except Exception as e:
        return f"Judgment error: {e}", {}

def extract_winner(judgment, solution_mapping):
    """Extract the winning model from the judgment."""
    import re
    vote_match = re.search(r'\*\*VOTE:\*\*\s*([A-F])', judgment)
    if not vote_match:
        vote_match = re.search(r'VOTE:\s*([A-F])', judgment)
    
    if vote_match:
        winning_letter = vote_match.group(1)
        if winning_letter in solution_mapping:
            return solution_mapping[winning_letter], winning_letter
    return None, None

def run_unit_tests(solution_code, test_code, task_name):
    """
    Execute the generated solution against unit tests.
    Returns: (passed, total_tests, error_message)
    """
    try:
        # Create a namespace for execution
        namespace = {}
        
        # Execute the solution code
        exec(solution_code, namespace)
        
        # Execute the test code
        exec(test_code, namespace)
        
        # If we get here, all tests passed
        return True, "All tests passed", None
        
    except AssertionError as e:
        return False, "Tests failed", f"AssertionError: {str(e)}"
    except SyntaxError as e:
        return False, "Syntax error", f"SyntaxError: {str(e)}"
    except Exception as e:
        return False, "Runtime error", f"{type(e).__name__}: {str(e)}"

# ================== GENERATE PERMUTATIONS ==================
def get_model_permutations():
    """Generate all 3-model combinations from the 5 models."""
    model_names = list(ALL_MODELS.keys())
    return list(combinations(model_names, 3))

# ================== MAIN TEST INTERFACE ==================
st.header("Test Configuration")

col1, col2 = st.columns(2)
with col1:
    st.metric("Total Models", len(ALL_MODELS))
    st.metric("Models per Test", 3)
with col2:
    st.metric("Total Permutations", len(list(get_model_permutations())))
    st.metric("Selected Tasks", len(selected_tasks))

st.info(f"**Total Tests to Run:** {len(list(get_model_permutations())) * len(selected_tasks)}")

# Display permutations
with st.expander("View All Permutations"):
    perms = get_model_permutations()
    for idx, perm in enumerate(perms, 1):
        st.write(f"{idx}. {' + '.join(perm)}")

# ================== RUN TESTS ==================
if 'test_results' not in st.session_state:
    st.session_state.test_results = []

if st.button("Run All Tests", type="primary"):
    if not selected_tasks:
        st.warning("Please select at least one task to run.")
    else:
        st.header("Running Tests")
        
        # Clear previous results
        st.session_state.test_results = []
        
        permutations = get_model_permutations()
        total_tests = len(permutations) * len(selected_tasks)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        test_counter = 0
        
        # Iterate through each permutation
        for perm_idx, model_triplet in enumerate(permutations, 1):
            st.subheader(f"Permutation {perm_idx}/{len(permutations)}: {' + '.join(model_triplet)}")
            
            # Iterate through each task
            for task_name in selected_tasks:
                test_counter += 1
                task_data = TEST_TASKS[task_name]
                task_prompt = task_data["prompt"]
                task_tests = task_data["test_code"]
                
                status_text.text(f"Test {test_counter}/{total_tests}: {task_name} with {model_triplet}")
                
                st.write(f"**{task_name}**")
                
                # Store results for this test
                test_result = {
                    "permutation_id": perm_idx,
                    "models": model_triplet,
                    "task": task_name,
                    "timestamp": datetime.now().isoformat(),
                    "solutions": {},
                    "test_results": {},
                    "judgment": None,
                    "winner": None,
                    "winner_letter": None
                }
                
                # Generate solutions from each model in the triplet
                solutions = {}
                
                for model_name in model_triplet:
                    model_path = ALL_MODELS[model_name]
                    
                    # Clean up GPU memory (but keep disk cache for reuse)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    with st.spinner(f"Generating from {model_name}..."):
                        tokenizer, model, device = load_model_safe(model_path)
                        
                        if tokenizer and model:
                            output = generate_code(tokenizer, model, device, task_prompt, max_new_tokens=max_tokens, temp=temperature)
                            solutions[model_name] = output
                            test_result["solutions"][model_name] = output
                            
                            # Run unit tests on the generated solution
                            passed, status, error = run_unit_tests(output, task_tests, task_name)
                            test_result["test_results"][model_name] = {
                                "passed": passed,
                                "status": status,
                                "error": error
                            }
                            
                            # Display compact test result
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                if passed:
                                    st.success(f"**{model_name}**: Tests passed")
                                else:
                                    st.error(f"**{model_name}**: {status}")
                            with col2:
                                with st.expander("View Code"):
                                    st.code(output, language="python")
                                    if not passed and error:
                                        st.caption(f"Error: {error}")
                            
                            # Clean up GPU memory (but keep disk cache for reuse)
                            del model, tokenizer
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            solutions[model_name] = "Failed to load model"
                            test_result["solutions"][model_name] = "Failed to load model"
                            test_result["test_results"][model_name] = {
                                "passed": False,
                                "status": "Model load failed",
                                "error": "Could not load model"
                            }
                
                # Run judge
                with st.spinner("Running judge..."):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    judge_tokenizer, judge_model, judge_device = load_model_safe(JUDGE_MODEL, is_judge=True)
                    
                    if judge_tokenizer and judge_model:
                        judgment, solution_mapping = judge_solutions(judge_tokenizer, judge_model, judge_device, task_prompt, solutions)
                        winner, winner_letter = extract_winner(judgment, solution_mapping)
                        
                        test_result["judgment"] = judgment
                        test_result["winner"] = winner
                        test_result["winner_letter"] = winner_letter
                        
                        # Determine if the judge made a good decision
                        if winner:
                            # Check which models passed tests
                            passed_models = [model for model, res in test_result["test_results"].items() if res["passed"]]
                            failed_models = [model for model, res in test_result["test_results"].items() if not res["passed"]]
                            
                            # Determine judge decision quality
                            if len(passed_models) == 0:
                                # All candidates failed - orange
                                st.warning(f"**Judge**: {winner} (all failed)")
                                test_result["judge_decision"] = "all_failed"
                            elif winner in passed_models:
                                # Judge chose a correct solution - green
                                st.success(f"**Judge**: {winner}")
                                test_result["judge_decision"] = "correct"
                            else:
                                # Judge chose wrong when correct options existed - red
                                st.error(f"**Judge**: {winner} (wrong - better: {', '.join([m.split()[0] for m in passed_models])})")
                                test_result["judge_decision"] = "incorrect"
                        else:
                            st.warning("Judge: Could not determine winner")
                            test_result["judge_decision"] = "unknown"
                        
                        del judge_model, judge_tokenizer
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        test_result["judgment"] = "Judge failed to load"
                        test_result["judge_decision"] = "judge_failed"
                        st.error("Judge failed to load")
                
                # Save result
                st.session_state.test_results.append(test_result)
                
                progress_bar.progress(test_counter / total_tests)
                
                st.divider()
        
        status_text.text("All tests complete!")
        st.success(f"Completed {total_tests} tests across {len(permutations)} permutations!")

# ================== RESULTS ANALYSIS ==================
if st.session_state.test_results:
    st.header("Results Analysis")
    
    # Create summary DataFrame
    summary_data = []
    for result in st.session_state.test_results:
        # Count how many tests passed
        passed_count = sum(1 for model, test_res in result["test_results"].items() if test_res["passed"])
        total_models = len(result["test_results"])
        
        # Get judge decision quality
        judge_decision = result.get("judge_decision", "unknown")
        if judge_decision == "correct":
            decision_emoji = "✅"
        elif judge_decision == "incorrect":
            decision_emoji = "❌"
        elif judge_decision == "all_failed":
            decision_emoji = "🟠"
        else:
            decision_emoji = "⚠️"
        
        summary_data.append({
            "Permutation": f"{result['permutation_id']}: {' + '.join([m.split()[0] for m in result['models']])}",
            "Task": result["task"],
            "Tests Passed": f"{passed_count}/{total_models}",
            "Winner": result["winner"] if result["winner"] else "N/A",
            "Judge Decision": f"{decision_emoji} {judge_decision}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    st.subheader("Test Summary")
    st.dataframe(df_summary, use_container_width=True)
    
    # Win count by model
    st.subheader("Win Count by Model")
    win_counts = {}
    for result in st.session_state.test_results:
        winner = result["winner"]
        if winner:
            win_counts[winner] = win_counts.get(winner, 0) + 1
    
    if win_counts:
        df_wins = pd.DataFrame(list(win_counts.items()), columns=["Model", "Wins"])
        df_wins = df_wins.sort_values("Wins", ascending=False)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(df_wins.set_index("Model"))
        with col2:
            st.dataframe(df_wins, use_container_width=True)
    
    # Unit test pass rates
    st.subheader("Unit Test Pass Rates by Model")
    test_pass_counts = {}
    test_total_counts = {}
    
    for result in st.session_state.test_results:
        for model_name, test_res in result["test_results"].items():
            if model_name not in test_pass_counts:
                test_pass_counts[model_name] = 0
                test_total_counts[model_name] = 0
            
            test_total_counts[model_name] += 1
            if test_res["passed"]:
                test_pass_counts[model_name] += 1
    
    if test_pass_counts:
        df_pass_rates = pd.DataFrame([
            {
                "Model": model,
                "Tests Passed": test_pass_counts[model],
                "Total Tests": test_total_counts[model],
                "Pass Rate": f"{(test_pass_counts[model] / test_total_counts[model] * 100):.1f}%"
            }
            for model in test_pass_counts.keys()
        ])
        df_pass_rates = df_pass_rates.sort_values("Tests Passed", ascending=False)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(df_pass_rates.set_index("Model")["Tests Passed"])
        with col2:
            st.dataframe(df_pass_rates, use_container_width=True)
    
    # Judge accuracy analysis
    st.subheader("Judge Decision Quality")
    judge_stats = {
        "correct": 0,
        "incorrect": 0, 
        "all_failed": 0,
        "unknown": 0
    }
    
    for result in st.session_state.test_results:
        decision = result.get("judge_decision", "unknown")
        if decision in judge_stats:
            judge_stats[decision] += 1
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Correct Choices", judge_stats["correct"])
    with col2:
        st.metric("Incorrect Choices", judge_stats["incorrect"])
    with col3:
        st.metric("All Failed", judge_stats["all_failed"])
    with col4:
        total_meaningful = judge_stats["correct"] + judge_stats["incorrect"]
        accuracy = (judge_stats["correct"] / total_meaningful * 100) if total_meaningful > 0 else 0
        st.metric("Judge Accuracy", f"{accuracy:.1f}%")
    
    # Detailed results
    with st.expander("View Detailed Results"):
        for idx, result in enumerate(st.session_state.test_results, 1):
            st.subheader(f"Test {idx}: {result['task']}")
            st.write(f"**Permutation:** {' + '.join(result['models'])}")
            
            # Show unit test results
            st.write("**Unit Test Results:**")
            test_cols = st.columns(len(result['test_results']))
            for col_idx, (model_name, test_res) in enumerate(result['test_results'].items()):
                with test_cols[col_idx]:
                    if test_res["passed"]:
                        st.success(f"{model_name.split()[0]}")
                    else:
                        st.error(f"{model_name.split()[0]}")
                        st.caption(test_res["status"])
            
            # Show judge decision with color coding
            if result.get('winner'):
                judge_decision = result.get("judge_decision", "unknown")
                if judge_decision == "correct":
                    st.success(f"**Judge Decision: CORRECT** - Chose {result['winner']} ({result['winner_letter']})")
                elif judge_decision == "incorrect":
                    passed_models = [m for m, r in result['test_results'].items() if r["passed"]]
                    st.error(f"**Judge Decision: INCORRECT** - Chose {result['winner']} ({result['winner_letter']}) when better options existed: {', '.join(passed_models)}")
                elif judge_decision == "all_failed":
                    st.warning(f"**Judge Decision: ALL FAILED** - Chose {result['winner']} ({result['winner_letter']}) but all candidates failed tests")
                else:
                    st.info(f"**Judge Winner:** {result['winner']} ({result['winner_letter']})")
            
            st.write("**Judgment:**")
            st.text(result['judgment'])
            
            st.write("**Solutions:**")
            for model_name, solution in result['solutions'].items():
                test_res = result['test_results'].get(model_name, {})
                status_icon = "✅" if test_res.get("passed", False) else "❌"
                with st.expander(f"{status_icon} {model_name}"):
                    st.code(solution, language="python")
                    if not test_res.get("passed", False) and test_res.get("error"):
                        st.error(f"Error: {test_res['error']}")
            
            st.divider()
    
    # Export results
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as JSON
        json_data = json.dumps(st.session_state.test_results, indent=2)
        st.download_button(
            label="Download Results (JSON)",
            data=json_data,
            file_name=f"slm_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        # Export summary as CSV
        csv_data = df_summary.to_csv(index=False)
        st.download_button(
            label="Download Summary (CSV)",
            data=csv_data,
            file_name=f"slm_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


