import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional, Tuple, List
import gc
import re
import pandas as pd
from datetime import datetime
import json

# Model configurations
MODELS = {
    "DeepSeek Coder 7B Instruct": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "CodeGemma Instruct": "google/codegemma-7b-it",
    "CodeQwen 1.5 7B": "Qwen/CodeQwen1.5-7B",
    "Code Llama Instruct": "codellama/CodeLlama-7b-Instruct-hf"
}

# Default task
DEFAULT_TASK = """from typing import List
 
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
"""

def extract_code_from_solution(solution: str) -> str:
    """Extract code from solution tags, preserving imports and complete code"""
    # Try to find content between <solution> tags
    match = re.search(r'<solution>(.*?)</solution>', solution, re.DOTALL)
    if match:
        code = match.group(1).strip()
        
        # Remove any >>> prefix at the start
        if code.startswith('>>>'):
            code = code[3:].strip()
        
        # Remove markdown code fences - handle multiple patterns
        # Pattern 1: ```python at start
        if code.startswith('```python'):
            code = code[9:].strip()
        # Pattern 2: ``` at start
        elif code.startswith('```'):
            code = code[3:].strip()
        
        # Remove closing ``` at end
        if code.endswith('```'):
            code = code[:-3].strip()
        
        return code
    
    # If no tags found, return the solution as-is
    return solution.strip()

def run_unit_tests(code: str) -> Dict:
    """Run unit tests on the provided code and return results"""
    result = {
        'passed': False,
        'total_tests': 7,
        'passed_tests': 0,
        'failed_tests': [],
        'error': None,
        'code_tested': None
    }
    
    try:
        # Extract code from various wrapping formats
        code_to_test = code.strip()
        
        # Skip if code is just "PASS"
        if 'PASS' in code_to_test.upper() and len(code_to_test) < 50:
            result['error'] = "Code is PASS, cannot test"
            return result
        
        # Step 1: Extract from <solution> tags if present
        solution_match = re.search(r'<solution>(.*?)</solution>', code_to_test, re.DOTALL)
        if solution_match:
            code_to_test = solution_match.group(1).strip()
        
        # Step 2: Remove >>> prefix if present
        if code_to_test.startswith('>>>'):
            code_to_test = code_to_test[3:].strip()
        
        # Step 3: Extract from markdown code fence if present
        # Look for ```python or ``` followed by code, then closing ```
        fence_match = re.search(r'```(?:python)?\s*\n(.*?)\n```', code_to_test, re.DOTALL)
        if fence_match:
            # Found properly fenced code
            code_to_test = fence_match.group(1).strip()
        else:
            # Try alternative: just strip the fences manually
            lines = code_to_test.split('\n')
            cleaned_lines = []
            in_code = False
            
            for line in lines:
                stripped = line.strip()
                
                # Start of code fence
                if stripped.startswith('```'):
                    in_code = not in_code
                    continue
                
                # If we're in code, keep the line
                if in_code:
                    cleaned_lines.append(line)
                # If not in code and line looks like code (not explanation), keep it
                elif line and not re.match(r'^[A-Z][a-z\s,]+', stripped):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                code_to_test = '\n'.join(cleaned_lines).strip()
        
        result['code_tested'] = code_to_test
        
        # Create a namespace for execution
        namespace = {}
        
        # Add typing import if not already in the code (models often omit it)
        if 'from typing import' not in code_to_test and 'import typing' not in code_to_test:
            exec("from typing import List, Dict, Tuple, Optional, Union", namespace)
        
        # Execute the code to define the function
        exec(code_to_test, namespace)
        
        # Check if has_close_elements function exists
        if 'has_close_elements' not in namespace:
            result['error'] = "Function 'has_close_elements' not found in code"
            return result
        
        has_close_elements = namespace['has_close_elements']
        
        # Define test cases
        test_cases = [
            ([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3, True),
            ([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05, False),
            ([1.0, 2.0, 5.9, 4.0, 5.0], 0.95, True),
            ([1.0, 2.0, 5.9, 4.0, 5.0], 0.8, False),
            ([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1, True),
            ([1.1, 2.2, 3.1, 4.1, 5.1], 1.0, True),
            ([1.1, 2.2, 3.1, 4.1, 5.1], 0.5, False),
        ]
        
        # Run each test
        for i, (numbers, threshold, expected) in enumerate(test_cases, 1):
            try:
                actual = has_close_elements(numbers, threshold)
                if actual == expected:
                    result['passed_tests'] += 1
                else:
                    result['failed_tests'].append({
                        'test_num': i,
                        'input': f"has_close_elements({numbers}, {threshold})",
                        'expected': expected,
                        'actual': actual
                    })
            except Exception as e:
                result['failed_tests'].append({
                    'test_num': i,
                    'input': f"has_close_elements({numbers}, {threshold})",
                    'expected': expected,
                    'actual': f"Error: {str(e)}"
                })
        
        # Mark as passed if all tests pass
        result['passed'] = (result['passed_tests'] == result['total_tests'])
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def create_reflection_prompt(task: str, previous_solution: str, iteration: int) -> str:
    """Create a prompt for reflection and improvement with XML encapsulation"""
    if iteration == 1:
        # First iteration - just provide the task
        return f"""<instruction>
<task>
{task}
</task>
Please provide a detailed solution to the task described within the <task> tag. Focus on clarity, correctness, and completeness.
Encapsulate your solution within <solution> tags. Include any necessary imports inside the <solution> tags.
Example response format:
>>> <solution>
Your solution here
</solution>
</instruction>
Your response:"""
    else:
        # Extract just the code from the previous solution
        previous_code = extract_code_from_solution(previous_solution)
        # Subsequent iterations - show only the code being tested
        return f"""<instruction>
<task>
{task}
</task>
<previousCode>
{previous_code}
</previousCode>
Review the code in the <previousCode> tag for the task described in the <task> tag.
**IMPORTANT:** If the previous code is already correct and complete, you may respond with:
>>> <solution>
PASS
</solution>
This indicates you approve the previous code as-is.
Otherwise, please improve the previous code by considering:
1. Are there any errors or bugs?
2. Can the logic be simplified or optimized?
3. Are there edge cases not handled?
Encapsulate your improved solution within <solution> tags. Include any necessary imports inside the <solution> tags.
Example response format:
>>> <solution>
Your improved solution here
</solution>
</instruction>
Your response:"""

# Global cache for models with LRU eviction (max 2 models to prevent OOM)
_model_cache = {}
_model_access_order = []
MAX_CACHED_MODELS = 2

def evict_oldest_model():
    """Remove the least recently used model from cache"""
    if len(_model_cache) >= MAX_CACHED_MODELS and _model_access_order:
        oldest_model = _model_access_order.pop(0)
        if oldest_model in _model_cache:
            st.write(f"🗑️ Evicting {oldest_model} from cache to free memory...")
            del _model_cache[oldest_model]
            torch.cuda.empty_cache()
            gc.collect()

def get_model(model_name: str):
    """Get model from cache or load it if not cached (LRU with max 2 models)"""
    
    # If model is in cache, move it to end (most recently used)
    if model_name in _model_cache:
        _model_access_order.remove(model_name)
        _model_access_order.append(model_name)
        st.write(f"♻️ Using cached {model_name}")
        return _model_cache[model_name]
    
    # Need to load the model - first check if we need to evict
    evict_oldest_model()
    
    try:
        st.write(f"🔄 Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[model_name],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        _model_cache[model_name] = (tokenizer, model)
        _model_access_order.append(model_name)
        st.write(f"✅ {model_name} loaded and cached")
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        return None, None

def clear_model_cache():
    """Clear all cached models to free memory"""
    global _model_cache, _model_access_order
    for model_name in list(_model_cache.keys()):
        del _model_cache[model_name]
    _model_cache = {}
    _model_access_order = []
    torch.cuda.empty_cache()
    gc.collect()
    st.write("🗑️ Model cache cleared")

def generate_response(tokenizer, model, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> Optional[str]:
    """Generate response from a model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Remove token_type_ids if present (Qwen doesn't use them)
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.001,  # Avoid exact 0
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt):].strip()
        return response
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def run_pairwise_experiment(
    model_a_name: str,
    model_b_name: str,
    task: str,
    temperature: float,
    max_tokens: int
) -> Dict:
    """Run a single pairwise cross-reflection experiment: A generates, B reviews"""
    
    result = {
        'generator': model_a_name,
        'reviewer': model_b_name,
        'temperature': temperature,
        'iteration_1_solution': None,
        'iteration_2_solution': None,
        'iteration_2_is_pass': False,
        'reviewer_action': None,
        'iteration_1_test_results': None,
        'iteration_2_test_results': None,
        'improvement': False,
        'success': False,
        'error': None
    }
    
    try:
        # Get Model A (generator) from cache or load
        tokenizer_a, model_a = get_model(model_a_name)
        if tokenizer_a is None or model_a is None:
            result['error'] = f"Failed to load {model_a_name}"
            return result
        
        # Iteration 1: Model A generates initial solution
        st.write(f"✍️ {model_a_name} generating initial solution...")
        prompt_1 = create_reflection_prompt(task, "", 1)
        response_1 = generate_response(tokenizer_a, model_a, prompt_1, temperature, max_tokens)
        
        if response_1 is None:
            result['error'] = "Failed to generate iteration 1"
            return result
        
        result['iteration_1_solution'] = response_1
        
        # Test iteration 1
        st.write(f"🧪 Testing iteration 1 solution...")
        test_results_1 = run_unit_tests(response_1)
        result['iteration_1_test_results'] = test_results_1
        
        # Get Model B (reviewer) from cache or load
        tokenizer_b, model_b = get_model(model_b_name)
        if tokenizer_b is None or model_b is None:
            result['error'] = f"Failed to load {model_b_name}"
            return result
        
        # Iteration 2: Model B reviews and improves
        st.write(f"🔍 {model_b_name} reviewing solution...")
        prompt_2 = create_reflection_prompt(task, response_1, 2)
        response_2 = generate_response(tokenizer_b, model_b, prompt_2, temperature, max_tokens)
        
        if response_2 is None:
            result['error'] = "Failed to generate iteration 2"
            return result
        
        # Check if response is empty or just whitespace - treat as PASS
        if not response_2.strip():
            response_2 = "PASS"
        
        result['iteration_2_solution'] = response_2
        result['iteration_2_is_pass'] = 'PASS' in response_2.upper()
        
        # Test iteration 2 - ALWAYS run tests, even if reviewer said PASS
        st.write(f"🧪 Testing iteration 2 solution...")
        
        # If reviewer said PASS, test the original code again
        if result['iteration_2_is_pass']:
            test_results_2 = run_unit_tests(response_1)  # Test original code
            result['reviewer_action'] = 'PASS'
        else:
            test_results_2 = run_unit_tests(response_2)  # Test new code
            result['reviewer_action'] = 'CHANGED'
        
        result['iteration_2_test_results'] = test_results_2
        
        # Determine if there was improvement
        if test_results_1 and test_results_2:
            result['improvement'] = (
                test_results_2['passed_tests'] > test_results_1['passed_tests']
            )
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def main():
    st.set_page_config(page_title="Pairwise Cross-Reflection", layout="wide")
    
    st.title("🔄 Pairwise Cross-Reflection Experiments")
    st.markdown("Systematic testing of all 2-model combinations: Model A generates, Model B reviews")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.1,
        help="Generation temperature (0.0 = deterministic, 1.0 = creative)"
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens", 
        min_value=128, 
        max_value=2048, 
        value=512, 
        step=64
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Experiment Info")
    st.sidebar.info(f"**Total combinations:** 12\n\n(4 models × 3 partners each)")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Model Cache")
    st.sidebar.info(f"**Max cached models:** {MAX_CACHED_MODELS}\n\n(Keeps only the 2 most recently used models to prevent OOM)")
    
    if _model_cache:
        st.sidebar.success(f"**Currently cached:** {len(_model_cache)}")
        for model_name in _model_access_order:
            st.sidebar.text(f"✓ {model_name}")
    else:
        st.sidebar.info("No models cached yet")
    
    if st.sidebar.button("🗑️ Clear Cache", help="Free GPU memory by clearing all cached models"):
        clear_model_cache()
        st.sidebar.success("Cache cleared!")
        st.rerun()
    
    # Task input
    st.header("📝 Task Definition")
    task = st.text_area(
        "Enter the coding task:",
        value=DEFAULT_TASK,
        height=200
    )
    
    # Run experiments
    st.header("🚀 Run Experiments")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_all = st.button("▶️ Run All 12 Combinations", type="primary", use_container_width=True)
    
    if run_all:
        # Initialize results storage
        all_results = []
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model_names = list(MODELS.keys())
        total_experiments = len(model_names) * (len(model_names) - 1)
        current_experiment = 0
        
        # Run all pairwise combinations
        for i, model_a in enumerate(model_names):
            for j, model_b in enumerate(model_names):
                if model_a == model_b:
                    continue
                
                current_experiment += 1
                status_text.markdown(f"**Experiment {current_experiment}/{total_experiments}:** {model_a} → {model_b}")
                
                # Create expander for this experiment
                with st.expander(f"🔬 {model_a} → {model_b}", expanded=False):
                    result = run_pairwise_experiment(
                        model_a,
                        model_b,
                        task,
                        temperature,
                        max_tokens
                    )
                    
                    if result['success']:
                        st.success("✅ Completed successfully")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.markdown(f"**Iteration 1** ({model_a})")
                            
                            # Show test results for iteration 1
                            if result['iteration_1_test_results']:
                                test_res = result['iteration_1_test_results']
                                if test_res.get('error'):
                                    st.error(f"❌ Test Error: {test_res['error']}")
                                else:
                                    passed = test_res['passed_tests']
                                    total = test_res['total_tests']
                                    if test_res['passed']:
                                        st.success(f"✅ All tests passed ({passed}/{total})")
                                    else:
                                        st.warning(f"⚠️ Tests: {passed}/{total} passed")
                                        if test_res['failed_tests']:
                                            with st.expander("View failed tests"):
                                                for fail in test_res['failed_tests']:
                                                    st.write(f"Test {fail['test_num']}: {fail['input']}")
                                                    st.write(f"  Expected: {fail['expected']}, Got: {fail['actual']}")
                            
                            # Add debug view
                            with st.expander("🔍 Debug: View Raw Output"):
                                st.text("Raw model output:")
                                st.code(result['iteration_1_solution'], language="text")
                                if result['iteration_1_test_results'] and result['iteration_1_test_results'].get('code_tested'):
                                    st.text("Code that was tested:")
                                    st.code(result['iteration_1_test_results']['code_tested'], language="python")
                            
                            st.code(result['iteration_1_solution'], language="python")
                        
                        with col_b:
                            st.markdown(f"**Iteration 2** ({model_b})")
                            
                            # Always show what the reviewer did
                            if result['iteration_2_is_pass']:
                                st.info("✅ Reviewer approved with PASS")
                            else:
                                st.info("🔄 Reviewer made changes")
                            
                            # Always show test results for iteration 2
                            if result['iteration_2_test_results']:
                                test_res = result['iteration_2_test_results']
                                if test_res.get('error'):
                                    st.error(f"❌ Test Error: {test_res['error']}")
                                else:
                                    passed = test_res['passed_tests']
                                    total = test_res['total_tests']
                                    if test_res['passed']:
                                        st.success(f"✅ All tests passed ({passed}/{total})")
                                    else:
                                        st.warning(f"⚠️ Tests: {passed}/{total} passed")
                                        if test_res['failed_tests']:
                                            with st.expander("View failed tests"):
                                                for fail in test_res['failed_tests']:
                                                    st.write(f"Test {fail['test_num']}: {fail['input']}")
                                                    st.write(f"  Expected: {fail['expected']}, Got: {fail['actual']}")
                            
                            # Show improvement indicator
                            if result['improvement']:
                                st.success("📈 Improvement detected!")
                            elif result.get('iteration_1_test_results') and result.get('iteration_2_test_results'):
                                iter1_passed = result['iteration_1_test_results']['passed_tests']
                                iter2_passed = result['iteration_2_test_results']['passed_tests']
                                if iter2_passed < iter1_passed:
                                    st.error("📉 Regression detected!")
                                elif iter2_passed == iter1_passed:
                                    st.info("➡️ No change in test performance")
                            
                            # Add debug view
                            with st.expander("🔍 Debug: View Raw Output"):
                                st.text("Raw model output:")
                                st.code(result['iteration_2_solution'], language="text")
                                if result['iteration_2_test_results'] and result['iteration_2_test_results'].get('code_tested'):
                                    st.text("Code that was tested:")
                                    st.code(result['iteration_2_test_results']['code_tested'], language="python")
                            
                            st.code(result['iteration_2_solution'], language="python")
                    else:
                        st.error(f"❌ Failed: {result['error']}")
                    
                    all_results.append(result)
                
                # Update progress
                progress_bar.progress(current_experiment / total_experiments)
        
        status_text.markdown("**✅ All experiments completed!**")
        
        # Analysis section
        st.header("📊 Results Analysis")
        
        # Create summary dataframe
        summary_data = []
        for r in all_results:
            if r['success']:
                iter1_tests = r.get('iteration_1_test_results')
                iter2_tests = r.get('iteration_2_test_results')
                
                iter1_passed = iter1_tests.get('passed_tests', 'N/A') if iter1_tests else 'N/A'
                iter2_passed = iter2_tests.get('passed_tests', 'N/A') if iter2_tests else 'N/A'
                
                summary_data.append({
                    'Generator': r['generator'],
                    'Reviewer': r['reviewer'],
                    'Iter1 Tests': f"{iter1_passed}/7" if iter1_passed != 'N/A' else 'N/A',
                    'Iter2 Tests': f"{iter2_passed}/7" if iter2_passed != 'N/A' else 'N/A',
                    'Action': r.get('reviewer_action', 'N/A'),
                    'Temperature': r['temperature']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            st.subheader("📈 Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_success = len([r for r in all_results if r['success']])
                st.metric("Successful Runs", f"{total_success}/{total_experiments}")
            
            with col2:
                total_pass = len([r for r in all_results if r.get('iteration_2_is_pass', False)])
                st.metric("Reviewer Approved (PASS)", f"{total_pass}/{total_success}")
            
            with col3:
                # Count how many iteration 2 solutions passed all tests
                iter2_perfect = len([
                    r for r in all_results 
                    if r.get('iteration_2_test_results') and r['iteration_2_test_results'].get('passed', False)
                ])
                st.metric("Iter2 Perfect (7/7)", f"{iter2_perfect}/{total_success}")
            
            # Detailed test performance analysis
            st.subheader("🧪 Test Performance Analysis")
            
            col_test1, col_test2 = st.columns(2)
            
            with col_test1:
                st.markdown("**Iteration 1 Performance**")
                iter1_scores = {}
                for r in all_results:
                    if r['success'] and r.get('iteration_1_test_results'):
                        gen = r['generator']
                        score = r['iteration_1_test_results'].get('passed_tests', 0)
                        if gen not in iter1_scores:
                            iter1_scores[gen] = []
                        iter1_scores[gen].append(score)
                
                for model in sorted(iter1_scores.keys()):
                    scores = iter1_scores[model]
                    avg_score = sum(scores) / len(scores)
                    st.write(f"{model}: {avg_score:.1f}/7 avg")
            
            with col_test2:
                st.markdown("**Iteration 2 Performance**")
                iter2_scores = {}
                for r in all_results:
                    if r['success'] and r.get('iteration_2_test_results'):
                        rev = r['reviewer']
                        score = r['iteration_2_test_results'].get('passed_tests', 0)
                        if rev not in iter2_scores:
                            iter2_scores[rev] = []
                        iter2_scores[rev].append(score)
                
                for model in sorted(iter2_scores.keys()):
                    scores = iter2_scores[model]
                    avg_score = sum(scores) / len(scores)
                    st.write(f"{model}: {avg_score:.1f}/7 avg")
            
            # Model-specific analysis
            st.subheader("🎯 Model Performance")
            
            col_gen, col_rev = st.columns(2)
            
            with col_gen:
                st.markdown("**As Generator**")
                gen_stats = {}
                for r in all_results:
                    if r['success']:
                        gen = r['generator']
                        if gen not in gen_stats:
                            gen_stats[gen] = {'total': 0, 'approved': 0}
                        gen_stats[gen]['total'] += 1
                        if r['iteration_2_is_pass']:
                            gen_stats[gen]['approved'] += 1
                
                for model, stats in gen_stats.items():
                    rate = (stats['approved'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    st.write(f"{model}: {stats['approved']}/{stats['total']} ({rate:.1f}%)")
            
            with col_rev:
                st.markdown("**As Reviewer**")
                rev_stats = {}
                for r in all_results:
                    if r['success']:
                        rev = r['reviewer']
                        if rev not in rev_stats:
                            rev_stats[rev] = {'total': 0, 'approved': 0}
                        rev_stats[rev]['total'] += 1
                        if r['iteration_2_is_pass']:
                            rev_stats[rev]['approved'] += 1
                
                for model, stats in rev_stats.items():
                    rate = (stats['approved'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    st.write(f"{model}: {stats['approved']}/{stats['total']} ({rate:.1f}%)")
            
            # Download results
            st.subheader("💾 Export Results")
            
            # Prepare detailed results for download
            detailed_results = []
            for r in all_results:
                detailed_results.append({
                    'generator': r['generator'],
                    'reviewer': r['reviewer'],
                    'temperature': r['temperature'],
                    'success': r['success'],
                    'reviewer_action': r.get('reviewer_action', 'N/A'),
                    'reviewer_approved': r.get('iteration_2_is_pass', False),
                    'iteration_1_solution': r.get('iteration_1_solution', ''),
                    'iteration_2_solution': r.get('iteration_2_solution', ''),
                    'iteration_1_test_results': r.get('iteration_1_test_results'),
                    'iteration_2_test_results': r.get('iteration_2_test_results'),
                    'improvement': r.get('improvement', False),
                    'error': r.get('error', '')
                })
            
            json_str = json.dumps(detailed_results, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_str,
                file_name=f"pairwise_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()