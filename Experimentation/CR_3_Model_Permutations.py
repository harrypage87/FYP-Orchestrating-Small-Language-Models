import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional, Tuple, List
import gc
import re
import pandas as pd
from datetime import datetime
import json
import itertools

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

def load_model_safe(model_name):
    """Load model with proper configuration - no caching."""
    import os
    os.environ["ACCELERATE_USE_FSDP"] = "false"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    
    try:
        st.write(f"🔄 Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        st.write(f"✅ {model_name} loaded")
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None, None

def cleanup_model(model, tokenizer):
    """Completely clean up model and free GPU memory."""
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_response(tokenizer, model, device, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> Optional[str]:
    """Generate response from a model"""
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Remove token_type_ids if present (Qwen doesn't use them)
            if 'token_type_ids' in inputs:
                inputs.pop('token_type_ids')
            
            prompt_length = inputs["input_ids"].shape[1]
            
            # SIMPLIFIED generation parameters - no repetition penalty
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.001,
                do_sample=temperature > 0,
                top_p=0.95,  # Add nucleus sampling
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # Remove repetition_penalty - can cause degeneration
            )
            
            # Extract only generated tokens
            generated_tokens = outputs[0][prompt_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up tensors immediately
            del inputs
            del outputs
            torch.cuda.empty_cache()
            
            return result.strip()
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def run_three_model_experiment(
    model_a_name: str,
    model_b_name: str,
    model_c_name: str,
    task: str,
    temperature: float,
    max_tokens: int
) -> Dict:
    """Run a single 3-model experiment: A generates, B reviews, C reviews"""
    
    result = {
        'model_1': model_a_name,
        'model_2': model_b_name,
        'model_3': model_c_name,
        'temperature': temperature,
        'iteration_1_solution': None,
        'iteration_2_solution': None,
        'iteration_3_solution': None,
        'iteration_2_is_pass': False,
        'iteration_3_is_pass': False,
        'model_2_action': None,
        'model_3_action': None,
        'iteration_1_test_results': None,
        'iteration_2_test_results': None,
        'iteration_3_test_results': None,
        'success': False,
        'error': None
    }
    
    try:
        # === ITERATION 1: Model A generates ===
        # Free memory before loading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer_a, model_a, device = load_model_safe(MODELS[model_a_name])
        if tokenizer_a is None or model_a is None:
            result['error'] = f"Failed to load {model_a_name}"
            return result
        
        st.write(f"✍️ {model_a_name} generating initial solution...")
        prompt_1 = create_reflection_prompt(task, "", 1)
        response_1 = generate_response(tokenizer_a, model_a, device, prompt_1, temperature, max_tokens)
        
        if response_1 is None:
            cleanup_model(model_a, tokenizer_a)
            result['error'] = "Failed to generate iteration 1"
            return result
        
        result['iteration_1_solution'] = response_1
        
        # Test iteration 1
        st.write(f"🧪 Testing iteration 1 solution...")
        test_results_1 = run_unit_tests(response_1)
        result['iteration_1_test_results'] = test_results_1
        
        # Clean up Model A completely
        cleanup_model(model_a, tokenizer_a)
        
        # === ITERATION 2: Model B reviews ===
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer_b, model_b, device = load_model_safe(MODELS[model_b_name])
        if tokenizer_b is None or model_b is None:
            result['error'] = f"Failed to load {model_b_name}"
            return result
        
        st.write(f"🔍 {model_b_name} reviewing solution...")
        prompt_2 = create_reflection_prompt(task, response_1, 2)
        response_2 = generate_response(tokenizer_b, model_b, device, prompt_2, temperature, max_tokens)
        
        if response_2 is None:
            cleanup_model(model_b, tokenizer_b)
            result['error'] = "Failed to generate iteration 2"
            return result
        
        # Check if response is empty or just whitespace - treat as PASS
        if not response_2.strip():
            response_2 = "PASS"
        
        result['iteration_2_solution'] = response_2
        result['iteration_2_is_pass'] = 'PASS' in response_2.upper()
        
        # Test iteration 2 - ALWAYS run tests
        st.write(f"🧪 Testing iteration 2 solution...")
        
        if result['iteration_2_is_pass']:
            test_results_2 = run_unit_tests(response_1)  # Test original code
            result['model_2_action'] = 'PASS'
        else:
            test_results_2 = run_unit_tests(response_2)  # Test new code
            result['model_2_action'] = 'CHANGED'
        
        result['iteration_2_test_results'] = test_results_2
        
        # Clean up Model B completely
        cleanup_model(model_b, tokenizer_b)
        
        # === ITERATION 3: Model C reviews ===
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer_c, model_c, device = load_model_safe(MODELS[model_c_name])
        if tokenizer_c is None or model_c is None:
            result['error'] = f"Failed to load {model_c_name}"
            return result
        
        st.write(f"🔍 {model_c_name} reviewing solution...")
        # Model C reviews the output from Model B (whether it was PASS or changed code)
        prompt_3 = create_reflection_prompt(task, response_2 if not result['iteration_2_is_pass'] else response_1, 3)
        response_3 = generate_response(tokenizer_c, model_c, device, prompt_3, temperature, max_tokens)
        
        if response_3 is None:
            cleanup_model(model_c, tokenizer_c)
            result['error'] = "Failed to generate iteration 3"
            return result
        
        # Check if response is empty or just whitespace - treat as PASS
        if not response_3.strip():
            response_3 = "PASS"
        
        result['iteration_3_solution'] = response_3
        result['iteration_3_is_pass'] = 'PASS' in response_3.upper()
        
        # Test iteration 3 - ALWAYS run tests
        st.write(f"🧪 Testing iteration 3 solution...")
        
        if result['iteration_3_is_pass']:
            # Test whatever was the input to iteration 3
            test_input = response_2 if not result['iteration_2_is_pass'] else response_1
            test_results_3 = run_unit_tests(test_input)
            result['model_3_action'] = 'PASS'
        else:
            test_results_3 = run_unit_tests(response_3)  # Test new code
            result['model_3_action'] = 'CHANGED'
        
        result['iteration_3_test_results'] = test_results_3
        
        # Clean up Model C completely
        cleanup_model(model_c, tokenizer_c)
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def main():
    st.set_page_config(page_title="3-Model Permutations", layout="wide")
    
    st.title("🔄 3-Model Cross-Reflection Permutations")
    st.markdown("Systematic testing of all 3-model permutations: Model A generates, Model B reviews, Model C reviews")
    
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
        value=256,  # Reduced from 512 to save memory
        step=64
    )
    
    # GPU Memory info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ GPU Memory")
    try:
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            reserved_mem = torch.cuda.memory_reserved(0) / 1e9
            allocated_mem = torch.cuda.memory_allocated(0) / 1e9
            free_mem = total_mem - reserved_mem
            
            st.sidebar.text(f"Total: {total_mem:.1f} GB")
            st.sidebar.text(f"Reserved: {reserved_mem:.1f} GB")
            st.sidebar.text(f"Allocated: {allocated_mem:.1f} GB")
            st.sidebar.text(f"Free: {free_mem:.1f} GB")
            
            if free_mem < 10:
                st.sidebar.error("⚠️ Low GPU memory! Other processes may be using GPU.")
    except:
        pass
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Experiment Info")
    model_names = list(MODELS.keys())
    total_permutations = len(list(itertools.permutations(model_names, 3)))
    st.sidebar.info(f"**Total permutations:** {total_permutations}\n\n(4P3 = 4 × 3 × 2 = 24)")
    
    if st.sidebar.button("🧹 Force GPU Cleanup", help="Aggressive GPU memory cleanup"):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        st.sidebar.success("GPU memory cleaned!")
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
        run_all = st.button("▶️ Run All 24 Permutations", type="primary", use_container_width=True)
    
    if run_all:
        # Initialize results storage
        all_results = []
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate all permutations
        permutations = list(itertools.permutations(model_names, 3))
        total_experiments = len(permutations)
        
        # Run all permutations
        for idx, (model_a, model_b, model_c) in enumerate(permutations, 1):
            status_text.markdown(f"**Experiment {idx}/{total_experiments}:** {model_a} → {model_b} → {model_c}")
            
            # Create expander for this experiment
            with st.expander(f"🔬 {model_a} → {model_b} → {model_c}", expanded=False):
                result = run_three_model_experiment(
                    model_a,
                    model_b,
                    model_c,
                    task,
                    temperature,
                    max_tokens
                )
                
                if result['success']:
                    st.success("✅ Completed successfully")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    # Iteration 1
                    with col_a:
                        st.markdown(f"**Iteration 1** ({model_a})")
                        
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
                        
                        with st.expander("🔍 Debug: View Raw Output"):
                            st.text("Raw model output:")
                            st.code(result['iteration_1_solution'], language="text")
                            if result['iteration_1_test_results'] and result['iteration_1_test_results'].get('code_tested'):
                                st.text("Code that was tested:")
                                st.code(result['iteration_1_test_results']['code_tested'], language="python")
                        
                        st.code(result['iteration_1_solution'], language="python")
                    
                    # Iteration 2
                    with col_b:
                        st.markdown(f"**Iteration 2** ({model_b})")
                        
                        if result['iteration_2_is_pass']:
                            st.info("✅ Reviewer approved with PASS")
                        else:
                            st.info("🔄 Reviewer made changes")
                        
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
                        
                        with st.expander("🔍 Debug: View Raw Output"):
                            st.text("Raw model output:")
                            st.code(result['iteration_2_solution'], language="text")
                            if result['iteration_2_test_results'] and result['iteration_2_test_results'].get('code_tested'):
                                st.text("Code that was tested:")
                                st.code(result['iteration_2_test_results']['code_tested'], language="python")
                        
                        st.code(result['iteration_2_solution'], language="python")
                    
                    # Iteration 3
                    with col_c:
                        st.markdown(f"**Iteration 3** ({model_c})")
                        
                        if result['iteration_3_is_pass']:
                            st.info("✅ Reviewer approved with PASS")
                        else:
                            st.info("🔄 Reviewer made changes")
                        
                        if result['iteration_3_test_results']:
                            test_res = result['iteration_3_test_results']
                            if test_res.get('error'):
                                st.error(f"❌ Test Error: {test_res['error']}")
                            else:
                                passed = test_res['passed_tests']
                                total = test_res['total_tests']
                                if test_res['passed']:
                                    st.success(f"✅ All tests passed ({passed}/{total})")
                                else:
                                    st.warning(f"⚠️ Tests: {passed}/{total} passed")
                        
                        with st.expander("🔍 Debug: View Raw Output"):
                            st.text("Raw model output:")
                            st.code(result['iteration_3_solution'], language="text")
                            if result['iteration_3_test_results'] and result['iteration_3_test_results'].get('code_tested'):
                                st.text("Code that was tested:")
                                st.code(result['iteration_3_test_results']['code_tested'], language="python")
                        
                        st.code(result['iteration_3_solution'], language="python")
                else:
                    st.error(f"❌ Failed: {result['error']}")
                
                all_results.append(result)
            
            # Update progress
            progress_bar.progress(idx / total_experiments)
        
        status_text.markdown("**✅ All experiments completed!**")
        
        # Analysis section
        st.header("📊 Results Analysis")
        
        # Create summary dataframe
        summary_data = []
        for r in all_results:
            if r['success']:
                iter1_tests = r.get('iteration_1_test_results')
                iter2_tests = r.get('iteration_2_test_results')
                iter3_tests = r.get('iteration_3_test_results')
                
                iter1_passed = iter1_tests.get('passed_tests', 'N/A') if iter1_tests else 'N/A'
                iter2_passed = iter2_tests.get('passed_tests', 'N/A') if iter2_tests else 'N/A'
                iter3_passed = iter3_tests.get('passed_tests', 'N/A') if iter3_tests else 'N/A'
                
                summary_data.append({
                    'Model 1': r['model_1'],
                    'Model 2': r['model_2'],
                    'Model 3': r['model_3'],
                    'Iter1': f"{iter1_passed}/7" if iter1_passed != 'N/A' else 'N/A',
                    'Iter2': f"{iter2_passed}/7" if iter2_passed != 'N/A' else 'N/A',
                    'Iter3': f"{iter3_passed}/7" if iter3_passed != 'N/A' else 'N/A',
                    'M2 Action': r.get('model_2_action', 'N/A'),
                    'M3 Action': r.get('model_3_action', 'N/A'),
                    'Temp': r['temperature']
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
                iter3_perfect = len([
                    r for r in all_results 
                    if r.get('iteration_3_test_results') and r['iteration_3_test_results'].get('passed', False)
                ])
                st.metric("Iter3 Perfect (7/7)", f"{iter3_perfect}/{total_success}")
            
            with col3:
                both_pass = len([
                    r for r in all_results
                    if r.get('model_2_action') == 'PASS' and r.get('model_3_action') == 'PASS'
                ])
                st.metric("Both Approved (PASS)", f"{both_pass}/{total_success}")
            
            # Model performance analysis
            st.subheader("🎯 Model Performance by Position")
            
            col_pos1, col_pos2, col_pos3 = st.columns(3)
            
            with col_pos1:
                st.markdown("**Position 1 (Generator)**")
                pos1_scores = {}
                for r in all_results:
                    if r['success'] and r.get('iteration_1_test_results'):
                        model = r['model_1']
                        score = r['iteration_1_test_results'].get('passed_tests', 0)
                        if model not in pos1_scores:
                            pos1_scores[model] = []
                        pos1_scores[model].append(score)
                
                for model in sorted(pos1_scores.keys()):
                    scores = pos1_scores[model]
                    avg_score = sum(scores) / len(scores)
                    st.write(f"{model}: {avg_score:.1f}/7 avg")
            
            with col_pos2:
                st.markdown("**Position 2 (First Reviewer)**")
                pos2_scores = {}
                for r in all_results:
                    if r['success'] and r.get('iteration_2_test_results'):
                        model = r['model_2']
                        score = r['iteration_2_test_results'].get('passed_tests', 0)
                        if model not in pos2_scores:
                            pos2_scores[model] = []
                        pos2_scores[model].append(score)
                
                for model in sorted(pos2_scores.keys()):
                    scores = pos2_scores[model]
                    avg_score = sum(scores) / len(scores)
                    st.write(f"{model}: {avg_score:.1f}/7 avg")
            
            with col_pos3:
                st.markdown("**Position 3 (Second Reviewer)**")
                pos3_scores = {}
                for r in all_results:
                    if r['success'] and r.get('iteration_3_test_results'):
                        model = r['model_3']
                        score = r['iteration_3_test_results'].get('passed_tests', 0)
                        if model not in pos3_scores:
                            pos3_scores[model] = []
                        pos3_scores[model].append(score)
                
                for model in sorted(pos3_scores.keys()):
                    scores = pos3_scores[model]
                    avg_score = sum(scores) / len(scores)
                    st.write(f"{model}: {avg_score:.1f}/7 avg")
            
            # Download results
            st.subheader("💾 Export Results")
            
            detailed_results = []
            for r in all_results:
                detailed_results.append({
                    'model_1': r['model_1'],
                    'model_2': r['model_2'],
                    'model_3': r['model_3'],
                    'temperature': r['temperature'],
                    'success': r['success'],
                    'model_2_action': r.get('model_2_action', 'N/A'),
                    'model_3_action': r.get('model_3_action', 'N/A'),
                    'iteration_1_solution': r.get('iteration_1_solution', ''),
                    'iteration_2_solution': r.get('iteration_2_solution', ''),
                    'iteration_3_solution': r.get('iteration_3_solution', ''),
                    'iteration_1_test_results': r.get('iteration_1_test_results'),
                    'iteration_2_test_results': r.get('iteration_2_test_results'),
                    'iteration_3_test_results': r.get('iteration_3_test_results'),
                    'error': r.get('error', '')
                })
            
            json_str = json.dumps(detailed_results, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json_str,
                file_name=f"three_model_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()