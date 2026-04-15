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

# Task 1: has_close_elements
TASK_1 = """from typing import List
 
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
"""

# Task 2: max_product
TASK_2 = """def max_product(num_str: str, N: int, K: int) -> int:
    \"\"\"
    Function name: max_product
    Arguments:
    num_str (str): A string representing the number string.
    N (int): An integer representing the length of the number string.
    K (int): An integer representing the number of multiplication signs to insert.
    Return type: int (The function returns the maximum product that can be obtained by inserting K multiplication signs into the number string.)
    
    Example:
    >>> max_product("123", 3, 1)
    36  # 12*3 = 36 (or 1*23 = 23, max is 36)
    >>> max_product("1234", 4, 2)
    144  # 12*3*4 = 144 (or 1*2*34 = 68, max is 144)
    \"\"\"
"""

TASKS = {
    "Task 1: has_close_elements": TASK_1,
    "Task 2: max_product": TASK_2
}

def extract_code_from_solution(solution: str) -> str:
    """Extract content from solution tags"""
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
    """Create a prompt for plan-then-code workflow"""
    if iteration == 1:
        # Model 1: Generate step-by-step plan
        return f"""<instruction>
<task>
{task}
</task>
Please provide a detailed step-by-step plan to solve this task. Do NOT write code yet.
Focus on:
1. What algorithm or approach should be used?
2. What are the key steps in the solution?
3. What data structures are needed?
4. What edge cases need to be considered?

Be specific and detailed in your plan. Explain the logic clearly.

Encapsulate your plan within <solution> tags.
Example response format:
>>> <solution>
Step 1: [Detailed description]
Step 2: [Detailed description]
Step 3: [Detailed description]
...
</solution>
</instruction>
Your response:"""
    
    elif iteration == 2:
        # Model 2: Review and improve the plan
        previous_plan = extract_code_from_solution(previous_solution)
        return f"""<instruction>
<task>
{task}
</task>
<previousPlan>
{previous_plan}
</previousPlan>
Carefully review the plan in the <previousPlan> tag for the task described in the <task> tag.

Your job is to improve this plan by thinking critically about:
1. **Algorithm correctness**: Is the approach fundamentally sound? Are there logical errors?
2. **Completeness**: Are any steps missing or vague? What needs more detail?
3. **Edge cases**: What special cases need to be handled? (empty inputs, single elements, zeros, duplicates, etc.)
4. **Data structures**: Are the right data structures being used for efficiency?
5. **Clarity**: Can the steps be explained more clearly for implementation?

Even if the plan seems good, think about how to make it MORE robust and detailed.

Provide an improved step-by-step plan (NOT code) within <solution> tags.
Example response format:
>>> <solution>
Step 1: [Improved/clarified description]
Step 2: [Improved/clarified description]
Step 3: [Additional edge case handling]
...
</solution>
</instruction>
Your response:"""
    
    else:  # iteration == 3
        # Model 3: Generate code from the plan
        previous_plan = extract_code_from_solution(previous_solution)
        return f"""<instruction>
<task>
{task}
</task>
<plan>
{previous_plan}
</plan>
Now implement the plan described in the <plan> tag as Python code for the task in the <task> tag.

Follow the plan exactly and write clean, correct code. Include any necessary imports.

Encapsulate your code within <solution> tags. Include any necessary imports inside the <solution> tags.
Example response format:
>>> <solution>
import module_name

def function_name(...):
    # Implementation following the plan
    ...
</solution>
</instruction>
Your response:"""

def run_unit_tests_task1(code: str) -> Dict:
    """Run unit tests for has_close_elements"""
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
        fence_match = re.search(r'```(?:python)?\s*\n(.*?)\n```', code_to_test, re.DOTALL)
        if fence_match:
            code_to_test = fence_match.group(1).strip()
        else:
            # Try alternative: just strip the fences manually
            lines = code_to_test.split('\n')
            cleaned_lines = []
            in_code = False
            
            for line in lines:
                stripped = line.strip()
                
                if stripped.startswith('```'):
                    in_code = not in_code
                    continue
                
                if in_code:
                    cleaned_lines.append(line)
                elif line and not re.match(r'^[A-Z][a-z\s,]+', stripped):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                code_to_test = '\n'.join(cleaned_lines).strip()
        
        result['code_tested'] = code_to_test
        
        # Create a namespace for execution
        namespace = {}
        
        # Add typing import if not already in the code
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

def run_unit_tests_task2(code: str) -> Dict:
    """Run unit tests for max_product"""
    result = {
        'passed': False,
        'total_tests': 3,
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
        fence_match = re.search(r'```(?:python)?\s*\n(.*?)\n```', code_to_test, re.DOTALL)
        if fence_match:
            code_to_test = fence_match.group(1).strip()
        else:
            # Try alternative: just strip the fences manually
            lines = code_to_test.split('\n')
            cleaned_lines = []
            in_code = False
            
            for line in lines:
                stripped = line.strip()
                
                if stripped.startswith('```'):
                    in_code = not in_code
                    continue
                
                if in_code:
                    cleaned_lines.append(line)
                elif line and not re.match(r'^[A-Z][a-z\s,]+', stripped):
                    cleaned_lines.append(line)
            
            if cleaned_lines:
                code_to_test = '\n'.join(cleaned_lines).strip()
        
        result['code_tested'] = code_to_test
        
        # Create a namespace for execution
        namespace = {}
        
        # Execute the code to define the function
        exec(code_to_test, namespace)
        
        # Check if max_product function exists
        if 'max_product' not in namespace:
            result['error'] = "Function 'max_product' not found in code"
            return result
        
        max_product = namespace['max_product']
        
        # Define test cases
        test_cases = [
            ("123", 3, 1, 36),   # 12*3 = 36
            ("1234", 4, 2, 144), # 12*3*4 = 144
            ("051", 3, 1, 5),    # 05*1 = 5
        ]
        
        # Run each test
        for i, (num_str, N, K, expected) in enumerate(test_cases, 1):
            try:
                actual = max_product(num_str, N, K)
                if actual == expected:
                    result['passed_tests'] += 1
                else:
                    result['failed_tests'].append({
                        'test_num': i,
                        'input': f'max_product("{num_str}", {N}, {K})',
                        'expected': expected,
                        'actual': actual
                    })
            except Exception as e:
                result['failed_tests'].append({
                    'test_num': i,
                    'input': f'max_product("{num_str}", {N}, {K})',
                    'expected': expected,
                    'actual': f"Error: {str(e)}"
                })
        
        # Mark as passed if all tests pass
        result['passed'] = (result['passed_tests'] == result['total_tests'])
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def run_unit_tests(code: str, task_name: str) -> Dict:
    """Route to the appropriate test function based on task"""
    if "has_close_elements" in task_name or "Task 1" in task_name:
        return run_unit_tests_task1(code)
    elif "max_product" in task_name or "Task 2" in task_name:
        return run_unit_tests_task2(code)
    else:
        return {
            'passed': False,
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': [],
            'error': 'Unknown task type',
            'code_tested': None
        }

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
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.001,
                do_sample=temperature > 0,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Extract only generated tokens
            generated_tokens = outputs[0][prompt_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # DEBUG: Log raw output length and first 200 chars
            st.write(f"🔍 DEBUG: Generated {len(result)} characters")
            if len(result) < 200:
                st.write(f"🔍 DEBUG: Full output: '{result}'")
            else:
                st.write(f"🔍 DEBUG: First 200 chars: '{result[:200]}...'")
            
            # Check if output is empty or just whitespace
            if not result.strip():
                st.warning("⚠️ Model generated empty output!")
                return None
            
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
    task_name: str,
    temperature: float,
    max_tokens: int
) -> Dict:
    """Run a single 3-model experiment: A creates plan, B reviews plan, C codes"""
    
    result = {
        'model_1': model_a_name,
        'model_2': model_b_name,
        'model_3': model_c_name,
        'task_name': task_name,
        'temperature': temperature,
        'iteration_1_plan': None,
        'iteration_2_plan': None,
        'iteration_3_code': None,
        'model_2_action': None,
        'iteration_3_test_results': None,
        'success': False,
        'error': None
    }
    
    try:
        # === ITERATION 1: Model A creates plan ===
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer_a, model_a, device = load_model_safe(MODELS[model_a_name])
        if tokenizer_a is None or model_a is None:
            result['error'] = f"Failed to load {model_a_name}"
            return result
        
        st.write(f"📝 {model_a_name} creating plan...")
        prompt_1 = create_reflection_prompt(task, "", 1)
        response_1 = generate_response(tokenizer_a, model_a, device, prompt_1, temperature, max_tokens)
        
        if response_1 is None or not response_1.strip():
            cleanup_model(model_a, tokenizer_a)
            result['error'] = f"Failed to generate iteration 1 plan (empty output from {model_a_name})"
            return result
        
        result['iteration_1_plan'] = response_1
        
        # Clean up Model A completely
        cleanup_model(model_a, tokenizer_a)
        
        # === ITERATION 2: Model B reviews plan ===
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer_b, model_b, device = load_model_safe(MODELS[model_b_name])
        if tokenizer_b is None or model_b is None:
            result['error'] = f"Failed to load {model_b_name}"
            return result
        
        st.write(f"🔍 {model_b_name} reviewing plan...")
        prompt_2 = create_reflection_prompt(task, response_1, 2)
        response_2 = generate_response(tokenizer_b, model_b, device, prompt_2, temperature, max_tokens)
        
        if response_2 is None or not response_2.strip():
            cleanup_model(model_b, tokenizer_b)
            result['error'] = f"Failed to generate iteration 2 plan (empty output from {model_b_name})"
            return result
        
        result['iteration_2_plan'] = response_2
        
        # Model 2 always provides an improved plan (no PASS option)
        result['model_2_action'] = 'IMPROVED'
        final_plan = response_2
        
        # Clean up Model B completely
        cleanup_model(model_b, tokenizer_b)
        
        # === ITERATION 3: Model C generates code ===
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        tokenizer_c, model_c, device = load_model_safe(MODELS[model_c_name])
        if tokenizer_c is None or model_c is None:
            result['error'] = f"Failed to load {model_c_name}"
            return result
        
        st.write(f"💻 {model_c_name} generating code from plan...")
        prompt_3 = create_reflection_prompt(task, final_plan, 3)
        response_3 = generate_response(tokenizer_c, model_c, device, prompt_3, temperature, max_tokens)
        
        if response_3 is None or not response_3.strip():
            cleanup_model(model_c, tokenizer_c)
            result['error'] = f"Failed to generate iteration 3 code (empty output from {model_c_name})"
            return result
        
        result['iteration_3_code'] = response_3
        
        # Test iteration 3 (only iteration we test)
        st.write(f"🧪 Testing final code...")
        test_results_3 = run_unit_tests(response_3, task_name)
        result['iteration_3_test_results'] = test_results_3
        
        # Clean up Model C completely
        cleanup_model(model_c, tokenizer_c)
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def main():
    st.set_page_config(page_title="Plan-Then-Code: 3-Model Permutations", layout="wide")
    
    st.title("🎯 Plan-Then-Code: 3-Model Cross-Reflection")
    st.markdown("**Workflow:** Model 1 creates plan → Model 2 reviews plan → Model 3 generates code (only final code tested)")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Task selection
    selected_task_name = st.sidebar.selectbox(
        "Select Task",
        list(TASKS.keys()),
        index=0
    )
    selected_task = TASKS[selected_task_name]
    
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
    
    # Task display
    st.header("📝 Selected Task")
    st.markdown(f"**{selected_task_name}**")
    st.code(selected_task, language="python")
    
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
        permutations_list = list(itertools.permutations(model_names, 3))
        total_experiments = len(permutations_list)
        
        # Run all permutations
        for idx, (model_a, model_b, model_c) in enumerate(permutations_list, 1):
            status_text.markdown(f"**Experiment {idx}/{total_experiments}:** {model_a} → {model_b} → {model_c}")
            
            # Create expander for this experiment
            with st.expander(f"🔬 {model_a} → {model_b} → {model_c}", expanded=False):
                result = run_three_model_experiment(
                    model_a,
                    model_b,
                    model_c,
                    selected_task,
                    selected_task_name,
                    temperature,
                    max_tokens
                )
                
                if result['success']:
                    st.success("✅ Completed successfully")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    # Iteration 1: Plan
                    with col_a:
                        st.markdown(f"**Iteration 1: Plan** ({model_a})")
                        st.markdown("📝 Step-by-step plan created")
                        
                        with st.expander("🔍 View Plan"):
                            st.text(result['iteration_1_plan'])
                    
                    # Iteration 2: Plan Review
                    with col_b:
                        st.markdown(f"**Iteration 2: Review** ({model_b})")
                        st.info("🔄 Reviewer improved plan")
                        
                        with st.expander("🔍 View Improved Plan"):
                            st.text(result['iteration_2_plan'])
                    
                    # Iteration 3: Code + Tests
                    with col_c:
                        st.markdown(f"**Iteration 3: Code** ({model_c})")
                        
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
                                    if test_res['failed_tests']:
                                        with st.expander("View failed tests"):
                                            for fail in test_res['failed_tests']:
                                                st.write(f"Test {fail['test_num']}: {fail['input']}")
                                                st.write(f"  Expected: {fail['expected']}, Got: {fail['actual']}")
                        
                        with st.expander("🔍 View Code That Was Tested"):
                            st.markdown("**Raw Model Output:**")
                            st.code(result['iteration_3_code'], language="text")
                            
                            if result['iteration_3_test_results'] and result['iteration_3_test_results'].get('code_tested'):
                                st.markdown("**Extracted Code (What Was Actually Tested):**")
                                st.code(result['iteration_3_test_results']['code_tested'], language="python")
                            else:
                                st.info("No code was extracted/tested")
                else:
                    st.error(f"❌ Failed: {result['error']}")
                
                all_results.append(result)
            
            # Update progress
            progress_bar.progress(idx / total_experiments)
        
        status_text.markdown("**✅ All experiments completed!**")
        
        st.header("📊 Comprehensive Results Analysis")
        
        # Determine total tests for display
        if all_results and all_results[0]['success']:
            total_tests = all_results[0]['iteration_3_test_results']['total_tests']
        else:
            total_tests = "?"
        
        # Summary table
        summary_data = []
        for r in all_results:
            if r['success']:
                iter3_tests = r.get('iteration_3_test_results')
                iter3_passed = iter3_tests.get('passed_tests', 'N/A') if iter3_tests else 'N/A'
                
                summary_data.append({
                    'Model 1 (Plan)': r['model_1'],
                    'Model 2 (Review)': r['model_2'],
                    'Model 3 (Code)': r['model_3'],
                    'Final Tests': f"{iter3_passed}/{total_tests}" if iter3_passed != 'N/A' else 'N/A',
                    'M2 Action': r.get('model_2_action', 'N/A'),
                    'Success': '✅' if iter3_tests and iter3_tests.get('passed') else '❌'
                })
        
        if summary_data:
            st.subheader("📋 Summary Table")
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
            
            st.subheader("📈 Overall Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_success = len([r for r in all_results if r['success']])
                st.metric("Successful Runs", f"{total_success}/{total_experiments}")
            
            with col2:
                iter3_perfect = len([
                    r for r in all_results 
                    if r.get('iteration_3_test_results') and r['iteration_3_test_results'].get('passed', False)
                ])
                st.metric(f"Perfect Solutions ({total_tests}/{total_tests})", f"{iter3_perfect}/{total_success}")
            
            with col3:
                if total_success > 0:
                    success_rate = (iter3_perfect / total_success) * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Download results
            st.subheader("💾 Export Results")
            
            detailed_results = []
            for r in all_results:
                detailed_results.append({
                    'task_name': r.get('task_name', ''),
                    'model_1_planner': r['model_1'],
                    'model_2_reviewer': r['model_2'],
                    'model_3_coder': r['model_3'],
                    'temperature': r['temperature'],
                    'success': r['success'],
                    'model_2_action': r.get('model_2_action', 'N/A'),
                    'iteration_1_plan': r.get('iteration_1_plan', ''),
                    'iteration_2_plan': r.get('iteration_2_plan', ''),
                    'iteration_3_code': r.get('iteration_3_code', ''),
                    'iteration_3_test_results': r.get('iteration_3_test_results'),
                    'error': r.get('error', '')
                })
            
            json_str = json.dumps(detailed_results, indent=2)
            st.download_button(
                label="📥 Download Full Results (JSON)",
                data=json_str,
                file_name=f"plan_then_code_results_{selected_task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()