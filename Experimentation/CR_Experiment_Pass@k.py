import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from typing import Dict, Optional
import gc
from datetime import datetime
import json
import multiprocessing
from multiprocessing import Process, Queue

# Model configurations
GEMMA_MODEL_NAME = "CodeGemma 7B Instruct"
GEMMA_MODEL_PATH = "google/codegemma-7b-it"

LLAMA_MODEL_NAME = "Code Llama 7B Instruct"
LLAMA_MODEL_PATH = "codellama/CodeLlama-7b-Instruct-hf"

DEEPSEEK_MODEL_NAME = "DeepSeek Coder 7B Instruct"
DEEPSEEK_MODEL_PATH = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"

DATASET_PATH = "/home/demouser/Desktop/121336311/McEval DataSet/McEval_Generation_Tasks.csv"


TOP_K = 10

def load_dataset() -> pd.DataFrame:
    """Load the McEval benchmark dataset - only task 2 (index 1)"""
    df = pd.read_csv(DATASET_PATH)
    
    # Get only task 2 (row at index 1, since 0-indexed)
    return df.iloc[[1]]

def _execute_test_in_process(code_to_test: str, test_code: str, entry_point: str, result_queue: Queue):
    """Execute test in a separate process - this function runs in the subprocess"""
    result = {
        'passed': False,
        'error': None,
        'code_tested': code_to_test
    }
    
    try:
        # Create namespace
        namespace = {}
        
        # Add common imports
        exec("from typing import List, Dict, Tuple, Optional, Union, Any", namespace)
        
        # Execute the generated code
        try:
            exec(code_to_test, namespace)
        except SyntaxError as e:
            result['error'] = f"Syntax error in generated code: {str(e)}"
            result_queue.put(result)
            return
        except Exception as e:
            result['error'] = f"Error executing generated code: {str(e)}"
            result_queue.put(result)
            return
        
        # Check if function exists
        if entry_point not in namespace:
            result['error'] = f"Function '{entry_point}' not found in code. Found: {list(namespace.keys())}"
            result_queue.put(result)
            return
        
        # Execute the test code
        try:
            exec(test_code, namespace)
        except AssertionError as e:
            result['error'] = f"Test assertion failed: {str(e)}"
            result_queue.put(result)
            return
        except Exception as e:
            result['error'] = f"Error running tests: {str(e)}"
            result_queue.put(result)
            return
        
        # If we get here, tests passed
        result['passed'] = True
        result_queue.put(result)
        
    except Exception as e:
        result['error'] = f"Unexpected error in subprocess: {str(e)}"
        result_queue.put(result)

def run_unit_test(code: str, test_code: str, entry_point: str, timeout: int = 40) -> Dict:
    """Run unit tests on generated code with timeout using multiprocessing"""
    result = {
        'passed': False,
        'error': None,
        'code_tested': None
    }
    
    try:
        # Extract code from various wrapping formats (same logic as working 3-model script)
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
        
        # Check if code is empty
        if not code_to_test or len(code_to_test.strip()) == 0:
            result['error'] = "Generated code is empty after extraction"
            return result
        
        # Create a queue to receive results from the subprocess
        result_queue = Queue()
        
        # Create and start the process
        process = Process(target=_execute_test_in_process, args=(code_to_test, test_code, entry_point, result_queue))
        process.start()
        
        # Wait for the process to complete or timeout
        process.join(timeout=timeout)
        
        if process.is_alive():
            # Process is still running after timeout - terminate it
            process.terminate()
            process.join()
            result['error'] = f"Infinite loop detected: Test execution exceeded {timeout} seconds timeout"
            return result
        
        # Check if we got a result from the queue
        if not result_queue.empty():
            result = result_queue.get()
        else:
            # Process finished but no result - likely crashed
            result['error'] = "Test process terminated unexpectedly without returning a result"
        
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
    
    return result

def load_model_safe(model_name: str, model_path: str):
    """Load model with proper configuration - no caching"""
    import os
    os.environ["ACCELERATE_USE_FSDP"] = "false"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    
    try:
        st.write(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        st.write(f"{model_name} loaded on {device}")
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None, None

def cleanup_model(model, tokenizer):
    """Completely clean up model and free GPU memory"""
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_response(tokenizer, model, device, prompt: str, temperature: float = 0.3, max_tokens: int = 512, top_k: int = TOP_K) -> Optional[str]:
    """Generate response from a model with specified top_k"""
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Remove token_type_ids if present (some models don't use them)
            if 'token_type_ids' in inputs:
                inputs.pop('token_type_ids')
            
            prompt_length = inputs["input_ids"].shape[1]
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.001,
                do_sample=temperature > 0,
                top_k=top_k,  
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Extract only generated tokens
            generated_tokens = outputs[0][prompt_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up tensors immediately
            del inputs
            del outputs
            torch.cuda.empty_cache()
            
            return result.strip() if result.strip() else None
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

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

def create_planning_prompt(task_prompt: str) -> str:
    return f"""Task: {task_prompt}

Write a detailed plan to solve this. Do not write code yet. Focus on:
- Algorithm approach
- Key steps
- Edge cases

Plan:"""

def create_plan_review_prompt(task_prompt: str, previous_plan: str) -> str:
    """Create prompt for LLaMA to review and improve CodeGemma's plan"""
    # Extract plan from solution tags if present
    plan_content = extract_code_from_solution(previous_plan)
    
    return f"""<instruction>
<task>
{task_prompt}
</task>
<previousPlan>
{plan_content}
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

def create_coding_prompt(task_prompt: str, plan: str) -> str:
    """Create prompt for DeepSeek to generate code from the plan"""
    # Extract plan from solution tags if present
    plan_content = extract_code_from_solution(plan)
    
    return f"""<instruction>
<task>
{task_prompt}
</task>
<plan>
{plan_content}
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

def run_three_model_workflow(
    task_prompt: str,
    test_code: str,
    entry_point: str,
    task_id: str,
    level: str,
    temperature_plan: float,
    temperature_review: float,
    temperature_code: float,
    max_tokens: int,
    timeout: int,
    gemma_tokenizer,
    gemma_model,
    gemma_device,
    llama_tokenizer,
    llama_model,
    llama_device,
    deepseek_tokenizer,
    deepseek_model,
    deepseek_device
) -> Dict:
    """Run the three-model workflow: Gemma plans -> LLaMA reviews -> DeepSeek codes"""
    
    result = {
        'task_id': task_id,
        'level': level,
        'entry_point': entry_point,
        'prompt': task_prompt,
        'plan': None,
        'reviewed_plan': None,
        'generated_code': None,
        'passed': False,
        'error': None,
        'code_tested': None
    }
    
    try:
        # Step 1: CodeGemma creates plan
        planning_prompt = create_planning_prompt(task_prompt)
        plan = generate_response(gemma_tokenizer, gemma_model, gemma_device, 
                                planning_prompt, temperature_plan, max_tokens, top_k=TOP_K)
        
        if plan is None or not plan.strip():
            result['error'] = "Failed to generate plan (empty output from CodeGemma)"
            return result
        
        result['plan'] = plan
        
        # Step 2: LLaMA reviews and improves plan
        review_prompt = create_plan_review_prompt(task_prompt, plan)
        reviewed_plan = generate_response(llama_tokenizer, llama_model, llama_device,
                                         review_prompt, temperature_review, max_tokens, top_k=TOP_K)
        
        if reviewed_plan is None or not reviewed_plan.strip():
            result['error'] = "Failed to review plan (empty output from LLaMA)"
            return result
        
        result['reviewed_plan'] = reviewed_plan
        
        # Step 3: DeepSeek generates code from reviewed plan
        coding_prompt = create_coding_prompt(task_prompt, reviewed_plan)
        code = generate_response(deepseek_tokenizer, deepseek_model, deepseek_device,
                                coding_prompt, temperature_code, max_tokens, top_k=TOP_K)
        
        if code is None or not code.strip():
            result['error'] = "Failed to generate code (empty output from DeepSeek)"
            return result
        
        result['generated_code'] = code
        
        # Step 4: Test the code
        test_result = run_unit_test(code, test_code, entry_point, timeout=timeout)
        
        result['passed'] = test_result['passed']
        result['error'] = test_result['error']
        result['code_tested'] = test_result['code_tested']
        
    except Exception as e:
        result['error'] = f"Workflow error: {str(e)}"
    
    return result

def test_task2(
    temperature_plan: float,
    temperature_review: float,
    temperature_code: float,
    max_tokens: int,
    timeout: int
) -> Dict:
    """Test only task 2 from the dataset"""
    
    # Load dataset (just task 2)
    dataset = load_dataset()
    
    if len(dataset) == 0:
        st.error("Failed to load task 2")
        return None
    
    row = dataset.iloc[0]
    
    # Load all three models
    st.write("### Loading Models")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gemma_tokenizer, gemma_model, gemma_device = load_model_safe(GEMMA_MODEL_NAME, GEMMA_MODEL_PATH)
        if gemma_tokenizer is None or gemma_model is None:
            st.error(f"Failed to load {GEMMA_MODEL_NAME}")
            return None
    
    with col2:
        llama_tokenizer, llama_model, llama_device = load_model_safe(LLAMA_MODEL_NAME, LLAMA_MODEL_PATH)
        if llama_tokenizer is None or llama_model is None:
            cleanup_model(gemma_model, gemma_tokenizer)
            st.error(f"Failed to load {LLAMA_MODEL_NAME}")
            return None
    
    with col3:
        deepseek_tokenizer, deepseek_model, deepseek_device = load_model_safe(DEEPSEEK_MODEL_NAME, DEEPSEEK_MODEL_PATH)
        if deepseek_tokenizer is None or deepseek_model is None:
            cleanup_model(gemma_model, gemma_tokenizer)
            cleanup_model(llama_model, llama_tokenizer)
            st.error(f"Failed to load {DEEPSEEK_MODEL_NAME}")
            return None
    
    st.success("All models loaded successfully")
    
    # Run workflow
    st.write(f"### Testing Task 2: {row['task_id']} ({row['level']})")
    
    result = run_three_model_workflow(
        task_prompt=row['prompt'],
        test_code=row['test'],
        entry_point=row['entry_point'],
        task_id=row['task_id'],
        level=row['level'],
        temperature_plan=temperature_plan,
        temperature_review=temperature_review,
        temperature_code=temperature_code,
        max_tokens=max_tokens,
        timeout=timeout,
        gemma_tokenizer=gemma_tokenizer,
        gemma_model=gemma_model,
        gemma_device=gemma_device,
        llama_tokenizer=llama_tokenizer,
        llama_model=llama_model,
        llama_device=llama_device,
        deepseek_tokenizer=deepseek_tokenizer,
        deepseek_model=deepseek_model,
        deepseek_device=deepseek_device
    )
    
    # Cleanup models
    cleanup_model(gemma_model, gemma_tokenizer)
    cleanup_model(llama_model, llama_tokenizer)
    cleanup_model(deepseek_model, deepseek_tokenizer)
    
    return result

def main():
    st.set_page_config(page_title="Task 2 Test - Gemma→LLaMA→DeepSeek", layout="wide")
    
    st.title("Task 2 Test: CodeGemma → LLaMA → DeepSeek")
    st.markdown("**Workflow:** CodeGemma creates plan → LLaMA reviews plan → DeepSeek generates code")
    st.markdown("**Dataset:** McEval Generation Tasks - Task 2 ONLY")
    st.markdown(f"**Top-k:** {TOP_K}")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    st.sidebar.markdown(f"**Planning Model:** {GEMMA_MODEL_NAME}")
    st.sidebar.markdown(f"**Review Model:** {LLAMA_MODEL_NAME}")
    st.sidebar.markdown(f"**Coding Model:** {DEEPSEEK_MODEL_NAME}")
    st.sidebar.markdown(f"**Testing:** Task 2 only")
    st.sidebar.markdown(f"**Top-k:** {TOP_K}")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Planning (CodeGemma)")
    temperature_plan = st.sidebar.slider(
        "Temperature (Plan)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Temperature for planning generation"
    )
    
    st.sidebar.markdown("### Review (LLaMA)")
    temperature_review = st.sidebar.slider(
        "Temperature (Review)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Temperature for plan review"
    )
    
    st.sidebar.markdown("### Coding (DeepSeek)")
    temperature_code = st.sidebar.slider(
        "Temperature (Code)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.05,
        help="Temperature for code generation"
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens", 
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
        step=5,
        help="Maximum time allowed for test execution"
    )
    
    # GPU info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### GPU Memory")
    if torch.cuda.is_available():
        if st.sidebar.button("Refresh GPU Stats"):
            pass
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            reserved_mem = torch.cuda.memory_reserved(0) / 1e9
            allocated_mem = torch.cuda.memory_allocated(0) / 1e9
            free_mem = total_mem - reserved_mem
            
            st.sidebar.text(f"Total: {total_mem:.1f} GB")
            st.sidebar.text(f"Reserved: {reserved_mem:.1f} GB")
            st.sidebar.text(f"Allocated: {allocated_mem:.1f} GB")
            st.sidebar.text(f"Free: {free_mem:.1f} GB")
        except:
            st.sidebar.text("GPU info unavailable")
    else:
        st.sidebar.text("No GPU detected")
    
    # Load dataset preview
    st.header("Dataset")
    try:
        dataset = load_dataset()
        st.success(f"Loaded task 2: {dataset.iloc[0]['task_id']} ({dataset.iloc[0]['level']})")
        
        with st.expander("View Task Details"):
            st.markdown(f"**Task ID:** {dataset.iloc[0]['task_id']}")
            st.markdown(f"**Difficulty:** {dataset.iloc[0]['level']}")
            st.markdown(f"**Entry Point:** {dataset.iloc[0]['entry_point']}")
            st.markdown("**Prompt:**")
            st.text(dataset.iloc[0]['prompt'])
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return
    
    # Run test
    if st.button("Run Test on Task 2", type="primary"):
        st.header("Running Test")
        
        result = test_task2(
            temperature_plan=temperature_plan,
            temperature_review=temperature_review,
            temperature_code=temperature_code,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        if result:
            # Display result
            st.header("Result")
            
            if result['passed']:
                st.success("✅ Task 2 PASSED")
            else:
                st.error(f"❌ Task 2 FAILED: {result['error']}")
            
            # Display workflow outputs
            st.header("Workflow Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Plan (CodeGemma):**")
                st.markdown(f"**({len(result['plan'])} chars):**")
                if result['plan']:
                    st.text(result['plan'])
                else:
                    st.warning("No plan generated")
            
            with col2:
                st.markdown("**Reviewed Plan (LLaMA):**")
                st.markdown(f"**({len(result.get('reviewed_plan', '')) if result.get('reviewed_plan') else 0} chars):**")
                if result.get('reviewed_plan'):
                    st.text(result['reviewed_plan'])
                else:
                    st.warning("No reviewed plan generated")
            
            with col3:
                st.markdown("**Generated Code (DeepSeek):**")
                st.markdown(f"**Raw output ({len(result.get('generated_code', '')) if result.get('generated_code') else 0} chars):**")
                if result.get('generated_code'):
                    st.code(result['generated_code'], language="text")
                else:
                    st.warning("No code generated")
                
                if result['code_tested']:
                    with st.expander("View processed code"):
                        st.code(result['code_tested'], language="python")
            
            # Download result
            st.header("Export Result")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"task2_test_topk{TOP_K}_{timestamp}.json"
            
            export_data = {
                'workflow': 'CodeGemma (Plan) → LLaMA (Review) → DeepSeek (Code)',
                'planning_model': GEMMA_MODEL_NAME,
                'planning_model_path': GEMMA_MODEL_PATH,
                'review_model': LLAMA_MODEL_NAME,
                'review_model_path': LLAMA_MODEL_PATH,
                'coding_model': DEEPSEEK_MODEL_NAME,
                'coding_model_path': DEEPSEEK_MODEL_PATH,
                'top_k': TOP_K,
                'timestamp': timestamp,
                'temperature_plan': float(temperature_plan),
                'temperature_review': float(temperature_review),
                'temperature_code': float(temperature_code),
                'max_tokens': int(max_tokens),
                'timeout': int(timeout),
                'result': result
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download Result (JSON)",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main()