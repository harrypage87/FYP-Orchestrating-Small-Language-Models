import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from typing import Dict, List, Optional
import gc
from datetime import datetime
import json
import multiprocessing
from multiprocessing import Process, Queue

# Model configuration - DeepSeek for all stages
MODEL_NAME = "DeepSeek Coder 7B Instruct"
MODEL_PATH = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"

DATASET_PATH = "/home/demouser/Desktop/121336311/McEval DataSet/McEval_Generation_Tasks.csv"

def load_dataset() -> pd.DataFrame:
    """Load the McEval benchmark dataset - first 50 tasks"""
    df = pd.read_csv(DATASET_PATH)
    return df.head(50)

def _execute_test_in_process(code_to_test: str, test_code: str, entry_point: str, result_queue: Queue):
    """Execute test in a separate process - this function runs in the subprocess"""
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
            result['error'] = f"Syntax error in generated code: {str(e)}"
            result_queue.put(result)
            return
        except Exception as e:
            result['error'] = f"Error executing generated code: {str(e)}"
            result_queue.put(result)
            return
        
        if entry_point not in namespace:
            result['error'] = f"Function '{entry_point}' not found in code. Found: {list(namespace.keys())}"
            result_queue.put(result)
            return
        
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
        code_to_test = code.strip()
        
        if 'PASS' in code_to_test.upper() and len(code_to_test) < 50:
            result['error'] = "Code is PASS, cannot test"
            return result
        
        solution_match = re.search(r'<solution>(.*?)</solution>', code_to_test, re.DOTALL)
        if solution_match:
            code_to_test = solution_match.group(1).strip()
        
        if code_to_test.startswith('>>>'):
            code_to_test = code_to_test[3:].strip()
        
        fence_match = re.search(r'```(?:python)?\s*\n(.*?)\n```', code_to_test, re.DOTALL)
        if fence_match:
            code_to_test = fence_match.group(1).strip()
        else:
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
        
        if not code_to_test or len(code_to_test.strip()) == 0:
            result['error'] = "Generated code is empty after extraction"
            return result
        
        result_queue = Queue()
        process = Process(target=_execute_test_in_process, args=(code_to_test, test_code, entry_point, result_queue))
        process.start()
        process.join(timeout=timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            result['error'] = f"Infinite loop detected: Test execution exceeded {timeout} seconds timeout"
            return result
        
        if not result_queue.empty():
            result = result_queue.get()
        else:
            result['error'] = "Test process terminated unexpectedly without returning a result"
        
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
    
    return result

def load_model_safe(model_name: str, model_path: str):
    """Load model with proper configuration"""
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

def generate_response(tokenizer, model, device, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> Optional[str]:
    """Generate response from a model"""
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            if 'token_type_ids' in inputs:
                inputs.pop('token_type_ids')
            
            prompt_length = inputs["input_ids"].shape[1]
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.001,
                do_sample=temperature > 0,
                top_k=10,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_tokens = outputs[0][prompt_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            del inputs
            del outputs
            torch.cuda.empty_cache()
            
            return result.strip() if result.strip() else None
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

def extract_code_from_solution(solution: str) -> str:
    """Extract content from solution tags"""
    match = re.search(r'<solution>(.*?)</solution>', solution, re.DOTALL)
    if match:
        code = match.group(1).strip()
        
        if code.startswith('>>>'):
            code = code[3:].strip()
        
        if code.startswith('```python'):
            code = code[9:].strip()
        elif code.startswith('```'):
            code = code[3:].strip()
        
        if code.endswith('```'):
            code = code[:-3].strip()
        
        return code
    
    return solution.strip()

def create_planning_prompt(task_prompt: str) -> str:
    return f"""Task: {task_prompt}

Write a detailed plan to solve this. Do not write code yet. Focus on:
- Algorithm approach
- Key steps
- Edge cases

Plan:"""

def create_plan_review_prompt(task_prompt: str, previous_plan: str) -> str:
    """Create prompt for DeepSeek to review and improve its own plan"""
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
    """Create prompt for DeepSeek to generate code from the reviewed plan"""
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

def run_three_stage_workflow(
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
    tokenizer,
    model,
    device
) -> Dict:
    """Run the three-stage workflow: DeepSeek plans -> DeepSeek reviews -> DeepSeek codes"""
    
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
        # Step 1: DeepSeek creates plan
        planning_prompt = create_planning_prompt(task_prompt)
        plan = generate_response(tokenizer, model, device, 
                                planning_prompt, temperature_plan, max_tokens)
        
        if plan is None or not plan.strip():
            result['error'] = "Failed to generate plan (empty output)"
            return result
        
        result['plan'] = plan
        
        # Step 2: DeepSeek reviews and improves plan
        review_prompt = create_plan_review_prompt(task_prompt, plan)
        reviewed_plan = generate_response(tokenizer, model, device,
                                         review_prompt, temperature_review, max_tokens)
        
        if reviewed_plan is None or not reviewed_plan.strip():
            result['error'] = "Failed to review plan (empty output)"
            return result
        
        result['reviewed_plan'] = reviewed_plan
        
        # Step 3: DeepSeek generates code from reviewed plan
        coding_prompt = create_coding_prompt(task_prompt, reviewed_plan)
        code = generate_response(tokenizer, model, device,
                                coding_prompt, temperature_code, max_tokens)
        
        if code is None or not code.strip():
            result['error'] = "Failed to generate code (empty output)"
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

def benchmark_workflow(
    dataset: pd.DataFrame,
    temperature_plan: float,
    temperature_review: float,
    temperature_code: float,
    max_tokens: int,
    timeout: int
) -> List[Dict]:
    """Benchmark the DeepSeek→DeepSeek→DeepSeek workflow on the dataset"""
    
    results = []
    total_tasks = len(dataset)
    
    # Load single model
    st.write("### Loading Model")
    tokenizer, model, device = load_model_safe(MODEL_NAME, MODEL_PATH)
    if tokenizer is None or model is None:
        st.error(f"Failed to load {MODEL_NAME}")
        return []
    
    st.success(f"{MODEL_NAME} loaded successfully")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    current_task_placeholder = st.empty()
    completed_tasks_container = st.container()
    
    for idx, row in dataset.iterrows():
        status_text.text(f"Processing task {idx + 1}/{total_tasks}: {row['task_id']} ({row['level']})")
        
        with current_task_placeholder.container():
            st.markdown(f"### Task {idx + 1}/{total_tasks}: {row['task_id']} ({row['level']})")
        
        result = run_three_stage_workflow(
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
            tokenizer=tokenizer,
            model=model,
            device=device
        )
        
        status_icon = "✅" if result['passed'] else "❌"
        with completed_tasks_container:
            with st.expander(
                f"{status_icon} Task {idx + 1}: {row['task_id']} ({row['level']}) - {'PASS' if result['passed'] else 'FAIL'}",
                expanded=False
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Plan (DeepSeek):**")
                    st.markdown(f"**({len(result['plan']) if result['plan'] else 0} chars):**")
                    if result['plan']:
                        st.text(result['plan'])
                    else:
                        st.warning("No plan generated")
                
                with col2:
                    st.markdown("**Reviewed Plan (DeepSeek):**")
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
                    st.markdown("**Test Result:**")
                    if result['passed']:
                        st.success("All tests passed")
                    else:
                        st.error(f"Failed: {result['error']}")
                    
                    if result['code_tested']:
                        with st.expander("View processed code"):
                            st.code(result['code_tested'], language="python")
        
        current_task_placeholder.empty()
        results.append(result)
        progress_bar.progress((idx + 1) / total_tasks)
    
    cleanup_model(model, tokenizer)
    status_text.text("Benchmark complete")
    
    return results

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze benchmark results and compute statistics"""
    df = pd.DataFrame(results)
    
    analysis = {
        'total_tasks': len(df),
        'total_passed': df['passed'].sum(),
        'total_failed': (~df['passed']).sum(),
        'overall_accuracy': df['passed'].mean() * 100,
        'by_difficulty': {}
    }
    
    for level in ['easy', 'middle', 'hard']:
        level_df = df[df['level'] == level]
        if len(level_df) > 0:
            analysis['by_difficulty'][level] = {
                'total': len(level_df),
                'passed': level_df['passed'].sum(),
                'failed': (~level_df['passed']).sum(),
                'accuracy': level_df['passed'].mean() * 100
            }
    
    error_df = df[~df['passed'] & df['error'].notna()]
    error_types = {}
    for error in error_df['error']:
        error_str = str(error)
        if 'infinite loop' in error_str.lower() or 'timeout' in error_str.lower():
            error_types['Infinite loop/Timeout'] = error_types.get('Infinite loop/Timeout', 0) + 1
        elif 'not found' in error_str.lower():
            error_types['Function not found'] = error_types.get('Function not found', 0) + 1
        elif 'assertion' in error_str.lower():
            error_types['Assertion failed'] = error_types.get('Assertion failed', 0) + 1
        elif 'syntaxerror' in error_str.lower():
            error_types['Syntax error'] = error_types.get('Syntax error', 0) + 1
        elif 'nameerror' in error_str.lower():
            error_types['Name error'] = error_types.get('Name error', 0) + 1
        elif 'typeerror' in error_str.lower():
            error_types['Type error'] = error_types.get('Type error', 0) + 1
        else:
            error_types['Other'] = error_types.get('Other', 0) + 1
    
    analysis['error_types'] = error_types
    
    return analysis

def main():
    st.set_page_config(page_title="DeepSeek→DeepSeek→DeepSeek CoT Benchmark", layout="wide")
    
    st.title("DeepSeek → DeepSeek → DeepSeek CoT Benchmark")
    st.markdown("**Workflow:** DeepSeek creates plan → DeepSeek reviews plan → DeepSeek generates code")
    st.markdown("**Dataset:** McEval Generation Tasks (50 tasks)")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    st.sidebar.markdown(f"**Model (all stages):** {MODEL_NAME}")
    st.sidebar.markdown(f"**Tasks:** 50 (first 50 from McEval)")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Planning (DeepSeek)")
    temperature_plan = st.sidebar.slider(
        "Temperature (Plan)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="Temperature for planning generation"
    )
    
    st.sidebar.markdown("### Review (DeepSeek)")
    temperature_review = st.sidebar.slider(
        "Temperature (Review)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="Temperature for plan review"
    )
    
    st.sidebar.markdown("### Coding (DeepSeek)")
    temperature_code = st.sidebar.slider(
        "Temperature (Code)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3, 
        step=0.1,
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
    
    # Load dataset
    st.header("Dataset")
    try:
        dataset = load_dataset()
        st.success(f"Loaded {len(dataset)} tasks from McEval")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            easy_count = len(dataset[dataset['level'] == 'easy'])
            st.metric("Easy Tasks", easy_count)
        with col2:
            middle_count = len(dataset[dataset['level'] == 'middle'])
            st.metric("Middle Tasks", middle_count)
        with col3:
            hard_count = len(dataset[dataset['level'] == 'hard'])
            st.metric("Hard Tasks", hard_count)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return
    
    # Run benchmark
    if st.button("Run Benchmark", type="primary"):
        st.header("Running Benchmark")
        
        results = benchmark_workflow(
            dataset=dataset,
            temperature_plan=temperature_plan,
            temperature_review=temperature_review,
            temperature_code=temperature_code,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        if results:
            analysis = analyze_results(results)
            
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
            
            st.header("Performance by Difficulty")
            for level in ['easy', 'middle', 'hard']:
                if level in analysis['by_difficulty']:
                    stats = analysis['by_difficulty'][level]
                    with st.expander(f"{level.upper()} - {stats['accuracy']:.2f}% accuracy", expanded=True):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total", stats['total'])
                        with col2:
                            st.metric("Passed", stats['passed'])
                        with col3:
                            st.metric("Failed", stats['failed'])
                        with col4:
                            st.metric("Accuracy", f"{stats['accuracy']:.1f}%")
            
            if analysis['error_types']:
                st.header("Error Analysis")
                error_df = pd.DataFrame([
                    {'Error Type': k, 'Count': v} 
                    for k, v in analysis['error_types'].items()
                ]).sort_values('Count', ascending=False)
                st.dataframe(error_df, use_container_width=True)
            
            st.header("Export Results")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_deepseek_deepseek_deepseek_{timestamp}.json"
            
            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                else:
                    return obj
            
            export_data = {
                'workflow': 'DeepSeek (Plan) → DeepSeek (Review) → DeepSeek (Code)',
                'model': MODEL_NAME,
                'model_path': MODEL_PATH,
                'timestamp': timestamp,
                'temperature_plan': float(temperature_plan),
                'temperature_review': float(temperature_review),
                'temperature_code': float(temperature_code),
                'max_tokens': int(max_tokens),
                'timeout': int(timeout),
                'n_tasks': len(dataset),
                'analysis': convert_to_serializable(analysis),
                'results': convert_to_serializable(results)
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download Full Results (JSON)",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()