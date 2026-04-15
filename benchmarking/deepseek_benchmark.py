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

# DeepSeek Model Configuration
MODEL_NAME = "DeepSeek Coder 7B Instruct"
MODEL_PATH = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"

DATASET_PATH = "/home/demouser/Desktop/121336311/McEval DataSet/McEval_Generation_Tasks.csv"

def load_dataset() -> pd.DataFrame:
    """Load the McEval benchmark dataset"""
    return pd.read_csv(DATASET_PATH)

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
        # Extract from <solution> tags if present
        code_to_test = code.strip()
        
        solution_match = re.search(r'<solution>(.*?)</solution>', code_to_test, re.DOTALL)
        if solution_match:
            # ONLY take what's inside the tags, nothing after </solution>
            code_to_test = solution_match.group(1).strip()
        else:
            # No solution tags found
            result['error'] = "No <solution> tags found in model output"
            return result
        
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

def load_model():
    """Load DeepSeek model with proper configuration"""
    import os
    os.environ["ACCELERATE_USE_FSDP"] = "false"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    
    try:
        st.write(f"Loading {MODEL_NAME}...")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        st.success(f"Loaded {MODEL_NAME} on {device}")
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading {MODEL_NAME}: {e}")
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

def generate_code(tokenizer, model, device, prompt: str, temperature: float = 0.0, max_tokens: int = 512, top_k: int = 50) -> str:
    """Generate code from DeepSeek model"""
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
                top_k=top_k,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            generated_tokens = outputs[0][prompt_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            del inputs
            del outputs
            torch.cuda.empty_cache()
            
            return result.strip()
    except Exception as e:
        st.error(f"Error generating code: {str(e)}")
        return ""

def benchmark_deepseek(dataset: pd.DataFrame, temperature: float, max_tokens: int, top_k: int, timeout: int) -> List[Dict]:
    """Benchmark DeepSeek on the entire dataset"""
    
    tokenizer, model, device = load_model()
    if tokenizer is None or model is None:
        st.error(f"Failed to load {MODEL_NAME}")
        return []
    
    results = []
    total_tasks = len(dataset)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create placeholder for current task
    current_task_placeholder = st.empty()
    
    # Create container for completed tasks
    completed_tasks_container = st.container()
    
    for idx, row in dataset.iterrows():
        status_text.text(f"Processing task {idx + 1}/{total_tasks}: {row['task_id']} ({row['level']})")
        
        # Show current task being processed
        with current_task_placeholder.container():
            st.markdown(f"### ⏳ Task {idx + 1}: {row['task_id']} ({row['level']}) - Generating...")
        
        # Create prompt for DeepSeek (using original XML format)
        original_prompt = row['prompt']
        prompt = f"""<instruction>
<task>
{original_prompt}
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
        
        # Generate code
        generated_code = generate_code(tokenizer, model, device, prompt, temperature, max_tokens, top_k)
        
        # Show generated code with RAW output
        with current_task_placeholder.container():
            st.markdown(f"### ⏳ Task {idx + 1}: {row['task_id']} ({row['level']}) - Testing...")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**RAW Model Output:**")
                st.code(generated_code[:800] if len(generated_code) > 800 else generated_code, language="text")
            
            with col2:
                st.markdown("**Will be processed and tested...**")
                st.text(f"Length: {len(generated_code)} characters")
                st.text(f"Timeout: {timeout}s")
        
        # Test code with timeout
        test_result = run_unit_test(generated_code, row['test'], row['entry_point'], timeout=timeout)
        
        # Move to completed tasks and show result
        status_icon = "✅" if test_result['passed'] else "❌"
        with completed_tasks_container:
            with st.expander(f"{status_icon} Task {idx + 1}: {row['task_id']} ({row['level']}) - {'PASS' if test_result['passed'] else 'FAIL'}", expanded=False):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**RAW Model Output:**")
                    st.code(generated_code[:800] if len(generated_code) > 800 else generated_code, language="text")
                
                with col2:
                    st.markdown("**Processed Code:**")
                    if test_result.get('code_tested'):
                        st.code(test_result['code_tested'][:800] if len(test_result['code_tested']) > 800 else test_result['code_tested'], language="python")
                    else:
                        st.warning("No code extracted")
                
                if test_result['passed']:
                    st.success("All tests passed!")
                else:
                    st.error(f"Failed: {test_result['error']}")
        
        # Clear current task placeholder
        current_task_placeholder.empty()
        
        # Store result
        results.append({
            'task_id': row['task_id'],
            'level': row['level'],
            'entry_point': row['entry_point'],
            'prompt': original_prompt,
            'generated_code': generated_code,
            'passed': test_result['passed'],
            'error': test_result['error'],
            'code_tested': test_result['code_tested']
        })
        
        # Update progress
        progress_bar.progress((idx + 1) / total_tasks)
    
    # Cleanup
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
    
    # Analyze by difficulty
    for level in ['easy', 'middle', 'hard']:
        level_df = df[df['level'] == level]
        if len(level_df) > 0:
            analysis['by_difficulty'][level] = {
                'total': len(level_df),
                'passed': level_df['passed'].sum(),
                'failed': (~level_df['passed']).sum(),
                'accuracy': level_df['passed'].mean() * 100
            }
    
    # Common error types
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
    st.set_page_config(page_title="DeepSeek Coder Benchmark", layout="wide")
    
    st.title("🚀 DeepSeek Coder 7B Benchmark")
    st.markdown("Benchmarking DeepSeek Coder on McEval Dataset")
    
    # Load dataset
    try:
        dataset = load_dataset()
        st.success(f"Loaded dataset: {len(dataset)} tasks")
        
        # Show dataset distribution
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
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.markdown(f"**Model:** {MODEL_NAME}")
    st.sidebar.markdown("---")
    
    temperature = st.sidebar.slider(
        "Temperature", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.0, 
        step=0.1,
        help="Set to 0 for deterministic output"
    )
    
    max_tokens = st.sidebar.slider(
        "Max Tokens", 
        min_value=256, 
        max_value=2048, 
        value=512,
        step=128
    )
    
    top_k = st.sidebar.slider(
        "Top-K Sampling",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        help="Number of highest probability vocabulary tokens to keep for top-k filtering"
    )
    
    timeout = st.sidebar.slider(
        "Test Timeout (seconds)",
        min_value=10,
        max_value=120,
        value=40,
        step=5,
        help="Maximum time allowed for test execution before marking as infinite loop"
    )
    
    # GPU info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 GPU Memory")
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
    
    # Run benchmark
    if st.button("🚀 Run Benchmark", type="primary"):
        st.header(f"Benchmarking: {MODEL_NAME}")
        
        results = benchmark_deepseek(dataset, temperature, max_tokens, top_k, timeout)
        
        if results:
            # Analyze results
            analysis = analyze_results(results)
            
            # Display overall statistics
            st.header("📊 Overall Performance")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tasks", analysis['total_tasks'])
            with col2:
                st.metric("Passed", analysis['total_passed'], delta=f"{analysis['overall_accuracy']:.1f}%")
            with col3:
                st.metric("Failed", analysis['total_failed'])
            with col4:
                st.metric("Accuracy", f"{analysis['overall_accuracy']:.2f}%")
            
            # Display by difficulty
            st.header("📈 Performance by Difficulty")
            
            for level in ['easy', 'middle', 'hard']:
                if level in analysis['by_difficulty']:
                    stats = analysis['by_difficulty'][level]
                    with st.expander(f"{level.upper()} - {stats['accuracy']:.2f}% accuracy", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total", stats['total'])
                        with col2:
                            st.metric("Passed", stats['passed'])
                        with col3:
                            st.metric("Failed", stats['failed'])
            
            # Error analysis
            if analysis['error_types']:
                st.header("🔍 Error Analysis")
                error_df = pd.DataFrame([
                    {'Error Type': k, 'Count': v} 
                    for k, v in analysis['error_types'].items()
                ]).sort_values('Count', ascending=False)
                st.dataframe(error_df, use_container_width=True)
            
            # Detailed results table
            st.header("📋 Detailed Results")
            results_df = pd.DataFrame(results)
            display_df = results_df[['task_id', 'level', 'entry_point', 'passed', 'error']]
            
            # Add filter
            filter_option = st.selectbox(
                "Filter results",
                ["All", "Passed only", "Failed only", "Easy", "Middle", "Hard"]
            )
            
            if filter_option == "Passed only":
                display_df = display_df[display_df['passed']]
            elif filter_option == "Failed only":
                display_df = display_df[~display_df['passed']]
            elif filter_option in ["Easy", "Middle", "Hard"]:
                display_df = display_df[display_df['level'] == filter_option.lower()]
            
            st.dataframe(display_df, use_container_width=True)
            
            # View individual results
            st.header("🔎 Inspect Individual Results")
            task_ids = results_df['task_id'].tolist()
            selected_task = st.selectbox("Select task to view", task_ids)
            
            task_result = results_df[results_df['task_id'] == selected_task].iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Prompt")
                st.code(task_result['prompt'], language="python")
            with col2:
                st.subheader("Generated Code")
                if task_result['code_tested']:
                    st.code(task_result['code_tested'], language="python")
                else:
                    st.code(task_result['generated_code'], language="python")
            
            if not task_result['passed']:
                st.error(f"Error: {task_result['error']}")
            else:
                st.success("All tests passed ✅")
            
            # Download results
            st.header("💾 Export Results")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_deepseek_{timestamp}.json"
            
            def convert_to_serializable(obj):
                """Convert numpy/pandas types to native Python types"""
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
                'model': MODEL_NAME,
                'model_path': MODEL_PATH,
                'timestamp': timestamp,
                'temperature': float(temperature),
                'max_tokens': int(max_tokens),
                'top_k': int(top_k),
                'timeout': int(timeout),
                'analysis': convert_to_serializable(analysis),
                'results': convert_to_serializable(results)
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download Full Results (JSON)",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )

if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for compatibility
    multiprocessing.set_start_method('spawn', force=True)
    main()