import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict
import time

# Model configurations
MODELS = {
    "DeepSeek Coder 7B Instruct" : "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "CodeGemma Instruct": "google/codegemma-7b-it",
    "CodeQwen 1.5 7B": "Qwen/CodeQwen1.5-7B",
    "Code Llama Instruct": "codellama/CodeLlama-7b-Instruct-hf"
}

# Page configuration
st.set_page_config(
    page_title="Cross-Reflection LLM Enhancement",
    page_icon="CR",
    layout="wide"
)

st.markdown("""
<style>
    /* Make code blocks more readable with light background */
    .stCodeBlock {
        background-color: #f8f9fa !important;
    }
    code {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    pre {
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }
    /* Improve text area readability */
    textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'tokenizers' not in st.session_state:
    st.session_state.tokenizers = {}

def load_models(selected_models: List[str], progress_bar, status_text):
    """Load selected models and tokenizers"""
    models = {}
    tokenizers = {}
    
    for idx, model_name in enumerate(selected_models):
        try:
            status_text.text(f"Loading {model_name}...")
            progress_bar.progress((idx) / len(selected_models))
            
            model_id = MODELS[model_name]
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with appropriate settings for smaller models
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if not torch.cuda.is_available():
                model = model.to('cpu')
            
            model.eval()
            
            models[model_name] = model
            tokenizers[model_name] = tokenizer
            
        except Exception as e:
            st.error(f"Error loading {model_name}: {str(e)}")
            return None, None
    
    progress_bar.progress(1.0)
    status_text.text("All models loaded successfully!")
    return models, tokenizers

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 320, temperature: float = 0.1) -> str:
    """Generate response from a model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        
        # Remove token_type_ids if present (some models like CodeQwen don't use it)
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text
        original_prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        if response.startswith(original_prompt):
            response = response[len(original_prompt):]
        
        return response.strip()
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def create_reflection_prompt(task: str, previous_solution: str, iteration: int) -> str:
    """Create a prompt for reflection and improvement with XML encapsulation"""
    if iteration == 1:
        # First iteration - just provide the task
        return f"""<instruction>
<task>
{task}
</task>

Please provide a detailed solution to the task described within the <task> tag. Focus on clarity, correctness, and completeness.

Encapsulate your solution within <solution> tags.
Example response format:
>>> <solution>
Your solution here
</solution>
</instruction>

Your response:"""
    else:
        # Subsequent iterations - provide task and previous solution for review
        return f"""<instruction>
<task>
{task}
</task>

<previousSolution>
{previous_solution}
</previousSolution>

Review the solution encapsulated in the <previousSolution> tag for the task described in the <task> tag.

Please improve the previous solution by considering:
1. Are there any errors or bugs?
2. Can the logic be simplified or optimized?
3. Is the explanation clear and complete?
4. Are there edge cases not handled?

Encapsulate your improved solution within <solution> tags.
Example response format:
>>> <solution>
Your improved solution here
</solution>
</instruction>

Your response:"""

def extract_solution(response: str) -> str:
    """Extract solution from XML tags, or return full response with XML tags stripped"""
    import re
    
    # Try to find content within properly closed <solution> tags
    match = re.search(r'<solution>(.*?)</solution>', response, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
    else:
        # Handle unclosed <solution> tag - extract everything after the opening tag
        match = re.search(r'<solution>\s*(.*)', response, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
        else:
            # Strip any remaining XML tags from the response
            content = re.sub(r'</?solution>', '', response, flags=re.IGNORECASE).strip()
    
    # Strip markdown code fences that might confuse models
    content = re.sub(r'^```[\w]*\n', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n```$', '', content, flags=re.MULTILINE)
    content = re.sub(r'```', '', content)
    
    return content.strip()

def run_cross_reflection(task: str, model_order: List[str], max_tokens: int, temperature: float):
    """Run the cross-reflection process"""
    st.subheader("Cross-Reflection Process")
    
    current_solution = ""
    
    for idx, model_name in enumerate(model_order):
        iteration = idx + 1
        
        with st.expander(f"Iteration {iteration}: {model_name}", expanded=False):
            # Create prompt
            prompt = create_reflection_prompt(task, current_solution, iteration)
            
            st.markdown("**Prompt:**")
            with st.expander("View Full Prompt", expanded=False):
                st.code(prompt, language="xml")
            
            # Generate response
            st.markdown("**Generating response...**")
            start_time = time.time()
            
            model = st.session_state.models[model_name]
            tokenizer = st.session_state.tokenizers[model_name]
            
            with st.spinner(f"Model {iteration} is thinking..."):
                response = generate_response(model, tokenizer, prompt, max_tokens, temperature)
            
            elapsed_time = time.time() - start_time
            
            # Extract solution from XML tags
            extracted_solution = extract_solution(response)
            
            st.markdown(f"**Raw Response** (Generated in {elapsed_time:.2f}s):")
            with st.expander("View Raw Model Output", expanded=False):
                st.text_area(
                    f"Raw Response {iteration}", 
                    response, 
                    height=400, 
                    key=f"raw_response_{iteration}",
                    label_visibility="collapsed"
                )
            
            st.markdown("**Extracted Solution:**")
            st.text_area(
                f"Solution {iteration}", 
                extracted_solution, 
                height=400, 
                key=f"solution_{iteration}",
                label_visibility="collapsed"
            )
            
            # Update current solution for next iteration
            current_solution = extracted_solution
            
            st.success(f"Iteration {iteration} complete")
    
    return current_solution

# Main UI
st.title("Cross-Reflection LLM Enhancement")
st.markdown("""
This app enhances small language model performance through **cross-reflection** with XML encapsulation:
1. Model 1 generates an initial solution
2. Model 2 reviews and improves the solution
3. Model 3 performs final refinement and produces the best output

Solutions are passed between models using XML tags for structured communication.
""")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    st.subheader("Model Selection")
    st.markdown("Select 3 models in order:")
    
    available_models = list(MODELS.keys())
    
    model_1 = st.selectbox("Model 1 (Initial Solution)", available_models, index=0, key="model_1")
    model_2 = st.selectbox("Model 2 (First Refinement)", available_models, index=1, key="model_2")
    model_3 = st.selectbox("Model 3 (Final Refinement)", available_models, index=2, key="model_3")
    
    model_order = [model_1, model_2, model_3]
    
    st.subheader("Generation Parameters")
    max_tokens = st.slider("Max New Tokens", 128, 1024, 320, 64)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    
    st.divider()
    
    # Load models button
    if not st.session_state.models_loaded:
        if st.button("Load Models", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get unique models to load
            unique_models = list(set(model_order))
            
            models, tokenizers = load_models(unique_models, progress_bar, status_text)
            
            if models and tokenizers:
                st.session_state.models = models
                st.session_state.tokenizers = tokenizers
                st.session_state.models_loaded = True
                time.sleep(1)
                st.rerun()
    else:
        st.success("Models loaded!")
        if st.button("Reload Models", use_container_width=True):
            st.session_state.models_loaded = False
            st.session_state.models = {}
            st.session_state.tokenizers = {}
            st.rerun()
    
    st.divider()
    
    # Display GPU info
    if torch.cuda.is_available():
        st.info(f"GPU: {torch.cuda.get_device_name(0)}")

# Main content area
if not st.session_state.models_loaded:
    st.info("Please load the models using the sidebar to get started.")
    
    st.markdown("### How it works:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Step 1: Initial Solution")
        st.markdown("First model generates an initial solution wrapped in XML tags")
    
    with col2:
        st.markdown("#### Step 2: First Refinement")
        st.markdown("Second model reviews the XML-encapsulated solution and improves it")
    
    with col3:
        st.markdown("#### Step 3: Final Refinement")
        st.markdown("Third model performs final polish using the XML-structured context")

else:
    # Task input
    st.subheader("Enter Your Task")
    task = st.text_area(
        "Describe the task you want the models to solve:",
        height=150,
        placeholder="Example: Write a Python function to find the longest common subsequence of two strings."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        run_button = st.button("Run Cross-Reflection", use_container_width=True, type="primary")
    with col2:
        st.info(f"Model sequence: {model_1} → {model_2} → {model_3}")
    
    if run_button:
        if not task.strip():
            st.error("Please enter a task description.")
        else:
            final_solution = run_cross_reflection(task, model_order, max_tokens, temperature)
            
            st.divider()
            st.subheader("Final Output")
            st.markdown("**Best refined solution after 3 iterations:**")
            st.code(final_solution, language="text")
            
            # Download button
            st.download_button(
                label="Download Final Solution",
                data=final_solution,
                file_name="cross_reflection_solution.txt",
                mime="text/plain"
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
Built with Streamlit | Powered by HuggingFace Transformers | Using XML Encapsulation
</div>
""", unsafe_allow_html=True)