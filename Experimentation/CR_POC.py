import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional
import gc

# Model configurations
MODELS = {
    "DeepSeek Coder 7B Instruct": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "CodeGemma Instruct": "google/codegemma-7b-it",
    "CodeQwen 1.5 7B": "Qwen/CodeQwen1.5-7B",
    "Code Llama Instruct": "codellama/CodeLlama-7b-Instruct-hf"
}

# The exact prompt to use
PROMPT = """<task>
from typing import List
 
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    "\"\"\ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    "\"\"\
</task>

<previousCode>
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    "\"\"\ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    "\"\"\
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return True
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

Encapsulate your improved solution within <solution> tags.
Example response format:
>>> <solution>
Your improved solution here
</solution>
</instruction>

Your response:"""

@st.cache_resource
def load_model(model_name: str):
    """Load model and tokenizer with caching"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading {model_name}: {str(e)}")
        return None, None

def generate_response(model_name: str, tokenizer, model, prompt: str, temperature: float = 0.3, max_tokens: int = 512) -> Optional[str]:
    """Generate response from a model"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt):].strip()
        return response
    except Exception as e:
        st.error(f"Error generating response from {model_name}: {str(e)}")
        return None

def main():
    st.title("🐛 Bug Detection Test: 4 Coding Models")
    st.markdown("Testing whether models can catch the logic bug in `has_close_elements`")
    
    st.info("**The Bug**: The final `return True` should be `return False` (when no close elements are found)")
    
    # Sidebar settings
    st.sidebar.header("Generation Settings")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 128, 1024, 512, 64)
    
    # Display the prompt
    with st.expander("📝 View Full Prompt", expanded=False):
        st.code(PROMPT, language="python")
    
    # Run button
    if st.button("🚀 Run All Models", type="primary"):
        results = {}
        
        for model_display_name, model_id in MODELS.items():
            st.markdown(f"### {model_display_name}")
            
            with st.spinner(f"Loading {model_display_name}..."):
                tokenizer, model = load_model(model_id)
            
            if tokenizer is None or model is None:
                st.error(f"Failed to load {model_display_name}")
                continue
            
            with st.spinner(f"Generating response from {model_display_name}..."):
                response = generate_response(model_display_name, tokenizer, model, PROMPT, temperature, max_tokens)
            
            if response:
                st.markdown("**Response:**")
                st.code(response, language="python")
                results[model_display_name] = response
                
                # Check if the model caught the bug
                if "return False" in response and "return True" not in response.split("return False")[-1]:
                    st.success("✅ Appears to have caught the bug!")
                elif "PASS" in response:
                    st.error("❌ Approved incorrect code (returned PASS)")
                else:
                    st.warning("⚠️ Unclear if bug was caught")
            
            # Clean up to free memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
            st.markdown("---")
        
        # Summary
        if results:
            st.markdown("## 📊 Summary")
            st.write(f"Tested {len(results)} models successfully")

if __name__ == "__main__":
    main()