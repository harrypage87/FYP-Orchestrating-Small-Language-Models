import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import time
import re
from collections import Counter

st.set_page_config(page_title="Multi-Model Democracy", layout="wide")
st.title("Multi-Model Democracy with Encapsulation")
st.caption("Generate code with multiple models, then have them vote on the best solution")

# ================== MODELS ==================
ALL_MODELS = {
    "DeepSeek Coder 7B Instruct" : "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "CodeGemma Instruct": "google/codegemma-7b-it",
    "CodeQwen 1.5 7B": "Qwen/CodeQwen1.5-7B",
    "Code Llama Instruct": "codellama/CodeLlama-7b-Instruct-hf"
}

# ================== SIDEBAR CONFIGURATION ==================
st.sidebar.header("Configuration")

# Model selection for generation
st.sidebar.subheader("Select Generator Models")
selected_generators = st.sidebar.multiselect(
    "Choose models to generate solutions:",
    options=list(ALL_MODELS.keys()),
    default=list(ALL_MODELS.keys())[:3],
    help="These models will each generate a solution to your task"
)

st.sidebar.divider()

# Model selection for voting
st.sidebar.subheader("Select Voter Models")
selected_voters = st.sidebar.multiselect(
    "Choose models to vote:",
    options=list(ALL_MODELS.keys()),
    default=list(ALL_MODELS.keys())[:3],
    help="These models will vote on which solution is best"
)

st.sidebar.divider()

# Generation parameters
st.sidebar.subheader("Generation Parameters")
gen_temperature = st.sidebar.slider("Generation Temperature", 0.0, 1.0, 0.1, 0.1)
gen_max_tokens = st.sidebar.slider("Generation Max Tokens", 50, 500, 200)

st.sidebar.divider()

# Voting parameters
st.sidebar.subheader("Voting Parameters")
vote_temperature = st.sidebar.slider("Voting Temperature", 0.0, 1.0, 0.1, 0.1)
vote_top_k = st.sidebar.slider("Voting Top-K", 1, 100, 1)
vote_max_tokens = st.sidebar.slider("Voting Max Tokens", 10, 200, 50)

use_special_eos = st.sidebar.checkbox(
    "Use Special EOS Tokens",
    value=False,
    help="Enables EOS tokens. Disable if models output nothing."
)

st.sidebar.divider()

# Cache management
if st.sidebar.button("Clear GPU Cache"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.sidebar.success("GPU cache cleared")

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
            
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except:
                    continue
            
            st.sidebar.success(f"Cleared {size_before_gb:.2f} GB")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

# GPU info
if torch.cuda.is_available():
    st.sidebar.divider()
    st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")

# ================== HELPER FUNCTIONS ==================
def load_model_safe(model_name):
    """Load model with proper configuration."""
    import os
    os.environ["ACCELERATE_USE_FSDP"] = "false"
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
    
    try:
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
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None, None

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
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                repetition_penalty=1.2,
            )
            
            # Extract only generated tokens
            generated_tokens = outputs[0][prompt_length:]
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Combine with prompt
            full_output = prompt + result
            
            return full_output.strip()
    except Exception as e:
        return f"# Generation error: {e}"

def generate_vote(tokenizer, model, device, voting_prompt, max_tokens, temperature, top_k, use_eos):
    """Generate a vote from a model."""
    try:
        with torch.no_grad():
            inputs = tokenizer(voting_prompt, return_tensors="pt").to(device)
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            
            prompt_length = inputs["input_ids"].shape[1]
            
            # Setup EOS tokens - only use default if special EOS is disabled
            eos_token_ids = []
            
            if use_eos:
                # Only add special EOS tokens if explicitly enabled
                if tokenizer.eos_token_id is not None:
                    eos_token_ids.append(tokenizer.eos_token_id)
                    
                special_eos_strings = ["<fim_middle>", "\n--", "\ndef", "\n\n"]
                for eos_string in special_eos_strings:
                    try:
                        encoded = tokenizer.encode(eos_string, add_special_tokens=False)
                        if encoded:
                            eos_token_ids.extend(encoded)
                    except:
                        pass
                eos_token_ids = list(set(eos_token_ids))
            else:
                # Use only the model's default EOS token
                if tokenizer.eos_token_id is not None:
                    eos_token_ids = [tokenizer.eos_token_id]
            
            # Ensure temperature is never exactly 0
            actual_temp = temperature if temperature > 0 else 0.7
            
            # Generate with more permissive settings
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=actual_temp,
                top_k=top_k if top_k > 0 else None,
                do_sample=True,
                eos_token_id=eos_token_ids if eos_token_ids else None,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            )
            
            # Extract only generated tokens
            generated_tokens = outputs[0][prompt_length:]
            
            # Check if anything was generated
            if len(generated_tokens) == 0:
                return "ERROR: Model generated 0 tokens. Try disabling special EOS tokens.", 0
            
            result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return result.strip(), len(generated_tokens)
    except Exception as e:
        return f"Voting error: {e}", 0

def create_voting_prompt(task, solutions):
    """Create the voting prompt in encapsulated format."""
    candidate_letters = ['A', 'B', 'C', 'D', 'E', 'F']
    
    prompt = f"""<instruction>
    <task>
    {task}
    </task>
Among the {len(solutions)} solutions encapsulated in XML tags, which solution best implements the task described within the <task> tag.
Respond with the name of the tag that contains the best solution. Encapsulate the name of the tag in the <best> tag.
Example response:
>>> <best>solution tag name</best>
</instruction>
<allSolutions>
"""
    
    for idx, (model_name, solution) in enumerate(solutions.items()):
        letter = candidate_letters[idx]
        prompt += f"    <solution{letter}>\n"
        prompt += f"    {solution}\n"
        prompt += f"    </solution{letter}>\n"
    
    prompt += "</allSolutions>"
    
    return prompt

def extract_vote(vote_text, valid_solutions):
    """Extract vote from model output and validate it matches a real solution tag."""
    # Try to find <best>X</best>
    match = re.search(r'<best>(.*?)</best>', vote_text, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        # Normalize to solutionX format
        if extracted.lower().startswith('solution'):
            # Extract just the letter part
            letter_match = re.search(r'solution\s*([A-F])', extracted, re.IGNORECASE)
            if letter_match:
                normalized = f"solution{letter_match.group(1).upper()}"
                if normalized in valid_solutions:
                    return normalized
    
    # Try to find solutionX in the text
    match = re.search(r'solution([A-F])', vote_text, re.IGNORECASE)
    if match:
        normalized = f"solution{match.group(1).upper()}"
        if normalized in valid_solutions:
            return normalized
    
    # Try to find just a letter
    match = re.search(r'\b([A-F])\b', vote_text)
    if match:
        normalized = f"solution{match.group(1)}"
        if normalized in valid_solutions:
            return normalized
    
    return None

# ================== MAIN INTERFACE ==================
st.header("Enter Your Coding Task")
task_prompt = st.text_area(
    "Describe the coding task:",
    height=200,
    placeholder="""Example:
from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"
    Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\""""
)

# ================== RUN BUTTON ==================
if st.button("Generate & Vote", type="primary"):
    if not task_prompt.strip():
        st.warning("Please enter a coding task")
    elif not selected_generators:
        st.warning("Please select at least one generator model")
    elif not selected_voters:
        st.warning("Please select at least one voter model")
    else:
        # ================== PHASE 1: GENERATION ==================
        st.header("Phase 1: Code Generation")
        
        solutions = {}
        generation_times = {}
        
        total_steps = len(selected_generators) + len(selected_voters)
        progress_bar = st.progress(0)
        current_step = 0
        
        for model_name in selected_generators:
            st.subheader(f"{model_name}")
            
            start_time = time.time()
            
            # Free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with st.spinner(f"Loading {model_name}..."):
                tokenizer, model, device = load_model_safe(ALL_MODELS[model_name])
            
            if tokenizer and model:
                with st.spinner(f"Generating solution..."):
                    solution = generate_code(tokenizer, model, device, task_prompt, gen_max_tokens, gen_temperature)
                    solutions[model_name] = solution
                    generation_times[model_name] = time.time() - start_time
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    with st.expander("View Generated Code", expanded=False):
                        st.code(solution, language="python")
                with col2:
                    st.metric("Time", f"{generation_times[model_name]:.2f}s")
                
                # Cleanup
                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                solutions[model_name] = "# Failed to generate"
                st.error("Failed to load model")
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        # ================== CREATE VOTING PROMPT ==================
        st.header("Phase 2: Voting Prompt Construction")
        
        voting_prompt = create_voting_prompt(task_prompt, solutions)
        
        # Create list of valid solution tags
        candidate_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        valid_solutions = [f"solution{candidate_letters[i]}" for i in range(len(selected_generators))]
        
        st.success("Voting prompt created in encapsulated format")
        with st.expander("View Full Voting Prompt", expanded=False):
            st.code(voting_prompt, language="xml")
        
        # ================== PHASE 3: VOTING ==================
        st.header("Phase 3: Model Voting")
        
        votes = {}
        vote_details = {}
        
        for model_name in selected_voters:
            st.subheader(f"{model_name} Voting")
            
            # Free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with st.spinner(f"Loading {model_name}..."):
                tokenizer, model, device = load_model_safe(ALL_MODELS[model_name])
            
            if tokenizer and model:
                with st.spinner(f"Generating vote..."):
                    vote_output, num_tokens = generate_vote(
                        tokenizer, model, device, voting_prompt,
                        vote_max_tokens, vote_temperature, vote_top_k, use_special_eos
                    )
                    
                    extracted_vote = extract_vote(vote_output, valid_solutions)
                    votes[model_name] = extracted_vote
                    vote_details[model_name] = {
                        'raw_output': vote_output,
                        'tokens_generated': num_tokens,
                        'extracted_vote': extracted_vote
                    }
                
                # Display vote
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if extracted_vote:
                        st.success(f"Vote: {extracted_vote}")
                    else:
                        st.error(f"Vote: Could not extract valid vote")
                with col2:
                    st.metric("Tokens", num_tokens)
                with col3:
                    with st.expander("Raw Output"):
                        st.text(vote_output)
                
                # Cleanup
                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                st.error("Failed to load model")
            
            current_step += 1
            progress_bar.progress(current_step / total_steps)
        
        # ================== RESULTS ==================
        st.header("Final Results")
        
        # Count votes (only valid ones)
        valid_votes = [v for v in votes.values() if v]
        vote_counts = Counter(valid_votes)
        
        if vote_counts:
            # Display vote distribution
            st.subheader("Vote Distribution")
            
            vote_data = []
            for solution_name, count in vote_counts.most_common():
                vote_data.append({
                    "Solution": solution_name,
                    "Votes": count,
                    "Percentage": f"{(count / len(valid_votes) * 100):.1f}%"
                })
            
            import pandas as pd
            df_votes = pd.DataFrame(vote_data)
            st.dataframe(df_votes, use_container_width=True)
            
            # Show invalid votes if any
            invalid_vote_count = len(votes) - len(valid_votes)
            if invalid_vote_count > 0:
                st.warning(f"{invalid_vote_count} vote(s) were invalid and excluded from results")
            
            # Determine winner
            winner_solution = vote_counts.most_common(1)[0][0]
            winner_votes = vote_counts.most_common(1)[0][1]
            
            st.success(f"Winner: {winner_solution} ({winner_votes} votes)")
            
            # Show winning code
            if winner_solution and winner_solution.lower().startswith("solution"):
                letter = winner_solution[-1].upper()
                candidate_letters = ['A', 'B', 'C', 'D', 'E', 'F']
                if letter in candidate_letters:
                    winner_idx = candidate_letters.index(letter)
                    if winner_idx < len(selected_generators):
                        winner_model = selected_generators[winner_idx]
                        st.subheader(f"Winning Solution from {winner_model}")
                        st.code(solutions[winner_model], language="python")
        else:
            st.warning("No valid votes extracted from any model")
        
        # ================== DETAILED BREAKDOWN ==================
        with st.expander("Detailed Breakdown"):
            st.subheader("Individual Votes")
            for model_name, vote in votes.items():
                if vote:
                    st.write(f" {model_name}: {vote}")
                else:
                    st.write(f" {model_name}: Invalid vote")
                st.caption(f"Raw: {vote_details[model_name]['raw_output'][:100]}...")
            
            st.subheader("Solution Mapping")
            for idx, model_name in enumerate(selected_generators):
                letter = chr(65 + idx)
                st.write(f"Solution{letter}: {model_name}")
        
        progress_bar.progress(1.0)
        st.success("All done!")

# ================== FOOTER ==================
st.divider()