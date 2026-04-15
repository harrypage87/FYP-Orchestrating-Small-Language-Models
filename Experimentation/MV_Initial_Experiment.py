import streamlit as st
import torch
import gc
import time

st.set_page_config(page_title="SLM Test App with Democracy", layout="wide")
st.title("SLM Test App: Code Generation with Democracy Vote")

# ================== MODELS ==================
CANDIDATE_MODELS = {
    "DeepSeek Coder 6.7B": "deepseek-ai/deepseek-coder-7b-base-v1.5",
    "CodeGemma 7B": "google/codegemma-7b",
    "CodeQwen 1.5 7B": "Qwen/CodeQwen1.5-7B",
}

VOTE_SYSTEM_PROMPT = """### Goal
Select the **single candidate solution** that is **most likely to be correct, robust, and pass all unknown unit tests**. The highest priority is placed on **absolute correctness and functional integrity**.

### Analysis Strategy
1.  **Strict Adherence to Task:** Ensure the solution directly and completely addresses all requirements specified in the **<TASK>** description.
2.  **Edge Case Robustness:** Identify and check for common failure points:
    * **Input Validation:** Does the code handle invalid, empty, null, or out-of-range inputs?
    * **Boundary Conditions:** Does it correctly handle minimums, maximums, and critical transitions (e.g., 0, 1, N-1)?
    * **Assumptions:** Does the code make unwarranted assumptions about the input format or data constraints?
3.  **Correctness & Efficiency:**
    * **Algorithmic Soundness:** Is the underlying logic/algorithm mathematically or programmatically correct?
    * **Time Complexity:** Prefer solutions with better asymptotic time complexity (e.g., O(N) over O(N^2)) unless the task implies extremely small data sets where simplicity outweighs speed.
4.  **Code Termination and Integrity (Critical Check):**
    * **Complete Function:** Verify the code snippet is a complete, well-formed function or script (e.g., all necessary imports, function signatures, and closing braces/indentation are present).
    * **Integrity Check:** **Crucially, examine the end of the code (the final few lines) for signs of model failure, such as text repetition, incomplete syntax, or garbled/non-functional code that results from hitting a token limit (MAX_tokens). Any candidate with non-functional or repetitive trailing code must be immediately disqualified, as it guarantees a broken execution environment.**
5.  **Code Quality & Idiomaticity:**
    * **Readability:** Is the code clear, well-structured, and easy to understand?
    * **Error Handling:** Does it use appropriate exception handling or return codes?
    * **Language Idioms:** Does it follow the best practices and common patterns of the language used?

### Output Format
Your response must consist *only* of a reasoning step followed by the final decision. Use the following strict structure:
1.  **REASONING:** A concise, step-by-step summary of the comparative analysis. State the fatal flaw, critical edge-case omission, inefficiency, or **integrity failure** that disqualified the losing candidates, and why the chosen candidate is superior in terms of correctness and robustness.
2.  **VOTE:** The letter of the chosen candidate (e.g., **A**, **B**, or **C**).
"""

# ================== SIDEBAR ==================
st.sidebar.header("Controls")

# Vote settings
st.sidebar.subheader("Democracy Settings")
vote_temperature = st.sidebar.slider(
    "Voter Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    help="Lower = more deterministic. Set to 0.0-0.1 for consistent results."
)

if st.sidebar.button("Clear GPU Cache"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.sidebar.success("GPU cache cleared")

if st.sidebar.button("Clear HuggingFace Cache"):
    import shutil
    from pathlib import Path
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if hf_cache.exists():
        shutil.rmtree(hf_cache)
        hf_cache.mkdir(parents=True, exist_ok=True)
        st.sidebar.success("HuggingFace cache cleared")
    else:
        st.sidebar.warning("HuggingFace cache directory not found")

# Display GPU info
if torch.cuda.is_available():
    st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")
    st.sidebar.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
else:
    st.sidebar.warning("No GPU available - using CPU")

# ================== PROMPT INPUT ==================
st.header("1. Enter Coding Task")
prompt = st.text_area(
    "Describe the coding task:",
    height=150,
    placeholder="Example: Write a Python function to calculate the Fibonacci sequence up to n terms"
)

# ================== HELPER FUNCTIONS ==================
def clean_output(output: str) -> str:
    """Remove trailing whitespace and empty lines."""
    lines = output.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()

def load_model_safe(model_name):
    """Safely load tokenizer and model to CPU/GPU."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )

        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None, None, None

def generate_code(tokenizer, model, device, prompt, max_tokens=150):
    """Generate code using a model."""
    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                do_sample=True
            )

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return clean_output(text)
    except Exception as e:
        return f"Generation error: {e}"

def cast_vote(tokenizer, model, device, voter_name, task, solutions, temperature=0.1):
    try:
        solution_mapping = {}
        candidate_letters = ['A', 'B', 'C', 'D', 'E', 'F']

        vote_prompt = f"{task}\n\n"

        for idx, (model_name, solution) in enumerate(solutions.items()):
            letter = candidate_letters[idx]
            solution_mapping[letter] = model_name
            separator = f"{'-' * 35}{letter}{'-' * 35}"
            vote_prompt += f"{separator}\n{solution}\n"

        vote_prompt += f"{'-' * 71}\n"

        with torch.no_grad():
            # Fix: also check that chat_template is actually set
            has_chat_template = (
                hasattr(tokenizer, 'apply_chat_template') and
                tokenizer.chat_template is not None
            )

            if has_chat_template:
                messages = [
                    {"role": "system", "content": VOTE_SYSTEM_PROMPT},
                    {"role": "user", "content": vote_prompt}
                ]
                inputs = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(device)
            else:
                full_prompt = f"{VOTE_SYSTEM_PROMPT}\n\n{vote_prompt}"
                inputs = tokenizer(full_prompt, return_tensors="pt").to(device)


            if isinstance(inputs, dict) and "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            outputs = model.generate(
                inputs if isinstance(inputs, torch.Tensor) else inputs["input_ids"],
                max_new_tokens=500,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                temperature=temperature,
                do_sample=temperature > 0
            )

            judgment = tokenizer.decode(outputs[0], skip_special_tokens=True)

            reasoning_positions = [i for i in range(len(judgment)) if judgment[i:].startswith("REASONING")]
            if reasoning_positions:
                judgment = judgment[reasoning_positions[-1]:]

            if "assistant\n" in judgment:
                judgment = judgment.split("assistant\n", 1)[1]
            elif "assistant" in judgment and "REASONING" in judgment:
                assistant_idx = judgment.rfind("assistant", 0, judgment.find("REASONING"))
                if assistant_idx != -1:
                    judgment = judgment[assistant_idx + len("assistant"):].strip()

            return judgment.strip(), solution_mapping
    except Exception as e:
        return f"Vote error: {e}", {}

def parse_vote(judgment):
    """Extract reasoning text and vote letter from a judgment string."""
    import re

    reasoning_text = ""
    vote_letter = None

    if "1." in judgment and "Strict Adherence to Task:" in judgment:
        lines = judgment.split('\n')
        actual_reasoning_start = -1
        for i, line in enumerate(lines):
            if re.match(r'^\s*Strict Adherence to Task:', line.strip()):
                actual_reasoning_start = i
                break
        if actual_reasoning_start > 0:
            judgment = '\n'.join(lines[actual_reasoning_start:])

    if "VOTE:" in judgment:
        parts = judgment.split("VOTE:", 1)
        reasoning_section = parts[0]
        vote_section = parts[1] if len(parts) > 1 else ""

        reasoning_section = reasoning_section.replace("REASONING:", "").replace("**REASONING:**", "").strip()

        reasoning_lines = []
        for line in reasoning_section.split('\n'):
            line = line.strip()
            if line and not line.startswith("A concise") and not line.startswith("State the fatal"):
                reasoning_lines.append(line)

        reasoning_text = '\n'.join(reasoning_lines)

        vote_match = re.search(r'([A-F])', vote_section)
        if vote_match:
            vote_letter = vote_match.group(1)

    return reasoning_text, vote_letter

# ================== RUN ALL MODELS BUTTON ==================
st.header("2. Generate Solutions & Vote")

if st.button("Run All Models + Democracy Vote", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a coding task before generating.")
    else:
        results = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Each model: generate code, then vote. 3 models = 6 steps total.
        total_steps = len(CANDIDATE_MODELS) * 2

        # ================== STEP 1: Generate from all candidate models ==================
        st.subheader("Candidate Solutions")

        for idx, (model_name, model_path) in enumerate(CANDIDATE_MODELS.items()):
            status_text.text(f"Generating: {model_name}...")

            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{model_name}**")

            start_time = time.time()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            tokenizer, model, device = load_model_safe(model_path)

            if tokenizer is None or model is None:
                results[model_name] = "Failed to load model."
                st.error(f"{model_name}: Failed to load")
            else:
                output = generate_code(tokenizer, model, device, prompt)
                results[model_name] = output

                with st.expander(f"View {model_name} Output", expanded=False):
                    st.code(output, language="python")

                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            elapsed = time.time() - start_time
            with col2:
                st.metric("Time", f"{elapsed:.2f}s")

            progress_bar.progress((idx + 1) / total_steps)

        # ================== STEP 2: Each model votes ==================
        st.subheader("Democracy Vote")

        vote_tally = {}   # model_name -> vote count
        vote_details = [] # list of dicts per voter

        for idx, (voter_name, voter_path) in enumerate(CANDIDATE_MODELS.items()):
            status_text.text(f"Voting: {voter_name}...")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            voter_tokenizer, voter_model, voter_device = load_model_safe(voter_path)

            if voter_tokenizer and voter_model:
                judgment, solution_mapping = cast_vote(
                    voter_tokenizer, voter_model, voter_device,
                    voter_name, prompt, results,
                    temperature=vote_temperature
                )

                reasoning_text, vote_letter = parse_vote(judgment)

                voted_for = solution_mapping.get(vote_letter) if vote_letter else None

                vote_details.append({
                    "voter": voter_name,
                    "vote_letter": vote_letter,
                    "voted_for": voted_for,
                    "reasoning": reasoning_text,
                    "raw": judgment
                })

                if voted_for:
                    vote_tally[voted_for] = vote_tally.get(voted_for, 0) + 1

                del voter_model, voter_tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                vote_details.append({
                    "voter": voter_name,
                    "vote_letter": None,
                    "voted_for": None,
                    "reasoning": "",
                    "raw": "Failed to load voter model."
                })

            progress_bar.progress((len(CANDIDATE_MODELS) + idx + 1) / total_steps)

        # ================== STEP 3: Display votes & winner ==================
        st.markdown("### Individual Votes")
        for detail in vote_details:
            with st.expander(f"{detail['voter']} → voted for: {detail['voted_for'] or 'Unknown'}"):
                st.markdown(f"**Vote Letter:** {detail['vote_letter'] or 'N/A'}")
                st.markdown(f"**Reasoning:**\n{detail['reasoning'] or detail['raw']}")

        if vote_tally:
            winner = max(vote_tally, key=vote_tally.get)
            winner_votes = vote_tally[winner]

            st.success("Democracy Vote Complete")
            st.markdown(f"## Winner: {winner}")
            st.markdown(f"**Votes received:** {winner_votes} / {len(CANDIDATE_MODELS)}")

            st.markdown("### Full Tally")
            for model_name, votes in sorted(vote_tally.items(), key=lambda x: -x[1]):
                st.write(f"- **{model_name}**: {votes} vote(s)")

            st.markdown("### Winning Solution")
            st.code(results[winner], language="python")
        else:
            st.warning("No valid votes were cast — could not determine a winner.")

        progress_bar.progress(1.0)
        status_text.text("All done!")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ================== FOOTER ==================
st.divider()
