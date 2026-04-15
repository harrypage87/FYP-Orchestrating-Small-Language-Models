import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import gc
import time
from typing import Dict

st.set_page_config(page_title="Levenshtein Voting", layout="wide")
st.title("SLM Test App: Code Generation with Levenshtein Voting")

# ================== MODELS ==================

MODELS = {
    "DeepSeek Coder 7B": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "CodeGemma 7B":      "google/codegemma-7b-it",
    "CodeLlama 7B":      "codellama/CodeLlama-7b-Instruct-hf",
}

MODEL_NAMES = list(MODELS.keys())

# ================== SIDEBAR ==================

st.sidebar.header("Controls")

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

if torch.cuda.is_available():
    st.sidebar.info(f"GPU: {torch.cuda.get_device_name(0)}")
    st.sidebar.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
else:
    st.sidebar.warning("No GPU available - using CPU")

# ================== HELPERS ==================

def load_model_safe(model_key: str):
    model_id = MODELS[model_key]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Try cache first, fall back to download
    for local_only in (True, False):
        try:
            if not local_only:
                st.info(f"{model_key} not cached — downloading from HuggingFace...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=local_only,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
            )
            model.eval()
            return tokenizer, model
        except Exception as e:
            if local_only:
                continue  # not cached, try downloading
            st.error(f"Error loading {model_key}: {e}")
            return None, None


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_generation_prompt(model_name: str, task: str) -> str:
    if "CodeGemma" in model_name:
        return (
            "<start_of_turn>user\n"
            "Implement the following Python function. "
            "Wrap your solution in <solution></solution> tags. Return only code, no explanation.\n\n"
            f"{task}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
            "<solution>\n"
        )
    elif "CodeLlama" in model_name:
        return (
            "[INST] Implement the following Python function. "
            "Wrap your solution in <solution></solution> tags. Return only code, no explanation.\n\n"
            f"{task} [/INST]\n"
            "<solution>\n"
        )
    else:  # DeepSeek
        return (
            "### Instruction:\n"
            "Implement the following Python function. "
            "Wrap your solution in <solution></solution> tags. Return only code, no explanation.\n\n"
            f"{task}\n"
            "### Response:\n"
            "<solution>\n"
        )


def format_refinement_prompt(model_name: str, task: str, candidate: str) -> str:
    instruction = (
        "Below is a Python task and a candidate solution. "
        "Make only the minimal edits required to fix any errors. "
        "If the solution is already correct, return it exactly unchanged. "
        "Wrap your output in <solution></solution> tags. Return only code, no explanation.\n\n"
        f"Task:\n{task}\n\n"
        f"Candidate solution:\n{candidate}"
    )
    if "CodeGemma" in model_name:
        return (
            f"<start_of_turn>user\n{instruction}\n<end_of_turn>\n"
            "<start_of_turn>model\n<solution>\n"
        )
    elif "CodeLlama" in model_name:
        return f"[INST] {instruction} [/INST]\n<solution>\n"
    else:
        return f"### Instruction:\n{instruction}\n### Response:\n<solution>\n"


def extract_code(raw: str) -> str:
    solution_match = re.search(r"<solution>(.*?)</solution>", raw, re.DOTALL)
    if solution_match:
        code = solution_match.group(1).strip()
    elif "</solution>" in raw:
        code = raw.split("</solution>")[0].strip()
    else:
        code = raw.strip()

    for marker in ["<end_of_turn>", "### Instruction", "[INST]"]:
        if marker in code:
            code = code.split(marker)[0].strip()

    code = re.sub(r"^```(?:python)?\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    lines = [l for l in code.split("\n") if l.strip() not in ("```", "```python", "```py")]
    return "\n".join(lines).strip()


def generate_code(tokenizer, model, prompt: str, max_new_tokens: int = 512) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = outputs[0][prompt_len:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    except Exception as e:
        return f"Generation error: {e}"


def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


# ================== PROMPT INPUT ==================

st.header("1. Enter Coding Task")
task = st.text_area(
    "Paste the function spec (signature + docstring):",
    height=180,
    placeholder=(
        "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
        '    """Check if in a given list of numbers, any two numbers are closer to each other than the threshold.\n'
        "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
        "    False\n"
        '    """'
    ),
)

# ================== RUN ==================

st.header("2. Generate Solutions & Vote")

if st.button("Run All Models + Levenshtein Vote", type="primary"):
    if not task.strip():
        st.warning("Please enter a coding task before running.")
    else:
        originals: Dict[str, str] = {}
        refinements: Dict[str, Dict[str, str]] = {m: {} for m in MODEL_NAMES}

        # 3 generation + 3 refinement passes = 6 total load cycles
        total_steps = len(MODEL_NAMES) * 2
        progress_bar = st.progress(0)
        status_text = st.empty()

        # ================== STAGE 1: Generation ==================
        st.subheader("Candidate Solutions")

        for idx, model_name in enumerate(MODEL_NAMES):
            status_text.text(f"Loading {model_name} for generation...")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{model_name}**")

            start = time.time()
            tokenizer, model = load_model_safe(model_name)

            if tokenizer is None or model is None:
                originals[model_name] = "Failed to load model."
                st.error(f"{model_name}: failed to load")
            else:
                status_text.text(f"{model_name} generating...")
                prompt = format_generation_prompt(model_name, task)
                raw = generate_code(tokenizer, model, prompt)
                code = extract_code(raw)
                originals[model_name] = code

                with st.expander(f"View {model_name} output", expanded=False):
                    st.code(code, language="python")

                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            elapsed = time.time() - start
            with col2:
                st.metric("Time", f"{elapsed:.1f}s")

            progress_bar.progress((idx + 1) / total_steps)

        # ================== STAGE 2: Refinement ==================
        st.subheader("Refinement Pass")
        st.caption(
            "Each model is loaded once and refines all three candidates before being unloaded. "
            "Levenshtein distance between original and refined output is the implicit vote signal."
        )

        for idx, refiner_name in enumerate(MODEL_NAMES):
            status_text.text(f"Loading {refiner_name} for refinement...")
            start = time.time()

            tokenizer, model = load_model_safe(refiner_name)

            if tokenizer is None or model is None:
                for candidate_name in MODEL_NAMES:
                    refinements[refiner_name][candidate_name] = originals[candidate_name]
                st.error(f"{refiner_name}: failed to load for refinement")
            else:
                for candidate_name in MODEL_NAMES:
                    status_text.text(f"{refiner_name} refining {candidate_name}...")
                    prompt = format_refinement_prompt(refiner_name, task, originals[candidate_name])
                    raw = generate_code(tokenizer, model, prompt)
                    refined = extract_code(raw)
                    refinements[refiner_name][candidate_name] = refined

                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            elapsed = time.time() - start
            st.write(f"**{refiner_name}** refinement pass — {elapsed:.1f}s")

            progress_bar.progress((len(MODEL_NAMES) + idx + 1) / total_steps)

        with st.expander("Inspect all refinements", expanded=False):
            for candidate_name in MODEL_NAMES:
                st.markdown(f"**Candidate: {candidate_name}**")
                rcols = st.columns(3)
                for col, refiner_name in zip(rcols, MODEL_NAMES):
                    with col:
                        st.markdown(f"*{refiner_name}*")
                        st.code(refinements[refiner_name][candidate_name], language="python")

        # ================== STAGE 3: Scoring & Winner ==================
        st.subheader("Levenshtein Vote")

        scores: Dict[str, int] = {}
        breakdown: Dict[str, Dict[str, int]] = {}

        for candidate_name in MODEL_NAMES:
            orig = originals[candidate_name]
            dists = {
                refiner: levenshtein(orig, refinements[refiner][candidate_name])
                for refiner in MODEL_NAMES
            }
            scores[candidate_name] = sum(dists.values())
            breakdown[candidate_name] = dists

        winner = min(MODEL_NAMES, key=lambda m: scores[m])

        st.markdown("### Individual Scores")
        res_cols = st.columns(3)
        for col, name in zip(res_cols, MODEL_NAMES):
            with col:
                is_winner = name == winner
                border = "#2d9e75" if is_winner else "#ddd"
                bg = "#f0faf5" if is_winner else "#fafafa"
                border_width = "2" if is_winner else "1"
                winner_badge = "<p style='color:#2d9e75;font-size:12px;font-weight:600;margin:0 0 4px;'>WINNER</p>" if is_winner else ""
                rows_html = "".join(
                    "<p style='font-size:12px;color:#555;margin:3px 0;'>"
                    + r + ": <strong>" + str(breakdown[name][r]) + "</strong></p>"
                    for r in MODEL_NAMES
                )
                card_html = (
                    "<div style='border:" + border_width + "px solid " + border + ";"
                    "border-radius:8px;padding:16px;background:" + bg + ";min-height:160px;'>"
                    + winner_badge
                    + "<p style='font-size:24px;font-weight:600;margin:0 0 2px;'>" + str(scores[name]) + "</p>"
                    + "<p style='font-size:12px;color:#888;margin:0 0 10px;'>total edits | " + name + "</p>"
                    + rows_html + "</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)

        st.success(f"Levenshtein Vote Complete")
        st.markdown(f"## Winner: {winner}")
        st.markdown(f"**Total edit distance:** {scores[winner]} (lowest across all refiners)")

        st.markdown("### Full Tally")
        for name in sorted(MODEL_NAMES, key=lambda m: scores[m]):
            st.write(f"- **{name}**: {scores[name]} total edits")

        st.markdown("### Winning Solution")
        st.code(originals[winner], language="python")

        with st.expander("Full distance table"):
            rows = []
            for name in MODEL_NAMES:
                row = {"Candidate (generator)": name}
                for refiner in MODEL_NAMES:
                    row[f"{refiner} (edits)"] = breakdown[name][refiner]
                row["Total"] = scores[name]
                rows.append(row)
            st.dataframe(rows, use_container_width=True)

        progress_bar.progress(1.0)
        status_text.text("All done!")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ================== FOOTER ==================
st.divider()
st.caption("Tip: Clear GPU cache between runs to avoid memory issues")