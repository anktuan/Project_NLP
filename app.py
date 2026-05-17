"""
Knowledge Editing Web App -- Flask Backend
==========================================
Provides API endpoints for demonstrating Knowledge Editing
in LLMs using EasyEdit + ROME on Qwen2.5-1.5B (Multilingual).

Key insight: EasyEdit's editor.edit() internally restores weights
after evaluation. We call apply_rome_to_model() directly to keep
the edited weights persistent.
"""

import os
import gc
import sys
import json
import threading
import time
from copy import deepcopy
from datetime import datetime

import torch
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM

# -- Model Configuration --
MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # Multilingual model with Vietnamese support
MODEL_DISPLAY = "Qwen2.5-1.5B (Multilingual)"
HPARAMS_PATH = "hparams/ROME/qwen2.5-1.5b.yaml"

# -- Detect device availability --
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1
print(f"[*] Using device: {DEVICE}")
print(f"[*] Using model: {MODEL_DISPLAY}")

# -- Ensure easyeditor is importable --
sys.path.insert(0, ".")
from easyeditor import ROMEHyperParams
from easyeditor.models.rome.rome_main import apply_rome_to_model
from easyeditor.evaluate import compute_edit_quality

# -- Flask app --
app = Flask(__name__, static_folder="static", static_url_path="")

# -- Global lock (GPU 4 GB -- one request at a time) --
gpu_lock = threading.Lock()

HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

# -- Global state --
state = {
    "model": None,
    "tokenizer": None,
    "hparams": None,
    "edit_history": load_history(),
    "is_edited": False,
}


# -----------------------------------------------------------------
# Startup: load model once
# -----------------------------------------------------------------
def load_model():
    """Load Qwen2.5-1.5B into GPU memory (or CPU if CUDA unavailable)."""
    print(f"[*] Loading {MODEL_DISPLAY} model to {DEVICE}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)
    model.eval()

    # Load ROME hyper-parameters from YAML config
    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)
    # Override device to match runtime detection
    hparams.device = DEVICE_ID

    state["tokenizer"] = tokenizer
    state["model"] = model
    state["hparams"] = hparams
    state["is_edited"] = False
    print(f"[OK] {MODEL_DISPLAY} model and ROME editor ready.")


# -----------------------------------------------------------------
# Helper: score target probability
# -----------------------------------------------------------------
def score_target(model, tokenizer, prompt, target):
    """Return per-token probabilities of *target* given *prompt*."""
    model.eval()
    text = prompt + target
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits

    target_ids = tokenizer(target, return_tensors="pt")["input_ids"][0].to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    probs = []
    tokens = []
    for i, token_id in enumerate(target_ids):
        logit_pos = prompt_len + i - 1
        prob = torch.softmax(logits[0, logit_pos], dim=-1)[token_id].item()
        probs.append(round(prob, 6))
        tokens.append(tokenizer.decode([token_id]))

    return probs, tokens


def generate_text(model, tokenizer, prompt, max_new_tokens=30):
    """Generate continuation for a prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def get_top_predictions(model, tokenizer, prompt, top_k=5):
    """Get top-k predicted next tokens for a prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)

    results = []
    for i in range(top_k):
        token_str = tokenizer.decode([top_ids[i].item()])
        results.append({
            "token": token_str,
            "prob": round(top_probs[i].item(), 6),
        })
    return results


# -----------------------------------------------------------------
# Routes
# -----------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "loaded": state["model"] is not None,
        "is_edited": state["is_edited"],
        "edit_count": len(state["edit_history"]),
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Generate text from the current model."""
    data = request.json
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    with gpu_lock:
        tok = state["tokenizer"]
        model = state["model"]
        text = generate_text(model, tok, prompt)
        top_preds = get_top_predictions(model, tok, prompt)

    return jsonify({
        "prompt": prompt,
        "generated": text,
        "top_predictions": top_preds,
        "is_edited": state["is_edited"],
    })


@app.route("/api/edit", methods=["POST"])
def api_edit():
    """Apply ROME edit DIRECTLY and keep edited weights."""
    data = request.json
    prompt = data.get("prompt", "").strip()
    subject = data.get("subject", "").strip()
    target_new = data.get("target_new", "").strip()
    ground_truth = data.get("ground_truth", "").strip()

    if not all([prompt, subject, target_new]):
        return jsonify({"error": "prompt, subject, and target_new are required"}), 400

    with gpu_lock:
        tok = state["tokenizer"]
        model = state["model"]
        hparams = state["hparams"]

        # -- Before edit --
        before_text = generate_text(model, tok, prompt)
        before_probs, before_tokens = score_target(model, tok, prompt, target_new)
        before_top = get_top_predictions(model, tok, prompt)

        # -- Prepare request in EasyEdit format --
        edit_request = [{
            "prompt": prompt,
            "target_new": target_new,
            "ground_truth": ground_truth if ground_truth else "unknown",
            "subject": subject,
            "portability": {},
            "locality": {},
        }]

        # -- Compute pre-edit metric --
        pre_metric = compute_edit_quality(
            model, hparams.model_name, hparams, tok,
            edit_request[0], hparams.device
        )

        # -- Apply ROME directly (in-place edit, NO weight restore) --
        try:
            edited_model, weights_copy = apply_rome_to_model(
                model, tok, edit_request, hparams,
                copy=False,
                return_orig_weights=True,
            )
            # NOTE: We do NOT restore weights_copy!
            # The model weights are now permanently changed.
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

        state["model"] = edited_model
        state["is_edited"] = True

        # Store weights_copy so we can reset later
        if not hasattr(state, "_weights_copies"):
            state["_weights_copies"] = []
        state["_weights_copies"] = [weights_copy]  # only need latest for reset

        # -- Compute post-edit metric --
        post_metric = compute_edit_quality(
            edited_model, hparams.model_name, hparams, tok,
            edit_request[0], hparams.device
        )

        # -- After edit --
        after_text = generate_text(edited_model, tok, prompt)
        after_probs, after_tokens = score_target(edited_model, tok, prompt, target_new)
        after_top = get_top_predictions(edited_model, tok, prompt)

        # -- Extract metrics --
        pre_acc = float(pre_metric.get("rewrite_acc", [0])[0])
        post_acc = float(post_metric.get("rewrite_acc", [0])[0])

        result = {
            "prompt": prompt,
            "subject": subject,
            "target_new": target_new,
            "ground_truth": ground_truth,
            "before": {
                "generated": before_text,
                "target_probs": before_probs,
                "target_tokens": before_tokens,
                "top_predictions": before_top,
            },
            "after": {
                "generated": after_text,
                "target_probs": after_probs,
                "target_tokens": after_tokens,
                "top_predictions": after_top,
            },
            "metrics": {
                "rewrite_acc_before": pre_acc,
                "rewrite_acc_after": post_acc,
            },
            "timestamp": datetime.now().isoformat(),
        }

        state["edit_history"].append(result)
        save_history(state["edit_history"])

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return jsonify(result)


@app.route("/api/locality", methods=["POST"])
def api_locality():
    """Test locality -- check that unrelated knowledge is preserved."""
    data = request.json
    prompt = data.get("prompt", "").strip()
    expected = data.get("expected", "").strip()

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    with gpu_lock:
        tok = state["tokenizer"]
        model = state["model"]

        generated = generate_text(model, tok, prompt)
        top_preds = get_top_predictions(model, tok, prompt)

        probs, tokens = ([], [])
        if expected:
            probs, tokens = score_target(model, tok, prompt, expected)

    return jsonify({
        "prompt": prompt,
        "expected": expected,
        "generated": generated,
        "target_probs": probs,
        "target_tokens": tokens,
        "top_predictions": top_preds,
        "avg_prob": round(sum(probs) / len(probs), 6) if probs else 0,
    })


@app.route("/api/history")
def api_history():
    return jsonify(state["edit_history"])


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reload a fresh Qwen2.5-1.5B (discard all edits)."""
    with gpu_lock:
        if state["model"] is not None:
            del state["model"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(DEVICE)
        model.eval()
        state["model"] = model
        state["is_edited"] = False
        state["edit_history"] = []
        save_history(state["edit_history"])
        state["_weights_copies"] = []

    return jsonify({"status": f"Model reset to original {MODEL_DISPLAY}"})


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
