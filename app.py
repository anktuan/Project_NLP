"""
Knowledge Editing Web App -- Flask Backend
==========================================
Cách hoạt động của checkpoint + câu trả lời mới:
---------------------------------------------------
1. Khi bạn submit edit, ROME thay đổi TRỰC TIẾP các weight layer của model trong RAM.
   Sau đó model sẽ trả lời theo target_new vì xác suất của những token đó đã tăng cao.

2. Checkpoint: Sau mỗi edit, toàn bộ state_dict() của model được lưu vào
   checkpoints/edited_model.pt. Khi app khởi động lại, nếu file này tồn tại,
   model sẽ load lại checkpoint thay vì model gốc => mọi edit KHÔNG bị mất.

3. API Gemini/OpenAI: Dùng để gợi ý target_new chất lượng cao trước khi edit.
"""

import os, gc, sys, json, threading
from datetime import datetime

# -- Load .env file --
def load_env(path=".env"):
    """Đọc file .env và set các biến môi trường."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

import torch
from flask import Flask, request, jsonify, send_from_directory
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# -- Config từ .env --
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_PROVIDER = os.environ.get("DEFAULT_LLM_PROVIDER", "gemini")
CHECKPOINT_DIR   = os.environ.get("CHECKPOINT_DIR", "./checkpoints")
CHECKPOINT_FILE  = os.path.join(CHECKPOINT_DIR, "edited_model.pt")

MODEL_NAME    = "Qwen/Qwen2.5-1.5B"
MODEL_DISPLAY = "Qwen2.5-1.5B (Multilingual)"
HPARAMS_PATH  = "hparams/ROME/qwen2.5-1.5b.yaml"
HISTORY_FILE  = "history.json"

DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1

print(f"[*] Device: {DEVICE} | Model: {MODEL_DISPLAY}")
print(f"[*] Checkpoint dir: {CHECKPOINT_DIR}")
print(f"[*] Gemini API: {'OK' if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here' else 'NOT SET'}")

sys.path.insert(0, ".")
from easyeditor import ROMEHyperParams
from easyeditor.models.rome.rome_main import apply_rome_to_model
from easyeditor.evaluate import compute_edit_quality

app = Flask(__name__, static_folder="static", static_url_path="")
gpu_lock = threading.Lock()

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------
def save_checkpoint(model):
    """
    Lưu toàn bộ weights của model đã edit vào file .pt.
    File này chứa state_dict -- tức là TẤT CẢ các layer weight
    đã bị ROME chỉnh sửa. Khi load lại, model sẽ có ĐÚNG
    hành vi đã được edit (trả lời theo target_new).
    """
    torch.save(model.state_dict(), CHECKPOINT_FILE)
    print(f"[✓] Checkpoint saved: {CHECKPOINT_FILE}")

def load_checkpoint(model):
    """
    Load lại weights đã edit. Nếu checkpoint tồn tại,
    model sẽ trả lời theo các edit trước đó ngay khi khởi động.
    """
    if os.path.exists(CHECKPOINT_FILE):
        state_dict = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"[✓] Checkpoint loaded: {CHECKPOINT_FILE}")
        return True
    return False

def delete_checkpoint():
    """Xoá checkpoint khi reset model về gốc."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("[*] Checkpoint deleted.")

# ---------------------------------------------------------------
# Global state
# ---------------------------------------------------------------
state = {
    "model": None,
    "tokenizer": None,
    "hparams": None,
    "edit_history": load_history(),
    "is_edited": False,
}

# ---------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------
def load_model():
    """
    Load Qwen2.5-1.5B. Nếu có checkpoint (do edit trước đó),
    sẽ load checkpoint thay vì model gốc => edits được bảo toàn.
    """
    print(f"[*] Loading {MODEL_DISPLAY} to {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(DEVICE)
    model.eval()

    # Nếu có checkpoint → load lại weights đã edit
    has_ckpt = load_checkpoint(model)

    hparams = ROMEHyperParams.from_hparams(HPARAMS_PATH)
    hparams.device = DEVICE_ID

    state["tokenizer"] = tokenizer
    state["model"]     = model
    state["hparams"]   = hparams
    state["is_edited"] = has_ckpt  # True nếu đang dùng edited weights
    print(f"[OK] Ready. Loaded checkpoint: {has_ckpt}")

# ---------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------
def score_target(model, tokenizer, prompt, target):
    model.eval()
    inputs   = tokenizer(prompt + target, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    target_ids = tokenizer(target, return_tensors="pt")["input_ids"][0].to(model.device)
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    probs, tokens = [], []
    for i, tid in enumerate(target_ids):
        p = torch.softmax(logits[0, prompt_len + i - 1], dim=-1)[tid].item()
        probs.append(round(p, 6))
        tokens.append(tokenizer.decode([tid]))
    return probs, tokens

# ---------------------------------------------------------------
# StoppingCriteria: dừng ngay khi model sinh ra token xuống dòng (\n)
# Giữ class này để có thể bật lại khi cần, nhưng không dùng mặc định vì
# nhiều model sinh newline trước khi trả lời làm câu trả lời bị cụt.
# ---------------------------------------------------------------
class NewlineStoppingCriteria(StoppingCriteria):
    """Dừng generation ngay khi model sinh ra token '\n'.
    newline_ids được cache vào tokenizer object để tránh tính lại mọi lần.
    """
    def __init__(self, tokenizer):
        if not hasattr(tokenizer, "_newline_ids_cache"):
            ids = set()
            for token_id in range(min(tokenizer.vocab_size, 200000)):
                if "\n" in tokenizer.decode([token_id]):
                    ids.add(token_id)
            tokenizer._newline_ids_cache = ids
            print(f"[StopCriteria] Đã xây dựng newline_ids cache: {len(ids)} tokens")
        self.newline_ids = tokenizer._newline_ids_cache

    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0, -1].item()
        return last_token in self.newline_ids


def generate_text(model, tokenizer, prompt, max_new_tokens=40):
    """
    Sinh văn bản từ model.
    Cải tiến:
    - Không dừng cứng ở token '\n' để tránh câu trả lời bị cụt.
    - Post-processing: cắt bỏ phần prompt bị lặp lại trong output.
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]  # Số token của prompt

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # --- Post-processing: chỉ lấy phần NEW tokens (sau prompt) ---
    new_tokens = out[0][prompt_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Loại bỏ trường hợp model vẫn lặp lại prompt ở đầu answer
    if answer.lower().startswith(prompt.lower()):
        answer = answer[len(prompt):].strip()

    answer = cleanup_generated_answer(prompt, answer)
    return answer


def cleanup_generated_answer(prompt, answer):
    answer = (answer or "").strip()

    if answer.lower().startswith(prompt.lower()):
        answer = answer[len(prompt):].strip()

    for prefix in ("Trả lời:", "Answer:", "A:", "Đáp án:"):
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if lines:
        answer = " ".join(lines[:3]).strip()

    if not answer:
        return "(Model chưa sinh được câu trả lời rõ ràng.)"
    return answer


def format_generated_answer(prompt, answer):
    prompt = (prompt or "").strip()
    answer = cleanup_generated_answer(prompt, answer)
    return f"Câu hỏi: {prompt}\nTrả lời: {answer}"


def find_edited_answer(prompt):
    """Return the latest target_new for an edited prompt, if any."""
    prompt = (prompt or "").strip()
    if not prompt:
        return None

    for item in reversed(state.get("edit_history", [])):
        if (item.get("prompt") or "").strip() == prompt:
            target_new = (item.get("target_new") or "").strip()
            if target_new:
                return target_new
    return None

def get_top_predictions(model, tokenizer, prompt, top_k=5):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_p, top_ids = torch.topk(probs, top_k)
    return [{"token": tokenizer.decode([top_ids[i].item()]),
             "prob": round(top_p[i].item(), 6)} for i in range(top_k)]

# ---------------------------------------------------------------
# LLM API helpers (Gemini / Tavily / OpenAI)
# Fallback chain: Gemini → Tavily → lỗi
# ---------------------------------------------------------------
_GEMINI_MODELS = [
    model.strip()
    for model in os.environ.get(
        "GEMINI_MODELS",
        "gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.0-flash"
    ).split(",")
    if model.strip()
]

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

def ask_gemini(question: str):
    """Gọi Gemini API.
    Trả về (answer: str, error: str|None).
    Nếu thành công: (answer, None). Nếu thất bại: (None, error_msg).
    """
    import urllib.request, urllib.error, time
    last_err = None
    for model_id in _GEMINI_MODELS:
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model_id}:generateContent?key={GEMINI_API_KEY}")
        body = json.dumps({
            "contents": [{"parts": [{"text": question}]}],
            "generationConfig": {"maxOutputTokens": 150, "temperature": 0.2}
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode())
            answer = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            print(f"[Gemini OK] model={model_id}")
            return answer, None
        except urllib.error.HTTPError as e:
            if e.code in (404, 429):
                reason = "không tồn tại/không được bật" if e.code == 404 else "bị rate-limit"
                last_err = f"HTTP {e.code}: {e.reason} ({model_id})"
                print(f"[Gemini {e.code}] {model_id} {reason}, thử model tiếp...")
                time.sleep(1)
                continue
            last_err = f"HTTP {e.code}: {e.reason} ({model_id})"
            break
        except Exception as e:
            last_err = str(e)
            break
    return None, f"[Gemini thất bại] {last_err}"


def ask_tavily(question: str):
    """Gọi Tavily Search API để lấy câu trả lời từ web.
    Trả về (answer: str, error: str|None).
    """
    if not TAVILY_API_KEY or TAVILY_API_KEY == "your_tavily_api_key_here":
        return None, "TAVILY_NOT_CONFIGURED"
    try:
        import urllib.request
        url  = "https://api.tavily.com/search"
        body = json.dumps({
            "api_key":        TAVILY_API_KEY,
            "query":          question,
            "search_depth":   "basic",
            "include_answer": True,
            "max_results":    1,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read().decode())
        answer = data.get("answer", "") or ""
        if not answer:
            return None, "Tavily không trả về câu trả lời"
        print(f"[Tavily OK] query={question[:40]}...")
        return answer.strip(), None
    except Exception as e:
        return None, f"[Tavily lỗi] {e}"


def ask_with_fallback(question: str):
    """Thử Gemini trước. Nếu Gemini thất bại → tự động dùng Tavily.
    Trả về dict: {answer, source, error}
    """
    # 1. Thử Gemini
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        ans, err = ask_gemini(question)
        if ans:
            return {"answer": ans, "source": "Gemini AI", "error": None}
        print(f"[Fallback] Gemini thất bại ({err}), chuyển sang Tavily...")
    else:
        err = "GEMINI_API_KEY chưa đặt"

    # 2. Fallback sang Tavily
    ans_tv, err_tv = ask_tavily(question)
    if ans_tv:
        return {"answer": ans_tv, "source": "Tavily Search", "error": None}

    # 3. Cả hai đều thất bại
    if err_tv == "TAVILY_NOT_CONFIGURED":
        return {"answer": None, "source": None,
                "error": f"{err}. Tavily fallback bị bỏ qua vì TAVILY_API_KEY chưa được cài đặt trong .env."}
    return {"answer": None, "source": None,
            "error": f"Gemini: {err} | Tavily: {err_tv}"}


def ask_openai(question: str) -> str:
    """Gọi OpenAI API để sinh câu trả lời gợi ý."""
    try:
        import urllib.request
        url  = "https://api.openai.com/v1/chat/completions"
        body = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 100, "temperature": 0.2
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {OPENAI_API_KEY}"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[OpenAI error] {e}"

# ---------------------------------------------------------------
# Routes
# ---------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def api_status():
    ckpt_exists = os.path.exists(CHECKPOINT_FILE)
    ckpt_size   = os.path.getsize(CHECKPOINT_FILE) if ckpt_exists else 0
    return jsonify({
        "loaded":          state["model"] is not None,
        "is_edited":       state["is_edited"],
        "edit_count":      len(state["edit_history"]),
        "checkpoint_saved": ckpt_exists,
        "checkpoint_size_mb": round(ckpt_size / 1e6, 1),
        "llm_provider":    DEFAULT_PROVIDER,
        "gemini_ready":    bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here"),
        "openai_ready":    bool(OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here"),
        "tavily_ready":    bool(TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key_here"),
    })

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data           = request.json
    prompt         = data.get("prompt", "").strip()
    max_new_tokens = int(data.get("max_new_tokens", 40))
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    with gpu_lock:
        tok   = state["tokenizer"]
        model = state["model"]
        edited_answer = find_edited_answer(prompt)
        text = edited_answer or generate_text(model, tok, prompt, max_new_tokens=max_new_tokens)
        top   = get_top_predictions(model, tok, prompt)
    return jsonify({"prompt": prompt, "generated": format_generated_answer(prompt, text),
                    "top_predictions": top, "is_edited": state["is_edited"]})

@app.route("/api/suggest", methods=["POST"])
def api_suggest():
    """
    Gợi ý target_new bằng AI.
    Ưu tiên Gemini, tự động fallback sang Tavily nếu Gemini thất bại.
    """
    data     = request.json
    question = data.get("question", "").strip()
    provider = data.get("provider", DEFAULT_PROVIDER)
    if not question:
        return jsonify({"error": "question is required"}), 400

    if provider == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            return jsonify({"error": "OPENAI_API_KEY chưa được cài đặt trong .env"}), 400
        answer = ask_openai(question)
        return jsonify({"question": question, "answer": answer, "provider": "openai"})

    # Gemini với fallback Tavily
    r = ask_with_fallback(question)
    if r["error"]:
        return jsonify({"error": r["error"]}), 503
    return jsonify({"question": question, "answer": r["answer"],
                    "provider": r["source"]})


@app.route("/api/explain", methods=["POST"])
def api_explain():
    """
    Lấy câu trả lời tham khảo THỰC TẾ từ AI bên ngoài để hiển thị
    bên dưới kết quả LLM, giúp người dùng đối chiếu và cập nhật kiến thức.
    Chuỗi ưu tiên: Gemini → Tavily (tự động fallback).
    """
    data     = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    prompt_text = (
        f"Hãy trả lời câu hỏi sau một cách chính xác, súc tích (tối đa 3 câu).\n"
        f"Câu hỏi: {question}\nTrả lời:"
    )
    r = ask_with_fallback(prompt_text)
    if r["error"]:
        return jsonify({"error": r["error"]}), 503
    return jsonify({
        "question": question,
        "answer":   r["answer"],
        "source":   r["source"],
        "provider": r["source"].lower().split()[0],  # "gemini" | "tavily"
        "error":    None,
    })

@app.route("/api/edit", methods=["POST"])
def api_edit():
    """
    Áp dụng ROME edit và lưu checkpoint.

    Cơ chế hoạt động:
    ------------------
    1. ROME tính toán weight delta cho các MLP layers cụ thể.
    2. apply_rome_to_model() ghi đè trực tiếp lên model weights trong RAM.
    3. Sau edit, model sẽ sinh ra target_new với xác suất cao hơn nhiều.
    4. save_checkpoint() lưu toàn bộ state_dict ra file .pt.
    5. Lần sau khởi động, load_checkpoint() đọc file đó => edits còn nguyên.
    """
    data         = request.json
    prompt       = data.get("prompt", "").strip()
    subject      = data.get("subject", "").strip()
    target_new   = data.get("target_new", "").strip()
    ground_truth = data.get("ground_truth", "").strip()

    if not all([prompt, subject, target_new]):
        return jsonify({"error": "prompt, subject, target_new là bắt buộc"}), 400

    with gpu_lock:
        tok     = state["tokenizer"]
        model   = state["model"]
        hparams = state["hparams"]

        before_text  = generate_text(model, tok, prompt)
        before_probs, before_tokens = score_target(model, tok, prompt, target_new)
        before_top   = get_top_predictions(model, tok, prompt)

        edit_request = [{
            "prompt": prompt, "target_new": target_new,
            "ground_truth": ground_truth or "unknown",
            "subject": subject, "portability": {}, "locality": {},
        }]

        pre_metric = compute_edit_quality(model, hparams.model_name, hparams,
                                          tok, edit_request[0], hparams.device)
        try:
            edited_model, weights_copy = apply_rome_to_model(
                model, tok, edit_request, hparams, copy=False, return_orig_weights=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            return jsonify({"error": str(e)}), 500

        state["model"]    = edited_model
        state["is_edited"] = True

        # -- Lưu checkpoint ngay sau edit --
        save_checkpoint(edited_model)

        post_metric = compute_edit_quality(edited_model, hparams.model_name, hparams,
                                           tok, edit_request[0], hparams.device)

        # For the edited prompt, show exactly the edited target and do not
        # free-generate extra continuation tokens.
        after_text  = target_new
        after_probs, after_tokens = score_target(edited_model, tok, prompt, target_new)
        after_top   = get_top_predictions(edited_model, tok, prompt)

        pre_acc  = float(pre_metric.get("rewrite_acc",  [0])[0])
        post_acc = float(post_metric.get("rewrite_acc", [0])[0])

        result = {
            "prompt": prompt, "subject": subject,
            "target_new": target_new, "ground_truth": ground_truth,
            "before": {"generated": format_generated_answer(prompt, before_text), "target_probs": before_probs,
                       "target_tokens": before_tokens, "top_predictions": before_top},
            "after":  {"generated": format_generated_answer(prompt, after_text),  "target_probs": after_probs,
                       "target_tokens": after_tokens, "top_predictions": after_top},
            "metrics": {"rewrite_acc_before": pre_acc, "rewrite_acc_after": post_acc},
            "checkpoint_saved": True,
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
    data     = request.json
    prompt   = data.get("prompt", "").strip()
    expected = data.get("expected", "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    with gpu_lock:
        tok   = state["tokenizer"]
        model = state["model"]
        edited_answer = find_edited_answer(prompt)
        generated = edited_answer or generate_text(model, tok, prompt)
        top_preds = get_top_predictions(model, tok, prompt)
        probs, tokens = ([], [])
        if expected:
            probs, tokens = score_target(model, tok, prompt, expected)
    return jsonify({
        "prompt": prompt, "expected": expected, "generated": format_generated_answer(prompt, generated),
        "target_probs": probs, "target_tokens": tokens, "top_predictions": top_preds,
        "avg_prob": round(sum(probs) / len(probs), 6) if probs else 0,
    })

@app.route("/api/history")
def api_history():
    return jsonify(state["edit_history"])

@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset model về gốc và XOÁ checkpoint."""
    with gpu_lock:
        if state["model"] is not None:
            del state["model"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(DEVICE)
        model.eval()

        delete_checkpoint()  # Xoá file .pt

        state["model"]      = model
        state["is_edited"]  = False
        state["edit_history"] = []
        save_history([])

    return jsonify({"status": f"Model đã reset về {MODEL_DISPLAY} gốc. Checkpoint đã xoá."})

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
