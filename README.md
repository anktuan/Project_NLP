# 🧠 Knowledge Editing in Large Language Models

> Demo chỉnh sửa tri thức trong mô hình ngôn ngữ lớn (**Qwen2.5-1.5B**) bằng phương pháp **ROME** (Rank-One Model Editing) — sử dụng framework **EasyEdit**, không cần huấn luyện lại toàn bộ mô hình.

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Chạy ứng dụng](#-chạy-ứng-dụng)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Xử lý sự cố](#-xử-lý-sự-cố)

---

## 🎯 Giới thiệu

Ứng dụng web này minh họa rằng các mô hình ngôn ngữ lớn (LLM) lưu trữ tri thức thực tế trong trọng số MLP, và ta có thể **chỉnh sửa trực tiếp** các tri thức đó mà không cần fine-tune toàn bộ mô hình.

### Các tính năng chính:
- ✅ **Before Edit** — Kiểm tra tri thức gốc của mô hình
- ✅ **Apply ROME** — Chỉnh sửa tri thức bằng can thiệp trọng số MLP
- ✅ **Verify Edit** — Xác nhận mô hình đã cập nhật tri thức mới
- ✅ **Metrics** — Đo lường hiệu quả chỉnh sửa (Rewrite Accuracy, Token Probability)
- ✅ **Locality Test** — Kiểm tra tri thức không liên quan không bị ảnh hưởng
- ✅ **Edit History** — Lưu lại lịch sử chỉnh sửa

---

## 💻 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|:---|:---|
| **Python** | 3.10 trở lên |
| **GPU** | NVIDIA GPU với ≥ 4 GB VRAM (khuyến nghị) |
| **CUDA** | 11.8 trở lên (nếu dùng GPU) |
| **RAM** | ≥ 8 GB |
| **Ổ cứng** | ≥ 10 GB trống (cho model + dependencies) |

> **Lưu ý:** Ứng dụng vẫn chạy được trên CPU nhưng sẽ rất chậm (~5-10 phút mỗi lần edit).

---

## 🔧 Cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/anktuan/Project_NLP.git
cd Project_NLP
```

### Bước 2: Tạo môi trường ảo (Virtual Environment)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt PyTorch (theo GPU của bạn)

Truy cập [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) để chọn lệnh phù hợp.

**Ví dụ cho CUDA 11.8:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Ví dụ cho CUDA 12.1:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Nếu không có GPU (chạy CPU):**
```bash
pip install torch
```

### Bước 4: Cài đặt các thư viện còn lại

```bash
pip install -r requirements.txt
```

> ⏱️ Quá trình cài đặt có thể mất 5-15 phút tùy tốc độ mạng.

---

## 🚀 Chạy ứng dụng

```bash
python app.py
```

**Lần đầu chạy**, ứng dụng sẽ tự động tải model **Qwen2.5-1.5B** (~3 GB) từ HuggingFace. Quá trình này chỉ xảy ra một lần, model sẽ được cache lại cho các lần sau.

Khi thấy thông báo sau nghĩa là đã sẵn sàng:

```
[*] Using device: cuda:0
[*] Using model: Qwen2.5-1.5B (Multilingual)
[*] Loading Qwen2.5-1.5B (Multilingual) model to cuda:0...
[OK] Qwen2.5-1.5B (Multilingual) model and ROME editor ready.
 * Running on http://0.0.0.0:5000
```

### Truy cập web

Mở trình duyệt và truy cập:

```
http://localhost:5000
```

Hoặc nếu truy cập từ máy khác trong cùng mạng LAN:

```
http://<địa-chỉ-IP-máy-chủ>:5000
```

---

## 📖 Hướng dẫn sử dụng

### Step 1: Before Edit — Kiểm tra tri thức gốc

1. Nhập prompt, ví dụ: `Thủ đô của Việt Nam là`
2. Nhấn **Generate** để xem mô hình trả lời gì
3. Quan sát Top-5 dự đoán token tiếp theo

### Step 2: Apply ROME — Chỉnh sửa tri thức

1. Nhập đầy đủ 4 trường:
   - **Prompt**: Câu hỏi chứa tri thức cần sửa
   - **Subject**: Chủ thể (VD: `Việt Nam`)
   - **Ground Truth**: Đáp án cũ (VD: ` Hà Nội`)
   - **Target New**: Đáp án mới (VD: ` Hồ Chí Minh`)
2. Nhấn **Apply ROME Edit**
3. Xem so sánh Before/After

> ⚠️ **Lưu ý:** Target New và Ground Truth nên bắt đầu bằng dấu cách (space) vì tokenizer tách token có space đứng trước.

### Step 3: Verify — Xác nhận chỉnh sửa

- Nhập lại prompt hoặc prompt khác liên quan
- Nhấn **Verify** để kiểm tra mô hình đã học tri thức mới chưa

### Step 4: Metrics — Xem chỉ số đánh giá

- **Rewrite Acc (Before)**: Độ chính xác trước khi edit (thường = 0)
- **Rewrite Acc (After)**: Độ chính xác sau khi edit (mục tiêu = 1.0)
- **Avg Target Prob**: Xác suất trung bình của target tokens

### Step 5: Locality — Kiểm tra tính cục bộ

- Nhập câu hỏi **không liên quan** để đảm bảo edit không phá hỏng tri thức khác
- VD: Nếu edit thủ đô Việt Nam, thì hỏi "Tổng thống Hoa Kỳ là" vẫn phải đúng

### Reset Model

- Nhấn **Reset Model** để khôi phục mô hình về trạng thái gốc (hủy tất cả edits)

---

## 📁 Cấu trúc dự án

```
Project_NLP/
├── app.py                  # Flask backend chính
├── requirements.txt        # Danh sách thư viện Python
├── .gitignore
├── README.md               # File này
│
├── static/                 # Frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── hparams/                # Cấu hình hyperparameters cho các phương pháp editing
│   └── ROME/
│       └── qwen2.5-1.5b.yaml   # Config ROME cho Qwen2.5-1.5B
│
└── easyeditor/             # Thư viện EasyEdit (Knowledge Editing framework)
    ├── models/
    │   └── rome/           # Thuật toán ROME
    ├── evaluate.py         # Hàm đánh giá
    └── ...
```

---

## 🔧 Xử lý sự cố

### ❌ `CUDA out of memory`

Mô hình Qwen2.5-1.5B cần ~3 GB VRAM (FP16). Nếu GPU không đủ VRAM:
- Đóng các ứng dụng khác đang dùng GPU
- Hoặc chạy trên CPU (sẽ chậm hơn nhưng vẫn hoạt động)

### ❌ `ModuleNotFoundError: No module named 'easyeditor'`

Đảm bảo bạn đang chạy từ thư mục gốc của dự án:
```bash
cd Project_NLP
python app.py
```

### ❌ Model tải chậm / lỗi kết nối

Lần đầu chạy cần tải model (~3 GB) từ HuggingFace. Nếu mạng chậm hoặc bị chặn:
```bash
# Đặt mirror HuggingFace (tùy chọn)
set HF_ENDPOINT=https://hf-mirror.com
python app.py
```

### ❌ `RuntimeError: ... expected scalar type Float but found Half`

Có thể do xung đột dtype. Thử sửa `fp16: false` trong file `hparams/ROME/qwen2.5-1.5b.yaml`.

---

## 📚 Tài liệu tham khảo

- **EasyEdit**: [https://github.com/zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit)
- **ROME Paper**: [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- **Qwen2.5**: [https://huggingface.co/Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)

---

## 👤 Tác giả

- GitHub: [anktuan](https://github.com/anktuan)
