# GPT From Scratch

Xây dựng mô hình ngôn ngữ lớn (LLM) theo kiến trúc GPT từ đầu bằng PyTorch — từ xử lý dữ liệu, BPE tokenizer, đến huấn luyện và sinh văn bản.


## Giới thiệu

Project này implement lại kiến trúc GPT (Generative Pre-trained Transformer) từ nền tảng, giúp hiểu rõ cơ chế hoạt động bên trong của các mô hình ngôn ngữ hiện đại. Bao gồm toàn bộ pipeline: trích xuất dữ liệu, BPE tokenizer (tự implement + HuggingFace), xây dựng mô hình, huấn luyện và sinh văn bản.


## Cấu trúc Project

```
.
├── data-extract.py       # Trích xuất và tiền xử lý dữ liệu từ OpenWebText
├── bpe_tokenizer.py      # BPE tokenizer (thuần Python + HuggingFace wrapper)
├── GPT.py                # Script huấn luyện GPT cơ bản (dùng BPE)
├── llm-full.ipynb        # Notebook GPT cải tiến với Flash Attention, cosine LR
├── wizard_of_oz.txt      # Dữ liệu huấn luyện mẫu
├── output_train.txt      # Dữ liệu training (sinh ra từ data-extract.py)
├── output_val.txt        # Dữ liệu validation (sinh ra từ data-extract.py)
├── bpe_vocab.pkl         # BPE tokenizer đã train (sinh ra từ bpe_tokenizer.py)
└── model.pkl             # Model đã lưu sau huấn luyện
```


## Tính năng

**Tokenizer**
- **BPE tokenizer tự implement** — thuần Python, không cần thư viện ngoài, train từ đầu trên dữ liệu đã có
- **HuggingFace tokenizers wrapper** — dùng `tokenizers` làm backend, giữ nguyên interface `encode`/`decode`
- Dễ swap giữa 2 backend bằng một biến `backend = 'custom' | 'hf'`

**Kiến trúc mô hình**
- **Multi-head Causal Self-Attention** với causal mask
- **Fused QKV projection** — hiệu quả hơn 3 projection riêng lẻ
- **Flash Attention** qua `F.scaled_dot_product_attention` (tự động dùng khi có GPU)
- **Pre-LayerNorm** (GPT-2 style) — ổn định hơn khi huấn luyện
- **Weight tying** giữa embedding và output layer — giảm số tham số
- **Cosine LR schedule** với warmup
- **Top-k sampling + Temperature** khi sinh văn bản

**Hạ tầng**
- Xử lý dữ liệu song song với `ProcessPoolExecutor`
- Hỗ trợ CUDA / MPS (Apple Silicon) / CPU


## Yêu cầu

- Python 3.10+
- PyTorch 2.0+

```bash
pip install torch tqdm
# Nếu dùng HuggingFace backend:
pip install tokenizers
```


## Hướng dẫn sử dụng

### 1. Chuẩn bị dữ liệu

Tải dataset [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) và đặt vào thư mục `openwebtext/`, sau đó chạy:

```bash
python data-extract.py
```

Sinh ra `output_train.txt`, `output_val.txt`.

> **Lưu ý:** Mặc định lấy mẫu 1% dữ liệu (~400MB từ 40GB gốc). Điều chỉnh qua biến `sample_rate`.

### 2. Train BPE tokenizer

```bash
python bpe_tokenizer.py --input output_train.txt --vocab_size 1000 --output bpe_vocab.pkl
```

Các tùy chọn:

| Flag | Mặc định | Mô tả |
|---|---|---|
| `--input` | `output_train.txt` | File text để train |
| `--vocab_size` | `1000` | Kích thước vocab |
| `--output` | `bpe_vocab.pkl` | File lưu tokenizer |
| `--backend` | `custom` | `custom` hoặc `hf` |

> Nếu dùng `wizard_of_oz.txt` để test nhanh: `python bpe_tokenizer.py --input wizard_of_oz.txt`

### 3. Huấn luyện mô hình

```bash
python GPT.py
```

Model lưu vào `model.pkl`. Sau khi huấn luyện, nhập prompt trực tiếp trong terminal để sinh văn bản.

### 4. Huấn luyện với Notebook (khuyến nghị)

```bash
jupyter notebook llm-full.ipynb
```

Chọn backend tokenizer ở cell đầu tiên:

```python
backend    = 'custom'   # hoặc 'hf'
VOCAB_SIZE = 1000
```

Notebook tự động train tokenizer nếu chưa có file cache, rồi tiếp tục train model.


## BPE Tokenizer

`bpe_tokenizer.py` cung cấp 2 class dùng chung interface:

```python
from bpe_tokenizer import BPETokenizer, HFTokenizerWrapper

# --- Custom BPE (thuần Python) ---
tok = BPETokenizer()
tok.train(open("wizard_of_oz.txt").read(), vocab_size=1000)
tok.save("bpe_vocab.pkl")

tok = BPETokenizer.load("bpe_vocab.pkl")
ids  = tok.encode("Hello world!")   # -> List[int]
text = tok.decode(ids)              # -> str

# --- HuggingFace backend ---
tok = HFTokenizerWrapper()
tok.train("wizard_of_oz.txt", vocab_size=1000)
tok.save("hf_bpe")

tok = HFTokenizerWrapper.load("hf_bpe")
ids  = tok.encode("Hello world!")
text = tok.decode(ids)
```

Interface `encode` / `decode` đồng nhất — swap backend không cần sửa code model hay training loop.

## Kiến trúc mô hình

```
Văn bản đầu vào
     │
     ▼
BPE Tokenizer  →  token ids
     │
     ▼
Token Embedding + Positional Embedding
     │
     ▼
[Transformer Block] × n_layer
  ├── Pre-LayerNorm
  ├── Multi-Head Causal Self-Attention (Flash Attention)
  ├── Pre-LayerNorm
  └── FeedForward (GELU, 4× expansion)
     │
     ▼
LayerNorm → Linear (weight-tied) → Logits
     │
     ▼
Top-k Sampling + Temperature  →  văn bản sinh ra
```


## Cấu hình Hyperparameter

| Tham số | `GPT.py` | `llm-full.ipynb` |
|---|---|---|
| `batch_size` | 8 | 32 |
| `block_size` | 128 | 128 |
| `n_embd` | 128 | 128 |
| `n_head` | 8 | 4 |
| `n_layer` | 1 | 3 |
| `dropout` | 0.2 | 0.1 |
| `max_iters` | 1000 | 500 |
| `learning_rate` | 1e-4 | 3e-4 (cosine) |
| `vocab_size` | BPE (1000) | BPE (1000) |

> `block_size` tăng từ 32/64 lên 128 so với character-level vì mỗi BPE token tương đương ~4–6 ký tự.
