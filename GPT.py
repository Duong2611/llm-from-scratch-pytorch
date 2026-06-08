import torch
import torch.nn as nn
import torch.nn.functional as F
import mmap, random, pickle

from bpe_tokenizer import BPETokenizer   # thay thế character-level

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 8
block_size  = 128   # BPE token dày hơn char
max_iters   = 1000
learning_rate = 1e-4
eval_iters  = 50
n_embd  = 128
n_head  = 8
n_layer = 1
dropout = 0.2

# Tokenizer
# Lần đầu: train rồi lưu
#   tokenizer = BPETokenizer()
#   tokenizer.train(open("wizard_of_oz.txt").read(), vocab_size=1000)
#   tokenizer.save("bpe_vocab.pkl")
#
# Các lần sau: load lại
tokenizer = BPETokenizer.load("bpe_vocab.pkl")

vocab_size = tokenizer.vocab_size

# Giữ đúng interface cũ: encode / decode
encode = tokenizer.encode
decode = tokenizer.decode

#Data loading

def get_random_chunk():
    """
    Đọc ngẫu nhiên một đoạn raw bytes từ file,
    decode thành text rồi BPE-encode.
    """
    with open("wizard_of_oz.txt", 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Lấy chunk lớn hơn để bù cho BPE token dài hơn char
            chunk_bytes = block_size * batch_size * 6
            start = random.randint(0, max(0, len(mm) - chunk_bytes))
            mm.seek(start)
            raw = mm.read(chunk_bytes).decode('utf-8', errors='ignore')
            raw = raw.replace('\r', '')
            ids = encode(raw)
            # Cắt đúng số token cần
            need = block_size * batch_size + 1
            if len(ids) < need:
                ids = ids * (need // len(ids) + 1)
            return torch.tensor(ids[:need], dtype=torch.long)

def get_batch():
    data = get_random_chunk()
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size]   for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Model

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril  = torch.tril(torch.ones(block_size, block_size)).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = self.dropout(F.softmax(wei, dim=-1))
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(x))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa  = MultiHeadAttention(n_head, n_embd // n_head)
        self.ff  = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ff(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)
        self.blocks    = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln        = nn.LayerNorm(n_embd)
        self.head      = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok  = self.token_emb(idx)
        pos  = self.pos_emb(torch.arange(T, device=device))
        x    = self.blocks(tok + pos)
        x    = self.ln(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Training 

model     = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch()
        _, loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean()

for i in range(max_iters):
    if i % 100 == 0:
        print(f"step {i} | loss: {estimate_loss().item():.4f}")
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved → model.pkl")

# Inference

while True:
    prompt = input("Prompt: ")
    if prompt == "None":
        break
    context = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    out = model.generate(context, 100)[0].tolist()
    print(decode(out))
