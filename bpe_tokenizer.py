"""
bpe_tokenizer.py
----------------
Byte-Pair Encoding (BPE) tokenizer — implement từ đầu, thuần Python.
Đồng bộ interface với character-level tokenizer cũ:
    encode(text) -> List[int]
    decode(ids)  -> str

Cách dùng:
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=1000)
    tokenizer.save("bpe_vocab.pkl")

    # Hoặc load lại
    tokenizer = BPETokenizer.load("bpe_vocab.pkl")

    ids = tokenizer.encode("Hello world")
    text = tokenizer.decode(ids)

Gợi ý thay thế nhanh bằng HuggingFace:
    Xem cuối file — class HFTokenizerWrapper giữ đúng interface encode/decode.
"""

import re
import pickle
from collections import defaultdict
from tqdm import tqdm


# ─────────────────────────────────────────────
# BPE Tokenizer (thuần Python, từ đầu)
# ─────────────────────────────────────────────

class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple, int] = {}   # (tok_a, tok_b) -> new_id
        self.vocab: dict[int, bytes] = {}    # id -> bytes
        self._encoder: dict[bytes, int] = {} # bytes -> id (cache ngược)

    # ── Huấn luyện ──────────────────────────

    def train(self, text: str, vocab_size: int = 1000, verbose: bool = True):
        """
        Huấn luyện BPE trên chuỗi text.
        vocab_size: tổng số token (bao gồm 256 byte gốc).
        """
        assert vocab_size >= 256, "vocab_size phải >= 256"
        num_merges = vocab_size - 256

        # Bắt đầu với byte-level tokens (0-255)
        ids = list(text.encode("utf-8"))

        # Khởi tạo vocab từ 256 byte cơ bản
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        for i in tqdm(range(num_merges), desc="BPE training", disable=not verbose):
            stats = self._count_pairs(ids)
            if not stats:
                break

            # Chọn cặp xuất hiện nhiều nhất
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i

            # Merge
            ids = self._merge(ids, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

        self._encoder = {v: k for k, v in self.vocab.items()}
        if verbose:
            print(f"Vocab size: {len(self.vocab)} | Merges: {len(self.merges)}")

    # ── Encode / Decode ──────────────────────

    def encode(self, text: str) -> list[int]:
        """Chuyển text -> list token id."""
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = self._count_pairs(ids)
            # Ưu tiên merge theo thứ tự đã học
            best_pair = min(
                stats,
                key=lambda p: self.merges.get(p, float("inf"))
            )
            if best_pair not in self.merges:
                break
            ids = self._merge(ids, best_pair, self.merges[best_pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Chuyển list token id -> text."""
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ── Thuộc tính tiện ích ──────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ── Lưu / Load ──────────────────────────

    def save(self, path: str = "bpe_vocab.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"merges": self.merges, "vocab": self.vocab}, f)
        print(f"Tokenizer saved → {path}")

    @classmethod
    def load(cls, path: str = "bpe_vocab.pkl") -> "BPETokenizer":
        tok = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        tok.merges = data["merges"]
        tok.vocab  = data["vocab"]
        tok._encoder = {v: k for k, v in tok.vocab.items()}
        print(f"Tokenizer loaded ← {path}  (vocab_size={tok.vocab_size})")
        return tok

    # ── Helpers nội bộ ───────────────────────

    @staticmethod
    def _count_pairs(ids: list[int]) -> dict[tuple, int]:
        stats = defaultdict(int)
        for a, b in zip(ids, ids[1:]):
            stats[(a, b)] += 1
        return stats

    @staticmethod
    def _merge(ids: list[int], pair: tuple, new_id: int) -> list[int]:
        result = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result


# ─────────────────────────────────────────────
# HuggingFace Tokenizer Wrapper (tùy chọn thay thế)
# Interface giữ nguyên encode/decode để swap dễ dàng
# pip install tokenizers
# ─────────────────────────────────────────────

class HFTokenizerWrapper:
    """
    Wrapper giữ đúng interface encode(str)->List[int] và decode(List[int])->str
    của BPETokenizer, nhưng dùng tokenizers (HuggingFace) bên dưới.

    Cách dùng:
        tok = HFTokenizerWrapper()
        tok.train("wizard_of_oz.txt", vocab_size=1000)
        tok.save("hf_bpe")

        tok2 = HFTokenizerWrapper.load("hf_bpe")
        ids  = tok2.encode("Hello!")
        text = tok2.decode(ids)
    """

    def __init__(self):
        self._tok = None

    def train(self, file_path: str, vocab_size: int = 1000):
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import ByteLevel
        except ImportError:
            raise ImportError("Chạy: pip install tokenizers")

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
        tokenizer.train([file_path], trainer)
        self._tok = tokenizer
        print(f"HF BPE trained | vocab_size={tokenizer.get_vocab_size()}")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def save(self, directory: str = "hf_bpe"):
        import os; os.makedirs(directory, exist_ok=True)
        self._tok.save(f"{directory}/tokenizer.json")
        print(f"HF tokenizer saved → {directory}/tokenizer.json")

    @classmethod
    def load(cls, directory: str = "hf_bpe") -> "HFTokenizerWrapper":
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError("Chạy: pip install tokenizers")
        obj = cls()
        obj._tok = Tokenizer.from_file(f"{directory}/tokenizer.json")
        print(f"HF tokenizer loaded ← {directory}")
        return obj


# ─────────────────────────────────────────────
# Script chạy trực tiếp: train + lưu tokenizer
# python bpe_tokenizer.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="output_train.txt", help="File text để train")
    parser.add_argument("--vocab_size", default=1000, type=int,     help="Kích thước vocab")
    parser.add_argument("--output",     default="bpe_vocab.pkl",    help="File lưu tokenizer")
    parser.add_argument("--backend",    default="custom",           choices=["custom", "hf"])
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    if args.backend == "custom":
        tok = BPETokenizer()
        tok.train(text, vocab_size=args.vocab_size)
        tok.save(args.output)
    else:
        tok = HFTokenizerWrapper()
        tok.train(args.input, vocab_size=args.vocab_size)
        tok.save("hf_bpe")

    # Kiểm tra nhanh
    sample = text[:200]
    ids    = tok.encode(sample)
    back   = tok.decode(ids)
    print(f"\n--- Kiểm tra ---")
    print(f"Input : {repr(sample[:60])}")
    print(f"Tokens: {ids[:20]} ...")
    print(f"Decode: {repr(back[:60])}")
    print(f"Match : {sample == back}")
