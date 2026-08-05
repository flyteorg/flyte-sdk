"""
One-time data preparation for autoresearch-style experiments (Flyte example).

Aligned with karpathy/autoresearch ``prepare.py`` — same HuggingFace shard URLs,
BPE training via rustbpe + tiktoken, and the same **val_bpb** evaluation
recipe. See upstream:
https://github.com/karpathy/autoresearch/blob/master/prepare.py

Differences for this repo:
- ``AUTORESEARCH_CACHE`` overrides the cache root (needed for tarball bundles in sandboxes).
- ``AUTORESEARCH_EVAL_TOKENS`` optionally caps validation token count (default: upstream scale).
- ``make_dataloader`` / ``evaluate_bpb`` support CPU or CUDA (upstream assumes CUDA).

Usage (local):
    uv run python prepare.py                  # full prep (download + tokenizer)
    uv run python prepare.py --num-shards 4   # fewer shards for a quick smoke test
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Paths (AUTORESEARCH_CACHE overrides ~/.cache/autoresearch)
# ---------------------------------------------------------------------------


def cache_dir() -> str:
    return os.environ.get(
        "AUTORESEARCH_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "autoresearch"),
    )


def data_dir() -> str:
    return os.path.join(cache_dir(), "data")


def tokenizer_dir() -> str:
    return os.path.join(cache_dir(), "tokenizer")


# ---------------------------------------------------------------------------
# Constants (fixed, match upstream intent)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048  # context length
TIME_BUDGET = 300  # training time budget in seconds (5 minutes) — informational for train.py

# Default matches upstream; override with AUTORESEARCH_EVAL_TOKENS for faster workshop runs.
_DEFAULT_EVAL = 40 * 524288


def eval_tokens() -> int:
    return int(os.environ.get("AUTORESEARCH_EVAL_TOKENS", str(_DEFAULT_EVAL)))


BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542  # the last datashard is shard_06542.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------


def download_single_shard(index: int) -> bool:
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    ddir = data_dir()
    filepath = os.path.join(ddir, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, OSError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2**attempt)
    return False


def download_data(num_shards: int, download_workers: int = 8) -> None:
    """Download training shards + pinned validation shard."""
    ddir = data_dir()
    os.makedirs(ddir, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    existing = sum(1 for i in ids if os.path.exists(os.path.join(ddir, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {ddir}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {ddir}")


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------


def list_parquet_files() -> list[str]:
    """Return sorted list of parquet file paths in the data directory."""
    ddir = data_dir()
    files = sorted(f for f in os.listdir(ddir) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(ddir, f) for f in files]


def text_iterator(max_chars: int = 1_000_000_000, doc_cap: int = 10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    val_path = os.path.join(data_dir(), VAL_FILENAME)
    parquet_paths = [p for p in list_parquet_files() if p != val_path]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer() -> None:
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tdir = tokenizer_dir()
    tokenizer_pkl = os.path.join(tdir, "tokenizer.pkl")
    token_bytes_path = os.path.join(tdir, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {tdir}")
        return

    os.makedirs(tdir, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------


class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc: tiktoken.Encoding) -> None:
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir_path: str | None = None) -> Tokenizer:
        tdir = tokenizer_dir_path or tokenizer_dir()
        with open(os.path.join(tdir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads: int = 8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes() -> torch.Tensor:
    path = os.path.join(tokenizer_dir(), "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")


def _document_batches(split: str, tokenizer_batch_size: int = 128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(data_dir(), VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(
    tokenizer: Tokenizer,
    B: int,
    T: int,
    split: str,
    buffer_size: int = 1000,
    device: torch.device | None = None,
):
    """
    BOS-aligned dataloader with best-fit packing (upstream algorithm).
    CPU or CUDA tensors depending on ``device``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer: list[list[int]] = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

    if device.type == "cuda":
        cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
        gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
        cpu_inputs = cpu_buffer[: B * T].view(B, T)
        cpu_targets = cpu_buffer[B * T :].view(B, T)
        inputs = gpu_buffer[: B * T].view(B, T)
        targets = gpu_buffer[B * T :].view(B, T)

        while True:
            for row_idx in range(B):
                pos = 0
                while pos < row_capacity:
                    while len(doc_buffer) < buffer_size:
                        refill_buffer()

                    remaining = row_capacity - pos
                    best_idx = -1
                    best_len = 0
                    for i, doc in enumerate(doc_buffer):
                        doc_len = len(doc)
                        if doc_len <= remaining and doc_len > best_len:
                            best_idx = i
                            best_len = doc_len

                    if best_idx >= 0:
                        doc = doc_buffer.pop(best_idx)
                        row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                        pos += len(doc)
                    else:
                        shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                        doc = doc_buffer.pop(shortest_idx)
                        row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                        pos += remaining

            cpu_inputs.copy_(row_buffer[:, :-1])
            cpu_targets.copy_(row_buffer[:, 1:])
            gpu_buffer.copy_(cpu_buffer, non_blocking=True)
            yield inputs, targets, epoch
    else:
        while True:
            for row_idx in range(B):
                pos = 0
                while pos < row_capacity:
                    while len(doc_buffer) < buffer_size:
                        refill_buffer()

                    remaining = row_capacity - pos
                    best_idx = -1
                    best_len = 0
                    for i, doc in enumerate(doc_buffer):
                        doc_len = len(doc)
                        if doc_len <= remaining and doc_len > best_len:
                            best_idx = i
                            best_len = doc_len

                    if best_idx >= 0:
                        doc = doc_buffer.pop(best_idx)
                        row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                        pos += len(doc)
                    else:
                        shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                        doc = doc_buffer.pop(shortest_idx)
                        row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                        pos += remaining

            inputs = row_buffer[:, :-1].contiguous().to(device)
            targets = row_buffer[:, 1:].contiguous().to(device)
            yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(model: torch.nn.Module, tokenizer: Tokenizer, batch_size: int, device: torch.device | None = None):
    """
    Bits per byte (BPB): vocab size-independent metric (upstream definition).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    token_bytes = get_token_bytes()
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val", device=device)
    etoks = eval_tokens()
    steps = max(1, etoks // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        x = x.to(device)
        y = y.to(device)
        loss_flat = model(x, y, reduction="none").reshape(-1).detach().cpu()
        y_flat = y.reshape(-1).long().cpu()
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask.to(loss_flat.dtype)).sum().item()
        total_bytes += int(nbytes.sum().item())
    return total_nats / (math.log(2) * total_bytes)
