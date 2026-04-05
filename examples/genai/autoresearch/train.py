"""
Editable training script for the Flyte autoresearch agent (PyTorch LM).

Mirrors the karpathy/autoresearch pattern: **this file is the only code the
agent rewrites** each experiment. Data + BPE runtime live in
``autoresearch_runtime`` (a copy of ``prepare.py`` shipped inside the dataset
bundle).

Metric: **val_bpb** (validation bits per byte) from ``evaluate_bpb`` — **lower
is better**, comparable across vocab/architecture changes (upstream design).

Environment (optional, for faster workshop runs):
- ``AUTORESEARCH_TIME_BUDGET`` — wall-clock training seconds (default 120).
- ``AUTORESEARCH_EVAL_TOKENS`` — validation token budget (see ``prepare.py``).
"""

from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants, agents should not edit these
# ---------------------------------------------------------------------------

ROOT_INPUTS_DIR = "/var/inputs"
DATA_TGZ = f"{ROOT_INPUTS_DIR}/data_tgz"
TOKENIZER_TGZ = f"{ROOT_INPUTS_DIR}/tokenizer_tgz"
DATA_DIR = f"{ROOT_INPUTS_DIR}/data"
TOKENIZER_DIR = f"{ROOT_INPUTS_DIR}/tokenizer"
PREPARE_PY = f"{ROOT_INPUTS_DIR}/prepare.py"
METRICS_PATH = "/var/outputs/metrics_json_str"

# Upstream context length (must match ``prepare.py``).
MAX_SEQ_LEN = 2048

# Smaller default eval than upstream full 40*524288 for practical Flyte demos.
os.environ.setdefault("AUTORESEARCH_EVAL_TOKENS", str(256 * MAX_SEQ_LEN))

# Set the cache directory to the inputs directory.
os.environ["AUTORESEARCH_CACHE"] = ROOT_INPUTS_DIR  # DO NOT CHANGE THIS


def _prepare_dirs():
    import sys
    import tarfile

    sys.path.insert(0, ROOT_INPUTS_DIR)
    with tarfile.open(DATA_TGZ, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    with tarfile.open(TOKENIZER_TGZ, "r:gz") as tar:
        tar.extractall(TOKENIZER_DIR)


# ---------------------------------------------------------------------------
# Architecture / training knobs (agent edits here)
# ---------------------------------------------------------------------------
N_LAYER = 4
N_HEAD = 4
N_EMBD = 256
DROPOUT = 0.0
DEVICE_BATCH_SIZE = 4  # sequences per step (reduce on OOM)
TIME_BUDGET_SEC = int(os.environ.get("AUTORESEARCH_TIME_BUDGET", "120"))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().reshape(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd)
        self.fc2 = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc2(F.gelu(self.fc1(x)))
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Causal LM with the forward signature expected by ``evaluate_bpb``."""

    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(B, T)

        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unknown reduction={reduction!r}")


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    _prepare_dirs()

    import prepare

    assert prepare.MAX_SEQ_LEN == MAX_SEQ_LEN, "Bundle MAX_SEQ_LEN must match train.py block size"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = prepare.Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    model = TinyGPT(
        vocab_size=vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        block_size=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    train_loader = prepare.make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", device=device)

    t0 = time.time()
    model.train()
    steps = 0
    while time.time() - t0 < TIME_BUDGET_SEC:
        x, y, _ = next(train_loader)
        x = x.to(device)
        y = y.to(device)
        loss = model(x, y, reduction="mean")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1

    val_bpb = float(prepare.evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE, device=device))
    model_name = f"TinyGPT-L{N_LAYER}H{N_HEAD}D{N_EMBD}"

    payload = {
        "val_metric": val_bpb,
        "model_name": model_name,
        "notes": (
            f"val_bpb (lower better); device={device.type}; steps={steps}; "
            f"params={_count_params(model)}; time_budget_s={TIME_BUDGET_SEC}"
        ),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload))


if __name__ == "__main__":
    main()
