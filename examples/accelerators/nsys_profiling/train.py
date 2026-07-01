"""Tiny GPU training loop, annotated with NVTX ranges for Nsight Systems.

Run standalone or (the point of this directory) under ``nsys profile`` — the
NVTX ranges (``step_*`` / ``forward`` / ``backward``) show up as labelled
regions on the timeline. Swap this out for your real training script (Lightning,
HF Trainer, etc.); just keep a few NVTX markers so ``--trace=nvtx`` has something
to show.
"""

import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn

dev = torch.device("cuda")
model = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 4096)).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for step in range(20):
    nvtx.range_push(f"step_{step}")
    x = torch.randn(512, 4096, device=dev)
    y = torch.randn(512, 4096, device=dev)

    nvtx.range_push("forward")
    out = model(x)
    loss = loss_fn(out, y)
    nvtx.range_pop()

    nvtx.range_push("backward")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    nvtx.range_pop()

    nvtx.range_pop()

torch.cuda.synchronize()
print("training done, final loss:", loss.item())
