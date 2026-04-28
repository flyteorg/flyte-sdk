# Team B's image: model training (GPU workload)
#
# Intentionally different from Team A:
#   - Different base: CUDA runtime instead of debian slim
#   - Different Python version: 3.10 (team B's standard)
#   - Different WORKDIR: /workspace (not /app)
#   - Different PYTHONPATH: /workspace
#   - Different system packages: CUDA tools
#
# Pure BYOI: Flyte injects nothing at runtime. Everything must be baked in.
# Each team owns their filesystem layout entirely.
#
# In practice this would be:
#   FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# Using python:3.10-slim here so you can build/test without a GPU machine.

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Team B uses /workspace, not /app. Each team controls their own layout.
WORKDIR /workspace

RUN pip install --no-cache-dir \
    flyte \
    torch==2.1.2 \
    numpy==1.26.4

# Bake in workflow code at /workspace/workflow_code/.
# Python finds it because PYTHONPATH includes /workspace.
COPY workflow_code/ /workspace/workflow_code/

# /workspace is on PYTHONPATH — same import path as Team A's image despite
# the different WORKDIR. Both images expose `import workflow_code.tasks`.
ENV PYTHONPATH=/workspace
