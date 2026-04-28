# Team B's image: model training
# Uses a pip venv at /opt/venv. Team B has no knowledge of Flyte.
# Python is at /opt/venv/bin/python.
# No PYTHONPATH set, PATH doesn't include /opt/venv/bin.

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir \
        torch==2.1.2+cpu \
        numpy==1.26.4 \
    --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace
