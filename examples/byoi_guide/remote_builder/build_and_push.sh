#!/usr/bin/env bash
# Build and push the BASE images for the remote builder pattern.
#
# These base images contain only system-level dependencies — no workflow code,
# no flyte, no PYTHONPATH. The Flyte engineer layers those on top in main.py.
#
# Run this when:
#   - You add/change a system package (apt-get install / conda install ...)
#   - You change the Python version or package manager in the base image
#
# Do NOT run this for workflow code changes or Python dep changes.
# Those are handled automatically by Flyte when you run `python main.py`.
#
# Usage:
#   ./build_and_push.sh
#   REGISTRY=ghcr.io/myorg/test-image TAG=v2 ./build_and_push.sh

set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/your-org/your-image}"
TAG="${TAG:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building Team A base image (data-prep, conda)"
echo "    continuumio/miniconda3 | WORKDIR /app | no PYTHONPATH"
echo "    Tag: ${REGISTRY}:data-prep-base-${TAG}"
docker build \
  --file "${SCRIPT_DIR}/data_prep/Dockerfile" \
  --tag  "${REGISTRY}:data-prep-base-${TAG}" \
  "${SCRIPT_DIR}/data_prep"

echo "==> Pushing Team A base image"
docker push "${REGISTRY}:data-prep-base-${TAG}"

echo ""
echo "==> Building Team B base image (training, pip venv)"
echo "    python:3.10-slim | venv at /opt/venv | WORKDIR /workspace | no PYTHONPATH"
echo "    Tag: ${REGISTRY}:training-base-${TAG}"
docker build \
  --file "${SCRIPT_DIR}/training/Dockerfile" \
  --tag  "${REGISTRY}:training-base-${TAG}" \
  "${SCRIPT_DIR}/training"

echo "==> Pushing Team B base image"
docker push "${REGISTRY}:training-base-${TAG}"

echo ""
echo "Done. Base image refs (used in main.py Image.from_base()):"
echo "  data-prep-base:  ${REGISTRY}:data-prep-base-${TAG}"
echo "  training-base:   ${REGISTRY}:training-base-${TAG}"
echo ""
echo "Next: run 'python remote_builder/main.py' from v2_guide/ to let Flyte"
echo "build the full image stack (base + flyte install + code bundle)."
