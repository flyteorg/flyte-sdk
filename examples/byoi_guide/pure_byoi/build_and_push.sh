#!/usr/bin/env bash
# Build and push both pure BYOI images to a single GHCR repo using tag prefixes.
#
# Each team runs a script like this in their CI pipeline. In pure BYOI, this
# is the ONLY way to ship code changes — there is no code bundle path.
#
# Usage:
#   ./build_and_push.sh                        # build both, push to default registry
#   REGISTRY=ghcr.io/youruser/test-image ./build_and_push.sh
#   TAG=2.2.0 ./build_and_push.sh              # override the image tag
#
# Prerequisites:
#   - Docker logged in to GHCR:
#       echo $GITHUB_TOKEN | docker login ghcr.io -u <username> --password-stdin

set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/your-org/your-image}"
TAG="${TAG:-$(git rev-parse --short HEAD)}"

DATA_PREP_TAG="${REGISTRY}:data-prep-${TAG}"
TRAINING_TAG="${REGISTRY}:training-${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Building data-prep image (Team A)"
echo "    Base:    python:3.11-slim"
echo "    WORKDIR: /app"
echo "    Tag:     ${DATA_PREP_TAG}"
docker build \
  --file "${SCRIPT_DIR}/data_prep/Dockerfile" \
  --tag  "${DATA_PREP_TAG}" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  "${SCRIPT_DIR}"

echo "==> Pushing data-prep image"
docker push "${DATA_PREP_TAG}"

echo ""
echo "==> Building training image (Team B)"
echo "    Base:    python:3.10-slim  (would be nvidia/cuda:12.1.0 in production)"
echo "    WORKDIR: /workspace"
echo "    Tag:     ${TRAINING_TAG}"
docker build \
  --file "${SCRIPT_DIR}/training/Dockerfile" \
  --tag  "${TRAINING_TAG}" \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  "${SCRIPT_DIR}"

echo "==> Pushing training image"
docker push "${TRAINING_TAG}"

echo ""
echo "Done. Update main.py with:"
echo "  DATA_PREP_IMAGE = \"${DATA_PREP_TAG}\""
echo "  TRAINING_IMAGE  = \"${TRAINING_TAG}\""
