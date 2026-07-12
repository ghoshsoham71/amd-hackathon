#!/bin/bash
# =============================================================================
# build_and_push.sh - Build and push to DockerHub for submission
# =============================================================================
# Usage:
#   DOCKERHUB_USER=yourusername ./build_and_push.sh
#   DOCKERHUB_USER=yourusername TAG=v2 ./build_and_push.sh
# =============================================================================

set -euo pipefail

# -- Config --------------------------------------------------------------------
DOCKERHUB_USER="${DOCKERHUB_USER:?Error: set DOCKERHUB_USER env var}"
IMAGE_NAME="amd-track1"
TAG="${TAG:-v14}"
FULL_TAG="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

echo "=================================================="
echo "Building AMD Track 1 Agent"
echo "Image: ${FULL_TAG}"
echo "Platform: linux/amd64 (required by grading harness)"
echo "=================================================="

# -- Ensure buildx builder exists ----------------------------------------------
docker buildx inspect amd-builder &>/dev/null || \
  docker buildx create --name amd-builder --use

# -- Build and push ------------------------------------------------------------
# --platform linux/amd64: REQUIRED - grading VM is linux/amd64
# --push: publish directly to DockerHub (image must be public)
docker buildx build \
  --platform linux/amd64 \
  --tag "${FULL_TAG}" \
  --push \
  .

echo ""
echo "=================================================="
echo "SUCCESS: Image pushed to DockerHub"
echo "  Image tag: ${FULL_TAG}"
echo ""
echo "Submit this tag to the hackathon portal:"
echo "  ${FULL_TAG}"
echo ""
echo "Verify it's publicly pullable:"
echo "  docker pull ${FULL_TAG}"
echo "=================================================="
