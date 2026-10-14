#!/usr/bin/env bash
# Build llama.cpp + Eliza custom CUDA kernels inside a CUDA devel
# Docker image. Compile/link only: a GPU is not required to run this
# script, but `test-backend-ops -b CUDA0` cannot execute without one.
#
# Usage:
#   scripts/cuda-docker-build.sh                 # default image
#   CUDA_IMAGE=nvidia/cuda:12.6.0-devel-ubuntu24.04 scripts/cuda-docker-build.sh
#
# Output: build artifacts land in ./build-cuda-docker/ on the host.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
BUILD_DIR="${BUILD_DIR:-build-cuda-docker}"
JOBS="${JOBS:-}"

echo "[cuda-docker-build] repo:  $REPO_ROOT"
echo "[cuda-docker-build] image: $CUDA_IMAGE"
echo "[cuda-docker-build] dir:   $BUILD_DIR"

docker run --rm --platform linux/amd64 \
  -v "$REPO_ROOT":/work \
  -w /work \
  -e BUILD_DIR="$BUILD_DIR" \
  -e JOBS="$JOBS" \
  "$CUDA_IMAGE" bash -lc '
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends \
      cmake build-essential git ccache ca-certificates >/dev/null

    : "${JOBS:=$(nproc)}"

    nvcc --version | tail -2 || true
    cmake --version | head -1

    cmake -B "$BUILD_DIR" \
      -DGGML_CUDA=ON \
      -DGGML_METAL=OFF \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"

    cmake --build "$BUILD_DIR" -j "$JOBS" \
      --target llama-cli test-backend-ops test-quantize-fns

    echo "=== Eliza custom CUDA kernels present ==="
    find "$BUILD_DIR" -name "*.o" \
      | grep -E "(turbo-tcq|polarquant|qjl|fused-attn-qjl-tbq|fattn-vec-instance-tbq)" \
      | sort
  '

echo "[cuda-docker-build] done. binaries in $REPO_ROOT/$BUILD_DIR/bin/"
echo "[cuda-docker-build] NOTE: test-backend-ops -b CUDA0 requires a real GPU at runtime."
