#!/usr/bin/env bash
# Validates the CUDA MTP gated_delta_net K-snapshot path on a real NVIDIA GPU.
#
# Background: commit 142e7ac65 ported upstream PR #22673 (MTP per-token state
# snapshots) to the CUDA backend. The kernel emits up to K snapshots into the
# trailing K-token slot region of the state tensor for partial rollback during
# multi-token-prediction speculative decoding. The TODO marker in
# ggml/src/ggml-cuda/gated_delta_net.cu (slot stride vs host-side allocation)
# cannot be closed without execution on an actual GPU.
#
# This script is the one-command answer to "where's the CUDA MTP validation?".
#
# Requirements:
#   - nvcc + CUDA Toolkit 12.4+ in PATH
#   - An NVIDIA GPU visible to the host (`nvidia-smi` must succeed)
#   - cmake >= 3.18, a C++17 toolchain, GNU make or ninja
#
# Usage:
#   bash scripts/cuda-mtp-validate.sh                    # build + run
#   bash scripts/cuda-mtp-validate.sh --rebuild          # wipe build dir first
#   bash scripts/cuda-mtp-validate.sh --build-only       # compile only, skip GPU run
#   bash scripts/cuda-mtp-validate.sh --list-tests       # enumerate planned ops
#
# Optional environment:
#   BUILD_DIR=build-cuda                                 # build directory
#   JOBS=$(nproc)                                        # parallel build jobs
#   MTP_GGUF=/tmp/Qwen3.5-2B-MTP-Q4_K_M.gguf             # smoke model path
#   SKIP_SMOKE=1                                         # skip llama-cli smoke
#   CUDA_ARCHITECTURES="75;80;86;89;90"                  # nvcc target archs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_DIR="${BUILD_DIR:-build-cuda}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
MTP_GGUF="${MTP_GGUF:-/tmp/Qwen3.5-2B-MTP-Q4_K_M.gguf}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-75;80;86;89;90}"

REBUILD=0
BUILD_ONLY=0
LIST_TESTS=0

for arg in "$@"; do
    case "$arg" in
        --rebuild)     REBUILD=1 ;;
        --build-only)  BUILD_ONLY=1 ;;
        --list-tests)  LIST_TESTS=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "[cuda-mtp-validate] unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

# The GATED_DELTA_NET test cases that exercise the MTP K-snapshot path.
# Registered in tests/test-backend-ops.cpp under the
# "K > 1: output keeps the last min(n_tokens, K)" block.
MTP_K_TESTS=(
    "GATED_DELTA_NET(type=f32,head_count=4,head_size=16,n_seq_tokens=2,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=2)"
    "GATED_DELTA_NET(type=f32,head_count=4,head_size=32,n_seq_tokens=4,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=4)"
    "GATED_DELTA_NET(type=f32,head_count=4,head_size=64,n_seq_tokens=4,n_seqs=2,v_repeat=1,permuted=0,kda=0,K=4)"
    "GATED_DELTA_NET(type=f32,head_count=8,head_size=128,n_seq_tokens=4,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=4)"
    "GATED_DELTA_NET(type=f32,head_count=4,head_size=64,n_seq_tokens=4,n_seqs=2,v_repeat=1,permuted=0,kda=1,K=4)"
    "GATED_DELTA_NET(type=f32,head_count=8,head_size=32,n_seq_tokens=4,n_seqs=2,v_repeat=2,permuted=0,kda=1,K=4)"
    "GATED_DELTA_NET(type=f32,head_count=4,head_size=32,n_seq_tokens=8,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=3)"
    "GATED_DELTA_NET(type=f32,head_count=4,head_size=64,n_seq_tokens=16,n_seqs=2,v_repeat=1,permuted=0,kda=0,K=4)"
)

if [ "$LIST_TESTS" = "1" ]; then
    echo "Planned CUDA MTP K-snapshot parity cases (CPU reference vs CUDA0):"
    for t in "${MTP_K_TESTS[@]}"; do
        echo "  - $t"
    done
    echo
    echo "Also runs full \`-o GATED_DELTA_NET\` filter (covers K=1 + K>1 cases)."
    echo "Smoke: llama-cli --spec-type draft-mtp --spec-draft-n-max 2 on \$MTP_GGUF (if present)."
    exit 0
fi

echo "[cuda-mtp-validate] repo:      $REPO_ROOT"
echo "[cuda-mtp-validate] build dir: $BUILD_DIR"
echo "[cuda-mtp-validate] jobs:      $JOBS"
echo "[cuda-mtp-validate] archs:     $CUDA_ARCHITECTURES"

if [ "$REBUILD" = "1" ] && [ -d "$BUILD_DIR" ]; then
    echo "[cuda-mtp-validate] removing $BUILD_DIR (--rebuild)"
    rm -rf "$BUILD_DIR"
fi

# 1. Configure + build
cmake -B "$BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DGGML_METAL=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"

cmake --build "$BUILD_DIR" -j "$JOBS" \
    --target test-backend-ops llama-cli

if [ "$BUILD_ONLY" = "1" ]; then
    echo "[cuda-mtp-validate] --build-only: skipping GPU execution. Build OK."
    exit 0
fi

# Confirm we actually have a GPU before any runtime step.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "::error::nvidia-smi not on PATH; this script requires a real NVIDIA GPU at runtime." >&2
    echo "        Re-run with --build-only on hosts without a GPU." >&2
    exit 3
fi
nvidia-smi -L || { echo "::error::nvidia-smi -L failed; GPU not visible." >&2; exit 3; }

LOG_DIR="${LOG_DIR:-/tmp}"
mkdir -p "$LOG_DIR"

# 2. Full GATED_DELTA_NET op parity sweep (covers K=1 + K>1 registered cases).
echo "[cuda-mtp-validate] running test-backend-ops -b CUDA0 -o GATED_DELTA_NET"
"./$BUILD_DIR/bin/test-backend-ops" -b CUDA0 -o GATED_DELTA_NET 2>&1 \
    | tee "$LOG_DIR/cuda-mtp-backend-ops.log"

# Hard-fail on any FAIL line for GATED_DELTA_NET.
if grep -qE "GATED_DELTA_NET.*FAIL" "$LOG_DIR/cuda-mtp-backend-ops.log"; then
    echo "::error::CUDA GATED_DELTA_NET parity failed against CPU reference."
    exit 1
fi

# Sanity check: confirm K>1 cases were actually scheduled, not skipped.
# Each K>1 case prints its vars including "K=<n>" with n>1.
n_k_gt_1=$(grep -cE "GATED_DELTA_NET.*K=[2-9][0-9]*" "$LOG_DIR/cuda-mtp-backend-ops.log" || true)
echo "[cuda-mtp-validate] K>1 cases observed: $n_k_gt_1"
if [ "$n_k_gt_1" -lt 4 ]; then
    echo "::error::expected at least 4 K>1 cases in GATED_DELTA_NET log, got $n_k_gt_1"
    echo "         (K-snapshot MTP path may not be reached)"
    exit 1
fi

# 3. End-to-end MTP smoke (optional; gated on the unsloth MTP GGUF being present).
if [ "$SKIP_SMOKE" = "1" ]; then
    echo "[cuda-mtp-validate] SKIP_SMOKE=1: skipping llama-cli MTP run."
elif [ ! -f "$MTP_GGUF" ]; then
    echo "[cuda-mtp-validate] SKIP smoke: $MTP_GGUF missing"
    echo "                    (pull from huggingface.co/unsloth/Qwen3.5-2B-MTP-GGUF)"
else
    echo "[cuda-mtp-validate] llama-cli MTP smoke ($MTP_GGUF)"
    timeout 120 "./$BUILD_DIR/bin/llama-cli" \
        -m "$MTP_GGUF" \
        -p "The capital of France is" \
        -n 32 \
        --temp 0 \
        -c 512 \
        -t 4 \
        -ngl 99 \
        --spec-type draft-mtp \
        --spec-draft-n-max 2 \
        2>&1 | tee "$LOG_DIR/cuda-mtp-smoke.log"

    # Smoke succeeds when llama-cli prints either a generation block or the
    # speculative-decoding stats line. Don't require exact text — content
    # depends on model weights.
    if ! grep -qE "Generation:|generated [0-9]+ tokens|^The capital of France" "$LOG_DIR/cuda-mtp-smoke.log"; then
        echo "::error::llama-cli MTP smoke produced no recognisable output"
        exit 1
    fi
fi

echo "[cuda-mtp-validate] CUDA MTP validation: PASS"
