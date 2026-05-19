# Eliza CUDA Runtime Validation

`.github/workflows/eliza-cuda-validation.yml` has two jobs:

1. **`ubuntu-cuda-build`** — compile-only. Runs on every push and PR on
   the GitHub-hosted `ubuntu-24.04` image inside the
   `nvidia/cuda:12.4.1-devel-ubuntu22.04` Docker image. No GPU needed.
   Catches compile/link regressions for the Eliza custom CUDA kernels
   (`turbo-tcq`, `polarquant`, `qjl`, `fused-attn-qjl-tbq`, and the
   `fattn-vec-instance-tbq*` template instances).

2. **`cuda-runtime-validation`** — runs `test-backend-ops` and the
   CUDA MTP `gated_delta_net` K-snapshot parity sweep on a real NVIDIA
   GPU. Requires a self-hosted runner with the `self-hosted-cuda`
   label. **Gated behind a workflow_dispatch input** so it does not
   queue indefinitely while no runner is registered.

## Triggering the runtime job

```
gh workflow run eliza-cuda-validation.yml \
  -R elizaOS/llama.cpp \
  --ref <branch> \
  -f force_cuda_runtime=true
```

The job is skipped automatically on push and PR events, and on
`workflow_dispatch` runs where `force_cuda_runtime` is left at its
default `false`. It executes only when explicitly forced.

## Runner setup

### Hardware

- NVIDIA GPU with **≥ 24 GB VRAM** (Qwen3.5-2B-MTP-Q4_K_M GGUF smoke
  plus `test-backend-ops` working sets). RTX 3090 / 4090 / A6000 /
  A100 / H100 all qualify.
- NVIDIA driver ≥ 550.x (CUDA 12.4 runtime).
- Linux x86_64 (Ubuntu 22.04 or 24.04 preferred — matches the Docker
  base image used by the job).
- Docker ≥ 24 with `nvidia-container-toolkit` installed and the
  `nvidia` Docker runtime registered (`docker info | grep -i runtime`
  should list it).

### Registering with GitHub

1. Repo Settings → Actions → Runners → **New self-hosted runner** →
   Linux / x64. Follow the install + token instructions.
2. When prompted for labels, include both `self-hosted` (default) and
   `self-hosted-cuda`. The workflow targets the pair `[self-hosted,
   self-hosted-cuda]`.
3. Install as a service:
   ```
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```
4. Optionally stage the MTP smoke GGUF at
   `/tmp/Qwen3.5-2B-MTP-Q4_K_M.gguf` to enable the
   `CUDA MTP end-to-end smoke` step (otherwise it self-skips).

### Cloud-runner options

If no in-house GPU is available, spin up an on-demand runner against
one of these providers:

| Provider     | Spec                       | ~Cost (USD/hr) |
| ------------ | -------------------------- | -------------- |
| Lambda Cloud | RTX 6000 Ada (48 GB)       | ~0.80          |
| RunPod       | RTX 4090 (24 GB) spot      | ~0.30–0.50     |
| vast.ai      | RTX 4090 (24 GB) interrupt | ~0.20–0.40     |
| Paperspace   | A6000 (48 GB)              | ~0.76          |

Bring up an instance with CUDA 12.4 + Docker, register it as a runner
with the two labels above, run the workflow, and tear the instance
down. The runtime job typically completes in 20–40 minutes; cost per
validation run is well under USD 1.

## Why this is gated rather than `if: false`

The `if:` expression skips cleanly (workflow shows "Skipped" rather
than "Cancelled" or "Queued"). Once a runner is online, the gate flips
just by passing `-f force_cuda_runtime=true` to `workflow run` — no
workflow edit required. When runtime validation becomes routine
enough to run on every PR, drop the gate and switch the trigger back
to unconditional `pull_request`.
