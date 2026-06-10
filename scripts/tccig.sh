#!/bin/bash
#SBATCH -J TCCIG
#SBATCH -p hexm
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA40:4
#SBATCH --output=logs/tccig/slurm_%j.out
#SBATCH --error=logs/tccig/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${GRAND_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
cd "$REPO_ROOT"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

CONFIG_PATH="${1:-configs/tccig/03_fixed_threshold.yaml}"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync --group dev --locked' before running TCCIG."
  exit 1
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

RUNTIME_PATCH_DIR=""
if [ "${GRAND_TCCIG_DISABLE_MHA_FASTPATH:-1}" = "1" ]; then
  RUNTIME_PATCH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tccig-python-XXXXXX")"
  trap 'if [ -n "$RUNTIME_PATCH_DIR" ]; then rm -rf "$RUNTIME_PATCH_DIR"; fi' EXIT
  cat > "${RUNTIME_PATCH_DIR}/sitecustomize.py" <<'PY'
import torch

torch.backends.mha.set_fastpath_enabled(False)
PY
  export PYTHONPATH="${RUNTIME_PATCH_DIR}:${PYTHONPATH}"
  echo "Disabled torch MHA/Transformer fastpath for TCCIG scoring"
fi

if [ "${GRAND_CUDA_LAUNCH_BLOCKING:-0}" = "1" ]; then
  export CUDA_LAUNCH_BLOCKING=1
  echo "CUDA_LAUNCH_BLOCKING=1 enabled for synchronous CUDA error reporting"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS=$(nvidia-smi -L | wc -l)
else
  DETECTED_GPUS=0
fi

REQUESTED_GPUS="${GRAND_TCCIG_GPUS:-$DETECTED_GPUS}"
if [ "$REQUESTED_GPUS" -gt "$DETECTED_GPUS" ]; then
  echo "Requested $REQUESTED_GPUS GPUs but only detected $DETECTED_GPUS; using $DETECTED_GPUS"
  NGPUS="$DETECTED_GPUS"
else
  NGPUS="$REQUESTED_GPUS"
fi

if [ "$NGPUS" -gt 0 ]; then
  echo "Detected $DETECTED_GPUS GPUs; launching TCCIG with $NGPUS process(es)"
  uv run --locked --no-sync --offline python -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" --module tccig.train --config "$CONFIG_PATH"
else
  echo "No GPUs detected; running TCCIG in a single process"
  uv run --locked --no-sync --offline python -m tccig.train --config "$CONFIG_PATH"
fi
