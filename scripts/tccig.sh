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

CONFIG_PATH="${1:-configs/tccig/01.yaml}"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync --group dev --locked' before running TCCIG."
  exit 1
fi

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

DETECTED_GPUS=0
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)"
fi
case "$DETECTED_GPUS" in
  "" | *[!0-9]*) DETECTED_GPUS=0 ;;
esac

REQUESTED_GPUS="${GRAND_TCCIG_GPUS:-$DETECTED_GPUS}"
if [ "$REQUESTED_GPUS" -gt "$DETECTED_GPUS" ]; then
  echo "Requested $REQUESTED_GPUS GPUs but only detected $DETECTED_GPUS; using $DETECTED_GPUS"
  NGPUS="$DETECTED_GPUS"
else
  NGPUS="$REQUESTED_GPUS"
fi

if [ "$NGPUS" -gt 0 ]; then
  echo "Detected $DETECTED_GPUS GPUs; launching TCCIG with $NGPUS process(es)"
  srun uv run --locked --no-sync --offline accelerate launch \
    --num_processes "$NGPUS" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    tccig/train.py \
    --config "$CONFIG_PATH"
else
  echo "No GPUs detected; running TCCIG in a single process"
  uv run --locked --no-sync --offline python -m tccig.train --config "$CONFIG_PATH"
fi
