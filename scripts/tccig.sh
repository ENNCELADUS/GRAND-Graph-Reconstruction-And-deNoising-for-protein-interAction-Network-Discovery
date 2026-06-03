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

CONFIG_PATH="${1:-configs/tccig/start.yaml}"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync --group dev --locked' before running TCCIG."
  exit 1
fi

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if command -v nvidia-smi >/dev/null 2>&1; then
  NGPUS=$(nvidia-smi -L | wc -l)
else
  NGPUS=0
fi

if [ "$NGPUS" -gt 0 ]; then
  echo "Detected $NGPUS GPUs"
  uv run --locked --no-sync --offline python -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" --module tccig.train --config "$CONFIG_PATH"
else
  echo "No GPUs detected; running TCCIG in a single process"
  uv run --locked --no-sync --offline python -m tccig.train --config "$CONFIG_PATH"
fi
