#!/bin/bash
#SBATCH -J TCCIG_BASE
#SBATCH -p hexm
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:NVIDIAA40:1
#SBATCH --output=logs/tccig/slurm_pairwise_baseline_%j.out
#SBATCH --error=logs/tccig/slurm_pairwise_baseline_%j.err
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
SOURCE_RUN_ID="${GRAND_TCCIG_BASELINE_SOURCE_RUN_ID:-01}"
OUTPUT_RUN_ID="${GRAND_TCCIG_BASELINE_OUTPUT_RUN_ID:-pairwise_baseline}"
THRESHOLD="${GRAND_TCCIG_BASELINE_THRESHOLD:-0.5}"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync --group dev --locked' before running TCCIG."
  exit 1
fi

export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

srun uv run --locked --no-sync --offline python -m tccig.raw_pairwise_topology_baseline \
  --config "$CONFIG_PATH" \
  --source-run-id "$SOURCE_RUN_ID" \
  --output-run-id "$OUTPUT_RUN_ID" \
  --threshold "$THRESHOLD"
