#!/bin/bash
#SBATCH -J V3
#SBATCH -p hexm_l40
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAL40:4
#SBATCH --output=logs/v3/slurm_%j.out
#SBATCH --error=logs/v3/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

cd /public/home/wangar2023/relic/
source ~/.bashrc
CONFIG_PATH="${1:-configs/v3.yaml}"

# Automatically detect number of GPUs from SLURM allocation
NGPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NGPUS GPUs"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

OPTIMIZATION_ENABLED=$(uv run --locked python - <<PY
import yaml
from pathlib import Path

config_path = Path("${CONFIG_PATH}")
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
optimization = config.get("optimization", {})
enabled = bool(isinstance(optimization, dict) and optimization.get("enabled", False))
print("1" if enabled else "0")
PY
)

if [ "$OPTIMIZATION_ENABLED" = "1" ]; then
  uv run --locked torchrun --standalone --nproc_per_node="$NGPUS" --module src.optimize.run --config "$CONFIG_PATH"
else
  uv run --locked torchrun --standalone --nproc_per_node="$NGPUS" --module src.run --config "$CONFIG_PATH"
fi
