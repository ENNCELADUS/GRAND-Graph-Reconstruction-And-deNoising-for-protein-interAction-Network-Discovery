#!/bin/bash
#SBATCH -J V3
#SBATCH -p hexm
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -t 4-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:NVIDIAA40:4
#SBATCH --output=logs/v3/slurm_%j.out
#SBATCH --error=logs/v3/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=2162352828@qq.com

set -euo pipefail

cd /public/home/wangar2023/grand/
source ~/.bashrc
TARGET_PATH="${1:-configs/v3/ablations/0407}"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync --group dev --locked' on the login node before sbatch."
  exit 1
fi

# Automatically detect number of GPUs from SLURM allocation
NGPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NGPUS GPUs"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

run_config() {
  local config_path="$1"
  local optimization_enabled

  echo "============================================================"
  echo "Running config: ${config_path}"
  echo "============================================================"

  OPTIMIZATION_ENABLED=$(uv run --locked --no-sync --offline python - <<PY
import yaml
from pathlib import Path

config_path = Path("${config_path}")
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}
optimization = config.get("optimization", {})
enabled = bool(isinstance(optimization, dict) and optimization.get("enabled", False))
print("1" if enabled else "0")
PY
)

  if [ "$OPTIMIZATION_ENABLED" = "1" ]; then
    uv run --locked --no-sync --offline python -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" --module src.optimize.run --config "$config_path"
  else
    uv run --locked --no-sync --offline python -m torch.distributed.run --standalone --nproc_per_node="$NGPUS" --module src.run --config "$config_path"
  fi
}

if [ -d "$TARGET_PATH" ]; then
  mapfile -t CONFIG_PATHS < <(find "$TARGET_PATH" -maxdepth 1 -type f -name "*.yaml" | sort)
  if [ "${#CONFIG_PATHS[@]}" -eq 0 ]; then
    echo "No YAML configs found under ${TARGET_PATH}"
    exit 1
  fi
  for config_path in "${CONFIG_PATHS[@]}"; do
    run_config "$config_path"
  done
else
  run_config "$TARGET_PATH"
fi
