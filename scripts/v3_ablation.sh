#!/bin/bash
#SBATCH -J V3-Ablation-Subgraph-Size
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

cd /public/home/wangar2023/grand/
source ~/.bashrc
TARGET_PATH="${1:-configs/v3/ablations/0427_loss_ablation}"

if [ ! -d ".venv" ]; then
  echo "Missing .venv. Run 'uv sync --group dev --locked' on the login node before sbatch."
  exit 1
fi

detect_torch_cuda_version() {
  uv run --locked --no-sync --offline python - <<'PY'
import torch

print(torch.version.cuda or "")
PY
}

configure_cuda_toolkit() {
  local requested_cuda_version="${GRAND_CUDA_VERSION:-}"
  local resolved_cuda_home=""
  local selected_cuda_version=""
  local torch_cuda_version=""
  local torch_cuda_major=""
  local candidate_path=""

  if command -v module >/dev/null 2>&1; then
    module load cuda/12.1 || true
  fi

  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    resolved_cuda_home="${CUDA_HOME}"
  fi

  torch_cuda_version="$(detect_torch_cuda_version)"
  torch_cuda_version="${torch_cuda_version//$'\n'/}"

  if [[ -z "${requested_cuda_version}" && -n "${torch_cuda_version}" ]]; then
    requested_cuda_version="${torch_cuda_version}"
  fi

  if [[ -z "${resolved_cuda_home}" && -n "${requested_cuda_version}" ]]; then
    candidate_path="/public/software/CUDA/cuda-${requested_cuda_version}"
    if [[ -x "${candidate_path}/bin/nvcc" ]]; then
      resolved_cuda_home="${candidate_path}"
    fi
  fi

  if [[ -z "${resolved_cuda_home}" && -n "${torch_cuda_version}" ]]; then
    torch_cuda_major="${torch_cuda_version%%.*}"
    while IFS= read -r candidate_path; do
      resolved_cuda_home="${candidate_path}"
    done < <(find /public/software/CUDA -maxdepth 1 -type d -name "cuda-${torch_cuda_major}.*" | sort -V)
  fi

  if [[ -z "${resolved_cuda_home}" && -x /public/software/CUDA/cuda-12.1/bin/nvcc ]]; then
    resolved_cuda_home="/public/software/CUDA/cuda-12.1"
  fi

  if [[ -z "${resolved_cuda_home}" ]]; then
    echo "DeepSpeed requires a CUDA toolkit with nvcc, but none was found." >&2
    echo "Set CUDA_HOME or GRAND_CUDA_VERSION before running." >&2
    exit 1
  fi

  export CUDA_HOME="${resolved_cuda_home}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

  if ! command -v nvcc >/dev/null 2>&1; then
    echo "DeepSpeed requires a CUDA toolkit with nvcc, but nvcc was not found." >&2
    echo "Tried CUDA_HOME=${CUDA_HOME:-unset}. Load a CUDA module or set CUDA_HOME before running." >&2
    exit 1
  fi

  selected_cuda_version="${CUDA_HOME##*/cuda-}"
  if [[ -n "${torch_cuda_version}" && "${selected_cuda_version}" != "${torch_cuda_version}" ]]; then
    export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
    echo "Warning: torch was built against CUDA ${torch_cuda_version}, but using toolkit ${selected_cuda_version} at ${CUDA_HOME}." >&2
    echo "Set GRAND_CUDA_VERSION or CUDA_HOME to an exact-match toolkit to avoid DS_SKIP_CUDA_CHECK=1." >&2
  fi
}

configure_cuda_toolkit

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
