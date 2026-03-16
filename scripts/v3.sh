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

cd /public/home/wangar2023/relic/
source ~/.bashrc
conda activate relic
CONFIG_PATH="${1:-configs/v3.yaml}"

# Automatically detect number of GPUs from SLURM allocation
NGPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NGPUS GPUs"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

torchrun --standalone --nproc_per_node="$NGPUS" --module src.run --config "$CONFIG_PATH"
