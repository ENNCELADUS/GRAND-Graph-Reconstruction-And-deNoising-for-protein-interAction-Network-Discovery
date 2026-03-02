# RELIC — Refined Embeddings via Latent Interaction Constraints for Protein–Protein Interactions

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#requirements--dependencies)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-0A1E2B?style=flat-square)](#testing)

RELIC predicts protein–protein interactions (PPI) by combining ESM-3 embeddings with purpose-built neural architectures and rigorous evaluation pipelines.

## Quick Install

Recommended to use **Conda** for environment management:

```bash
# 1. Clone the repository
git clone https://github.com/ENNCELADUS/RELIC-Refined-Embeddings-via-Latent-Interaction-Constraints.git ./relic
cd relic

# 2. Create environment
conda create -n relic python=3.11
conda activate relic
pip install uv

# 3. Install dependencies
uv pip install -r requirements.txt
```

## Quick Usage

RELIC uses a config-driven pipeline. Ensure your environment is active:

```bash
conda activate relic

# Run configured stages from YAML (default: train -> evaluate)
python -m src.run --config configs/v3.yaml

# To run only selected stages, edit `run_config.stages` in YAML, e.g.:
# stages: ["train"] or stages: ["evaluate"].
# If stages includes evaluate and domain_adaptation.enabled=true, SHOT runs before evaluation.
# Checkpoints are stored in models/v3/<stage>/<RUN_ID>/.
```

## Documentation

TBD.

## Testing

TBD.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
