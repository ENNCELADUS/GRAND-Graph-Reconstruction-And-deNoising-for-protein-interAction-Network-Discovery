# GRAND — Graph Reconstruction And deNoising for protein interAction Network Discovery

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](#requirements--dependencies)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-0A1E2B?style=flat-square)](#testing)

GRAND reconstructs protein interaction networks from protein features using graph structure learning (GSL) with topology-aware training objectives. Unlike traditional pairwise PPI predictors, GRAND optimizes directly on graph-level metrics (topology + function) to produce biologically realistic interactomes.

## Quick Install

Recommended to use **uv** for environment management:

```bash
# 1. Clone the repository
git clone https://github.com/ENNCELADUS/GRAND-Graph-Reconstruction-And-deNoising-for-protein-interAction-Network-Discovery.git ./grand
cd grand

# 2. Install dependencies into .venv
uv sync --group dev
```

## Quick Usage

GRAND uses a config-driven pipeline. Run commands through `uv run` so the project-local `.venv` is used:

```bash
# Run configured stages from YAML (default: train -> evaluate)
uv run python -m src.run --config configs/v3/v3.yaml

# To run only selected stages, edit `run_config.stages` in YAML, e.g.:
# stages: ["train"] or stages: ["evaluate"].
# If stages includes evaluate and domain_adaptation.enabled=true, SHOT runs before evaluation.
# Checkpoints are stored in models/v3/<stage>/<RUN_ID>/.
```

## Documentation

TBD.

## Testing

```bash
uv run python -m pytest
uv run ruff check .
uv run ruff format .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
