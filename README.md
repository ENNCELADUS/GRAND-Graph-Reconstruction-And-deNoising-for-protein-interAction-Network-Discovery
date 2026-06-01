<div align="center">
  <h1>GRAND</h1>
  <h2>Graph Reconstruction And deNoising for protein interAction Network Discovery</h2>

  <p>
    <a href="https://github.com/ENNCELADUS/GRAND-Graph-Reconstruction-And-deNoising-for-protein-interAction-Network-Discovery/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/ENNCELADUS/GRAND-Graph-Reconstruction-And-deNoising-for-protein-interAction-Network-Discovery"/></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"/></a>
    <a href="https://www.python.org/downloads/release/python-3110/"><img alt="Python" src="https://img.shields.io/badge/python-3.11-blue.svg"/></a>
    <a href="pyproject.toml"><img alt="Package manager" src="https://img.shields.io/badge/env-uv-0A1E2B"/></a>
  </p>

  <p>
    <a href="#why-grand">Why GRAND?</a> |
    <a href="#quick-start">Quick Start</a> |
    <a href="#example-workflows">Example Workflows</a> |
    <a href="#installation">Installation</a> |
    <a href="#architecture">Architecture</a> |
    <a href="#documentation">Documentation</a>
  </p>
</div>

GRAND is a config-driven research codebase for reconstructing protein-protein
interaction (PPI) graphs from intrinsic protein features. It uses ESM-3
embeddings, neural PPI models, topology-aware training objectives, and
graph-level PRING evaluation to study the gap between accurate pairwise edge
classification and biologically realistic interactome reconstruction.

## Latest News

- **[2026/06]** Archived the TCCIG `p2_fixed` run. Graph-density prior
  initialization and teacher disabling reduced the edge budget but did not fix
  saturated probabilities or graph reconstruction, so decoder calibration is the
  next priority.
- **[2026/05]** Added the TCCIG feature-only graph-generator training stage with
  graph assembly evaluation and topology monitoring.
- **[2026/05]** Updated the canonical TCCIG HPC launcher at
  `scripts/tccig.sh` for multi-GPU SLURM runs through `uv`.
- **[2026/05]** Added implementation-grounded experiment notes under
  `docs/experiment/` and pipeline design references under `docs/design_patterns/`.

## Why GRAND?

Pairwise PPI prediction asks whether two proteins interact. GRAND targets the
larger reconstruction problem: whether a model can assemble many local edge
scores into a sparse, modular, and biologically credible interactome.

- **Graph-first evaluation** - Reconstructs full predicted graphs and reports
  PRING-style topology metrics, not only pairwise AUROC/AUPRC.
- **Strict inductive boundary** - Uses protein-intrinsic features at inference
  time and treats target graph topology as evaluation signal rather than input.
- **Config-driven experiments** - Stores model choice, stage order, data paths,
  optimization settings, and artifact run IDs in YAML.
- **Multiple model families** - Supports `v3`, `v3.1`, `v4`, `v5`, `tuna`, and
  `tccig` through a shared model factory.
- **HPC-oriented execution** - Provides SLURM launchers for standard training,
  ablations, TUnA, and TCCIG runs.

## Quick Start

```bash
# 1. Clone and enter the repository
git clone https://github.com/ENNCELADUS/GRAND-Graph-Reconstruction-And-deNoising-for-protein-interAction-Network-Discovery.git grand
cd grand

# 2. Create or refresh the project environment
uv sync --group dev

# 3. Run the canonical TCCIG pipeline
uv run python -m src.run --config configs/tccig/01.yaml
```

> **Prerequisites**: Python 3.11, `uv`, PRING data under `data/PRING/`, and cached
> ESM-3 embeddings under `data/embeddings/esm3_1024/`.
>
> **Need a lighter check?** Run `uv run python -m pytest tests/unit` before
> launching GPU training.

## Example Workflows

### TCCIG Graph Generator

TCCIG trains a feature-only graph generator from cached protein embeddings,
scores the all-test candidate universe, and assembles a graph with the learned
edge budget.

```bash
uv run python -m src.run --config configs/tccig/01.yaml
```

Expected artifacts:

- `models/tccig/tccig_train/<run_id>/best_model.pth`
- `logs/tccig/tccig_train/<run_id>/tccig_train_step.csv`
- `logs/tccig/evaluate/<run_id>/evaluate.csv`
- `logs/tccig/topology_evaluate/<run_id>/topology_metrics.csv`

### Topology-Aware V3 Baseline

The V3 topology pipeline starts from the configured model family, applies
topology fine-tuning, evaluates pairwise metrics, and then reconstructs the
graph for topology metrics.

```bash
uv run python -m src.run --config configs/v3.yaml
```

### HPC Launch

Use the shell launchers in `scripts/` for cluster runs. They create log/model
directories, dispatch optimized configs through `src.optimize.run`, and use
`torch.distributed.run` for multi-GPU execution.

```bash
# TCCIG default config
sbatch scripts/tccig.sh

# TCCIG with an explicit config or config directory
sbatch scripts/tccig.sh configs/tccig/01.yaml
```

## Installation

This project uses `uv` and `pyproject.toml` as the environment source of truth.

```bash
# Install dependencies, including development tools
uv sync --group dev

# Run Python inside the managed environment
uv run python -m src.run --config configs/tccig/01.yaml

# Run tests and lint checks
uv run python -m pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
```

The environment pins the main training stack, including `torch`, `accelerate`,
`deepspeed`, `esm`, `networkx`, `optuna`, `numpy`, `scikit-learn`, and `scipy`.

### Data Layout

The checked-in configs expect the PRING benchmark and cached ESM-3 embeddings in
the following locations:

```text
data/
  PRING/
    species_processed_data/
      human/
        BFS/
          human_train_ppi.txt
          human_val_ppi.txt
          human_test_ppi.txt
          all_test_ppi.txt
  embeddings/
    esm3_1024/
      embeddings/
```

For dataset details, see `data/PRING/README.md`.

## Architecture

GRAND is organized as a single-runtime, stage-based pipeline. A YAML file
selects the stages, model family, data paths, device strategy, training settings,
and evaluation settings.

```text
YAML config
    |
    v
src.pipeline.__main__
    |
    v
PipelineConfig + PipelineRuntime
    |
    +--> build_dataloaders()
    +--> build_model()
    |
    v
Ordered stages from run_config.stages
    |
    +--> train
    +--> topology_finetune
    +--> tccig_train
    +--> adapt
    +--> evaluate
    +--> topology_evaluate
    |
    v
logs/<model>/<stage>/<run_id>/
models/<model>/<stage>/<run_id>/
```

Supported stage selections include:

```yaml
run_config:
  # Choose one stage list:
  stages: ["train", "evaluate"]
  # stages: ["train", "topology_finetune", "evaluate", "topology_evaluate"]
  # stages: ["tccig_train", "evaluate", "topology_evaluate"]
  # stages: ["evaluate"]  # requires run_config.load_checkpoint_path
```

`tccig_train` is mutually exclusive with `train` and `topology_finetune` in the
current stage contract. If SHOT domain adaptation is enabled, the pipeline
inserts `adapt` before `evaluate`.

### Key Components

- `src/pipeline/` - CLI entrypoint, typed config view, runtime, orchestration,
  and stage wrappers.
- `src/model/` - Neural model families registered by `model_config.model`.
- `src/train/` - General training utilities plus TCCIG-specific trainer,
  teacher, validation, and data preparation.
- `src/topology/` - Graph supervision, topology losses, topology metrics, and
  graph reconstruction reports.
- `src/evaluate/` - Pairwise and graph-assembly evaluation.
- `src/optimize/` - Optuna/NAS-lite optimization workflow.
- `configs/` - Reproducible experiment configurations.
- `scripts/` - HPC orchestration templates.
- `docs/` - Research framing, experiment notes, and design references.

## Configuration

All user-facing experiment behavior should live in YAML. The most important
sections are:

- `run_config` - Stage order, run IDs, checkpoint loading, and seeds.
- `device_config` - CPU/GPU selection, DDP, mixed precision, and backend.
- `data_config` - PRING split paths, embedding cache paths, and dataloader
  settings.
- `model_config` - Model family and model hyperparameters.
- `training_config` - Epochs, batch size, optimizer, scheduler, loss, sampling,
  logging, and domain adaptation.
- `topology_finetune` or `tccig_train` - Graph-specific training objectives and
  validation settings.
- `evaluate` and `topology_evaluate` - Metric lists, graph assembly mode, and
  topology report settings.

## Documentation

- `docs/introduction/research_problem.md` - Scientific motivation and the
  pair-to-graph gap.
- `docs/design_patterns/pipeline.md` - Pipeline runtime, stages, launchers, and
  artifact contracts.
- `docs/design_patterns/model.md` - Model construction and registration.
- `docs/design_patterns/trainer.md` - Trainer responsibilities and strategies.
- `docs/design_patterns/evaluator.md` - Evaluation semantics and metrics.
- `docs/design_patterns/logging.md` - Log, metric, checkpoint, and topology
  artifact paths.
- `docs/design_patterns/hpo.md` - Optimization workflow and artifact contracts.
- `docs/experiment/tccig/` - TCCIG model and experiment notes.
- `docs/experiment/v3/` and `docs/experiment/v3-1/` - Baseline and ablation
  notes.

## Testing

```bash
# Full default test suite
uv run python -m pytest

# Focused examples
uv run python -m pytest tests/unit/test_model_tccig.py
uv run python -m pytest tests/integration/test_tccig_train_stage.py
uv run python -m pytest tests/integration/test_run_pipeline_modes.py

# Static checks
uv run ruff check .
uv run mypy src
```

## Contributing

Contributions should stay config-driven and scoped to the relevant pipeline
stage or model family. Before opening a pull request:

1. Add focused tests for changed behavior.
2. Run the relevant `uv run python -m pytest ...` command.
3. Run `uv run ruff check .`.
4. Document new configs, artifact contracts, or stage behavior in `docs/`.

Use Conventional Commits such as `feat: add topology metric` or
`fix: stabilize tccig validation`.

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for
details.

## Acknowledgments

GRAND builds on PRING-style graph reconstruction evaluation, protein language
model embeddings, and topology-aware graph learning ideas. The codebase is
maintained as a research system for reproducible PPI graph reconstruction
experiments.
