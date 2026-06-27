# Repository Guidelines

## Quick Context
- **What**: This project predicts protein–protein interactions using ESM-3 + downstream neural network classifiers.
- **Goal**: Reproducible, config-driven pipelines for Pretraining, Finetuning, and Evaluation.
- **Role**: Act as a careful junior engineer. Follow **Plan → Confirm → Code**.

## Environment Requirement (Critical)
- Use `uv` + `pyproject.toml` as the single source of truth for the Python environment.
- Bootstrap or refresh the local environment with: `uv sync --group dev`
- Run Python, tests, lint, and scripts through `uv run`, for example `uv run python -m pytest`, `uv run ruff check .`, and `uv run mypy src`.
- If `.venv/` is missing or stale, run `uv sync --group dev` before executing repository Python commands.

## Code Style
Act as a careful junior engineer with strong tooling.
- **Core**: Write clean, efficient Python 3.10+. Prefer composition. Be concise.
- **Structure**: Target 200-400 line files (max 600). Keep functions <50 lines. Organize by feature.
- **Naming**: `snake_case` (files/funcs), `PascalCase` (classes), `UPPER_SNAKE` (constants).
- **Quality**:
  - No `print` statements (use logging).
  - No hardcoded values (use config).
  - Max nesting level: 4.
  - Strict type hints (avoid `Any`).
- **Best Practices**:
  - Absolute imports only (`from src.x import y`).
  - Google-style docstrings.
  - Handle specific exceptions (no bare `except`).

## Project Structure & Module Organization
Standard Deep Learning project layout:
- `src/model/`: Neural network architectures (`nn.Module` subclasses).
- `src/train/`: Training logic, including `Trainer` classes and strategies.
- `src/evaluate/`: Evaluation scripts and logic.
- `src/utils/`: Shared utilities (logging, config parsing, device management).
- `configs/`: YAML configuration files for experiments.
- `scripts/`: Shell scripts for orchestration (e.g., `v3.sh`).
- `tests/`: Project tests (Unit, Integration, E2E).

## Build, Test, and Development Commands
- **Environment bootstrap**: `uv sync --group dev`
- **Linting**: `uv run ruff check --fix .` (Fixes lint errors)
- **Formatting**: `uv run ruff format .` (Formats code)
- **Testing**: `uv run python -m pytest` (Runs all tests)
- **Orchestration**: Use shell scripts in `scripts/` to run pipelines on hpc. Avoid running `python src/run.py` directly.

## Testing Guidelines
- **Framework**: `pytest`.
- **Coverage**: Aim for ≥80% coverage on touched modules.
- **Workflow**: Test-Driven Development (TDD) — Write failing tests -> Implement -> Refactor.
- **Types**:
  - **Unit**: Fast, isolated tests for functions/classes.
  - **Integration**: Database/API interactions.
  - **E2E**: Critical user flows.

## Commit & Pull Request Guidelines
- **Commits**: Conventional Commits format `<type>: <description>`
  - Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `ci`.
- **Branches**: `short-description` (e.g., `new-scheduler`, `fix-nans`). Rebase on `main` before merging.
- **Pull Requests**:
  - Summarize changes and context.
  - Include a test plan and verification results.
  - Address all critical review feedback.

## Security & Best Practices
- **Secrets**: Never commit API keys or `.env` files. Use environment variables.
- **Validation**: Sanitize all user inputs.
- **Dependencies**: Periodically review and update dependencies.

## HPC Slurm Rules
- Submit experiment scripts through Slurm on `wangar2023@10.15.89.192`.
- For CPU-only jobs, use:
  ```bash
  #SBATCH -p critical
  #SBATCH -A hexm-critical
  ```
- For A40 GPU jobs, use:
  ```bash
  #SBATCH -p hexm
  #SBATCH -A hexm
  #SBATCH --gres=gpu:NVIDIAA40:4
  ```
- If the A40 queue is long, use the L40 queue instead:
  ```bash
  #SBATCH -p hexm_l40
  #SBATCH -A hexm
  #SBATCH --gres=gpu:NVIDIAL40:4
  ```