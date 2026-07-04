"""CLI for evaluating the frozen pairwise scorer as a topology baseline."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from accelerate.utils import set_seed

from tccig.prepare import GraphRule, load_pring_tables, strict_reject_legacy_hooks, write_json
from tccig.test import run_raw_pairwise_topology_baseline
from tccig.train import (
    _build_runtime,
    _cache_root,
    _configure_logging,
    _load_yaml_config,
    _log_root,
    _mapping_section,
    _required_path,
    _run_id,
    _sampling_seed,
    _score_split,
)


def run_baseline(
    config: Mapping[str, object],
    *,
    source_run_id: str | None = None,
    output_run_id: str = "pairwise_baseline",
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate raw v3.1 pairwise scores on the PRING topology-test protocol."""
    strict_reject_legacy_hooks(config)
    runtime = _build_runtime(config=config, build_accelerator_fn=None)
    _configure_logging(runtime)
    set_seed(_sampling_seed(config))

    cache_run_id = source_run_id or _run_id(config)
    processed_dir = _required_path(_mapping_section(config, "data"), "processed_dir")
    tables = load_pring_tables(processed_dir)
    scorer_cfg = _mapping_section(config, "pairwise_scorer")
    cache_dir = _cache_root(config) / "score_cache" / cache_run_id
    log_dir = _log_root(config) / "tccig" / output_run_id
    artifact_dir = log_dir / "raw_pairwise_topology_baseline"
    output_rule = GraphRule(type="threshold", value=threshold)

    metrics = run_raw_pairwise_topology_baseline(
        table=tables["topology_test"],
        processed_dir=processed_dir,
        scorer_cfg=scorer_cfg,
        runtime=runtime,
        cache_dir=cache_dir,
        log_dir=log_dir,
        output_dir=artifact_dir,
        raw_output_rule=output_rule,
        score_split_fn=_score_split,
    )
    if runtime.is_main_process:
        write_json(
            artifact_dir / "manifest.json",
            {
                "run_id": output_run_id,
                "source_run_id": cache_run_id,
                "score_cache_dir": str(cache_dir),
                "artifact_dir": str(artifact_dir),
                "raw_output_rule": output_rule.to_dict(),
                "self_pair_rows_dropped": {"topology_test": tables["topology_test"].self_pair_rows},
            },
        )
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for ``python -m tccig.raw_pairwise_topology_baseline``."""
    parser = argparse.ArgumentParser(description="Run raw pairwise topology baseline")
    parser.add_argument("--config", required=True, help="Path to a TCCIG YAML config")
    parser.add_argument(
        "--source-run-id",
        default=None,
        help="Run id whose score cache should be reused; defaults to config run.run_id",
    )
    parser.add_argument(
        "--output-run-id",
        default="pairwise_baseline",
        help="Run id under logs/tccig for baseline artifacts",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Raw pairwise probability threshold used to assemble the predicted graph",
    )
    args = parser.parse_args(argv)
    run_baseline(
        _load_yaml_config(Path(args.config)),
        source_run_id=args.source_run_id,
        output_run_id=args.output_run_id,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
