"""Unit tests for distributed optimization coordination."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import src.optimize.distributed as optimize_distributed_module
from src.optimize.search_space import SearchParameter


@dataclass
class _RecordingChannel:
    commands: list[object]
    barrier_calls: int = 0

    def receive(self) -> object:
        return self.commands.pop(0)

    def barrier(self) -> None:
        self.barrier_calls += 1


def test_worker_loop_executes_trial_and_best_pipeline_commands() -> None:
    from src.optimize.distributed import OptimizationCommand, run_distributed_worker_loop

    observed_trials: list[tuple[int, dict[str, object]]] = []
    observed_best: list[dict[str, object]] = []

    def fake_execute_trial(**kwargs: object) -> object:
        observed_trials.append(
            (
                int(kwargs["trial_number"]),
                dict(kwargs["sampled_values"]),
            )
        )
        assert kwargs["run_id_prefix"] == "20260320_110811"
        return None

    def fake_run_best_full_pipeline(**kwargs: object) -> str:
        observed_best.append(dict(kwargs["best_values"]))
        assert kwargs["run_id_prefix"] == "20260320_110811"
        return "20260320_110811_best_train"

    channel = _RecordingChannel(
        commands=[
            OptimizationCommand(
                kind="run_trial",
                trial_number=2,
                sampled_values={"scheduler_max_lr": 1.0e-4},
            ),
            OptimizationCommand(
                kind="run_best_pipeline",
                best_values={"scheduler_max_lr": 8.0e-5},
            ),
            OptimizationCommand(kind="stop"),
        ]
    )

    run_distributed_worker_loop(
        base_config={"run_config": {}, "device_config": {}, "model_config": {"model": "v3"}},
        search_space=[
            SearchParameter(
                name="scheduler_max_lr",
                path="training_config.scheduler.max_lr",
                parameter_type="float",
                low=1.0e-5,
                high=2.0e-4,
            )
        ],
        study_name="v3_hpo",
        run_id_prefix="20260320_110811",
        objective_metric="val_auprc",
        direction="maximize",
        execution_cfg={"trial_stages": ["train"], "ddp_per_trial": True},
        run_pipeline_fn=lambda cfg: None,
        channel=channel,
        execute_trial_fn=fake_execute_trial,
        run_best_full_pipeline_fn=fake_run_best_full_pipeline,
    )

    assert observed_trials == [(2, {"scheduler_max_lr": 1.0e-4})]
    assert observed_best == [{"scheduler_max_lr": 8.0e-5}]
    assert channel.barrier_calls == 2


def test_torch_distributed_channel_send_receive_and_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.optimize.distributed import (
        OptimizationCommand,
        TorchDistributedOptimizationChannel,
    )
    from src.pipeline.runtime import DistributedContext

    context = DistributedContext(
        ddp_enabled=True,
        is_distributed=True,
        rank=0,
        local_rank=1,
        world_size=4,
    )
    channel = TorchDistributedOptimizationChannel(context)
    observed: dict[str, object] = {}

    def fake_broadcast_object_list(payload: list[object], src: int) -> None:
        observed["src"] = src
        command = payload[0]
        if command is None:
            payload[0] = OptimizationCommand(kind="stop")
        else:
            observed["sent_kind"] = command.kind

    monkeypatch.setattr(
        optimize_distributed_module.dist,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )
    monkeypatch.setattr(
        optimize_distributed_module,
        "distributed_barrier",
        lambda distributed_context: observed.setdefault("barrier_rank", distributed_context.rank),
    )

    channel.send(OptimizationCommand(kind="run_trial", trial_number=0))
    received = channel.receive()
    channel.barrier()

    assert observed["src"] == 0
    assert observed["sent_kind"] == "run_trial"
    assert received.kind == "stop"
    assert observed["barrier_rank"] == 0


def test_build_optimization_channel_returns_none_for_single_process() -> None:
    from src.optimize.distributed import build_optimization_channel
    from src.pipeline.runtime import DistributedContext

    context = DistributedContext(ddp_enabled=False, is_distributed=False)

    assert build_optimization_channel(context) is None


def test_run_initialize_optimization_distributed_requires_ddp_per_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.optimize.run import _initialize_optimization_distributed

    monkeypatch.setenv("WORLD_SIZE", "4")

    with pytest.raises(ValueError, match="ddp_per_trial=true"):
        _initialize_optimization_distributed(execution_cfg={"ddp_per_trial": False})
