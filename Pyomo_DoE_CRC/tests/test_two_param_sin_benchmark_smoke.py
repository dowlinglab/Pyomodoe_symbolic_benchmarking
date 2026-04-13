from __future__ import annotations

from pathlib import Path

import pytest

from benchmarking.registry import DERIVATIVE_MODES, OBJECTIVE_SPECS
from benchmarking.runner import run_case


MODE_MATRIX = [
    (objective_key, derivative_mode)
    for objective_key in OBJECTIVE_SPECS
    for derivative_mode in DERIVATIVE_MODES
]


@pytest.mark.parametrize(
    ("objective_key", "derivative_mode"),
    MODE_MATRIX,
    ids=[f"{objective_key}-{derivative_mode}" for objective_key, derivative_mode in MODE_MATRIX],
)
def test_two_param_sin_mode_runs(objective_key: str, derivative_mode: str, tmp_path: Path):
    out_json = tmp_path / f"{objective_key}_{derivative_mode}.json"
    run_dir = tmp_path / f"{objective_key}_{derivative_mode}"

    artifact = run_case(
        example_name="two_param_sin",
        objective_key=objective_key,
        derivative_mode=derivative_mode,
        instance_name="default",
        initial_point_name="nominal",
        run_id="smoke_suite",
        out_json=str(out_json),
        run_dir=str(run_dir),
    )

    assert artifact["example_name"] == "two_param_sin"
    assert artifact["instance_name"] == "default"
    assert artifact["initial_point_name"] == "nominal"
    assert artifact["run_id"] == "smoke_suite"
    assert artifact["objective_key"] == objective_key
    assert artifact["derivative_mode"] == derivative_mode
    assert artifact["solver_status"] is not None
    assert artifact["termination_condition"] is not None
    assert artifact["git_hash"]
    assert out_json.exists()


def test_two_param_sin_nondefault_initial_point_runs(tmp_path: Path):
    artifact = run_case(
        example_name="two_param_sin",
        objective_key="d_opt_cholesky",
        derivative_mode="finite_difference",
        instance_name="default",
        initial_point_name="shifted",
        run_id="shifted_case",
        out_json=str(tmp_path / "artifact.json"),
        run_dir=str(tmp_path / "run"),
    )

    assert artifact["instance_name"] == "default"
    assert artifact["initial_point_name"] == "shifted"
    assert artifact["run_id"] == "shifted_case"
    assert artifact["solver_status"] is not None


def test_two_param_sin_nondefault_instance_runs(tmp_path: Path):
    artifact = run_case(
        example_name="two_param_sin",
        objective_key="d_opt_cholesky",
        derivative_mode="finite_difference",
        instance_name="smoke",
        initial_point_name="nominal",
        run_id="coarse_case",
        out_json=str(tmp_path / "artifact.json"),
        run_dir=str(tmp_path / "run"),
    )

    assert artifact["instance_name"] == "smoke"
    assert artifact["initial_point_name"] == "nominal"
    assert artifact["run_id"] == "coarse_case"
    assert artifact["solver_status"] is not None
