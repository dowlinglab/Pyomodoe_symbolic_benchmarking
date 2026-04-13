import pytest

from benchmarking.runner import run_case


@pytest.mark.parametrize(
    ("objective_key", "derivative_mode"),
    [
        ("d_opt_cholesky", "finite_difference"),
        ("d_opt_greybox", "symbolic"),
    ],
)
def test_reactor_runner_smoke(objective_key, derivative_mode, tmp_path):
    artifact = run_case(
        example_name="4_state_reactor",
        objective_key=objective_key,
        derivative_mode=derivative_mode,
        instance_name="smoke",
        initial_point_name="nominal",
        out_json=str(tmp_path / "artifact.json"),
        run_dir=str(tmp_path / "run"),
    )
    assert artifact["solver_status"] is not None
    assert artifact["termination_condition"] is not None
