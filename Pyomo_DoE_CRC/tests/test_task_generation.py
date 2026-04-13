from pathlib import Path

from benchmarking.tasks import build_tasks, write_tasks_tsv


def test_build_tasks_for_filtered_two_param_sin_grid():
    tasks = build_tasks(
        example_names=["two_param_sin"],
        objective_keys=["d_opt_cholesky", "d_opt_greybox"],
        derivative_modes=["finite_difference"],
        run_reps=2,
    )
    assert len(tasks) == 8
    first = tasks[0]
    assert first.example_name == "two_param_sin"
    assert first.run_rep == 1
    assert "__rep01" in first.run_id
    assert first.out_json.endswith(".json")


def test_write_tasks_tsv_round_trip(tmp_path: Path):
    tasks = build_tasks(
        example_names=["PDE_diffusion"],
        objective_keys=["d_opt_cholesky"],
        derivative_modes=["finite_difference"],
        run_reps=1,
        out_dir=str(tmp_path / "raw"),
    )
    out_path = write_tasks_tsv(tasks, str(tmp_path / "tasks.tsv"))
    text = out_path.read_text(encoding="utf-8")
    assert "example_name\tinstance_name\tinitial_point_name" in text
    assert "PDE_diffusion" in text
    assert "boundary_pulse" in text
