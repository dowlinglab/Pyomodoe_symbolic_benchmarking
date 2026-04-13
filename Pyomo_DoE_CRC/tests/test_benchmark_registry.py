from benchmarking.registry import EXAMPLE_SPECS


def test_example_registry_contains_expected_examples():
    assert {
        "two_param_sin",
        "4_state_reactor",
        "4st_6pmt",
        "PDE_diffusion",
    }.issubset(EXAMPLE_SPECS)


def test_registered_examples_build_labeled_models():
    for example_name, example_spec in sorted(EXAMPLE_SPECS.items()):
        experiment = example_spec.build_experiment()
        model = experiment.get_labeled_model()
        assert model is not None
        assert hasattr(model, "experiment_outputs")
        assert hasattr(model, "measurement_error")
        assert hasattr(model, "experiment_inputs")
        assert hasattr(model, "unknown_parameters")


def test_registered_examples_expose_instances_and_initial_points():
    for example_spec in EXAMPLE_SPECS.values():
        assert "default" in example_spec.instances
        assert "nominal" in example_spec.initial_points
