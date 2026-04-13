import numpy as np
import torch

from ai_models.sensitivity import SensitivityTarget
from ai_models_aurora.model import Aurora0p25Pretrained


def _model_stub():
    model = Aurora0p25Pretrained.__new__(Aurora0p25Pretrained)
    model.levels = tuple(Aurora0p25Pretrained.levels)
    model.surf_vars = tuple(Aurora0p25Pretrained.surf_vars)
    model.atmos_vars = tuple(Aurora0p25Pretrained.atmos_vars)
    model.ordering = list(Aurora0p25Pretrained.ordering)
    model.level_to_index = {int(level): index for index, level in enumerate(model.levels)}
    model.sensitivity_metric = "mean-square"
    return model


def test_parse_model_args_accepts_sensitivity_flags():
    model = _model_stub()
    args = model.parse_model_args(
        [
            "--sensitivity",
            "--model-checkpointing",
            "--no-rollout-checkpointing",
            "--sensitivity-metric",
            "mean",
        ]
    )

    assert args.sensitivity is True
    assert args.model_checkpointing is True
    assert args.rollout_checkpointing is False
    assert args.sensitivity_metric == "mean"


def test_parse_model_args_accepts_integrated_gradients_options():
    model = _model_stub()
    args = model.parse_model_args(
        [
            "--attribution-method",
            "integrated-gradients",
            "--ig-steps",
            "12",
            "--ig-baseline",
            "climatology",
        ]
    )

    assert args.attribution_method == "integrated-gradients"
    assert args.ig_steps == 12
    assert args.ig_baseline == "climatology"


def test_parse_target_field_surface_and_pressure():
    model = _model_stub()

    assert model.parse_target_field(target_param="2t", target_level=None) == "2t"
    assert model.parse_target_field(target_param="q", target_level=850) == "q850"


def test_parse_target_field_rejects_invalid_level():
    model = _model_stub()

    try:
        model.parse_target_field(target_param="q", target_level=775)
    except ValueError as exc:
        assert "Unsupported level" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid pressure level")


def test_channel_to_variable_level_mapping():
    model = _model_stub()

    assert model.channel_to_variable_level("2t") == ("surf", "2t", None)
    assert model.channel_to_variable_level("u500") == ("atmos", "u", 500)


def test_target_weights_wrap_longitude_across_dateline():
    weights = Aurora0p25Pretrained._target_weights(
        latitudes=np.array([10.0, 0.0, -10.0], dtype=np.float32),
        longitudes=np.array([350.0, 0.0, 10.0, 20.0], dtype=np.float32),
        area=(20.0, 355.0, -20.0, 5.0),
        device="cpu",
        dtype=torch.float32,
    )

    weights = weights.numpy()
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights[:, 2:] == 0.0)
    assert np.any(weights[:, :2] > 0.0)


def test_default_target_dataclass_roundtrip():
    target = SensitivityTarget(
        name="west-coast-q700",
        field="q700",
        area=(50.0, 230.0, 30.0, 245.0),
        metric="mean-square",
    )

    assert target.name == "west-coast-q700"
    assert target.field == "q700"
    assert target.metric == "mean-square"


def test_zero_baseline_batch_is_zero_for_inputs():
    model = _model_stub()
    model.__dict__["device"] = "cpu"

    batch = type("BatchLike", (), {})()
    batch.surf_vars = {"2t": torch.randn(1, 2, 4, 4), "10u": torch.randn(1, 2, 4, 4)}
    batch.atmos_vars = {"q": torch.randn(1, 2, len(model.levels), 4, 4)}
    batch.static_vars = {}
    batch.metadata = None

    baseline = model._zero_baseline_batch(batch)

    for tensor in baseline.surf_vars.values():
        assert torch.allclose(tensor, torch.zeros_like(tensor))
    for tensor in baseline.atmos_vars.values():
        assert torch.allclose(tensor, torch.zeros_like(tensor))


def test_sensitivity_objective_normalizes_surface_field_target():
    model = _model_stub()
    scale = model.surf_scales["2t"]

    prediction = type("PredictionLike", (), {})()
    prediction.surf_vars = {
        "2t": torch.tensor([[[[0.0]], [[scale]]]], dtype=torch.float32),
    }
    prediction.atmos_vars = {
        "q": torch.zeros((1, 2, len(model.levels), 1, 1), dtype=torch.float32),
    }
    prediction.metadata = None

    target = SensitivityTarget(name="two-t", field="2t", area=None, metric="mean")
    objective = model.sensitivity_objective(prediction, target)

    assert torch.isclose(objective, torch.tensor(1.0))


def test_sensitivity_objective_normalizes_pressure_field_target():
    model = _model_stub()
    scale = model.atmos_scales["q"][700]

    q_values = torch.zeros((1, 2, len(model.levels), 1, 1), dtype=torch.float32)
    q_values[0, -1, model.level_to_index[700], 0, 0] = scale

    prediction = type("PredictionLike", (), {})()
    prediction.surf_vars = {
        "2t": torch.zeros((1, 2, 1, 1), dtype=torch.float32),
    }
    prediction.atmos_vars = {"q": q_values}
    prediction.metadata = None

    target = SensitivityTarget(name="q700", field="q700", area=None, metric="mean-square")
    objective = model.sensitivity_objective(prediction, target)

    assert torch.isclose(objective, torch.tensor(1.0))
