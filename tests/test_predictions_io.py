"""Round-trip tests for the frozen-forecast store (src/predictions_io.py)."""
import numpy as np

from src.predictions_io import (
    list_settings,
    load_predictions,
    prediction_path,
    save_predictions,
)


def test_roundtrip_all_arrays(tmp_path):
    y = np.random.default_rng(0).normal(size=(7, 24)).astype(np.float32)
    q = np.random.default_rng(1).normal(size=(7, 24, 5)).astype(np.float32)
    levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    ctx = np.random.default_rng(2).normal(size=(7, 168)).astype(np.float32)
    p = prediction_path(tmp_path, "deepar", "test", "level_shift_4.0")
    save_predictions(p, y_true=y, quantiles=q, levels=levels, context=ctx,
                     meta={"model": "deepar", "alpha": 0.1})
    d = load_predictions(p)
    np.testing.assert_array_equal(d["y_true"], y)
    np.testing.assert_array_equal(d["quantiles"], q)
    np.testing.assert_array_equal(d["levels"], levels)
    np.testing.assert_array_equal(d["context"], ctx)
    assert d["meta"]["model"] == "deepar"


def test_point_only_roundtrip(tmp_path):
    y = np.zeros((3, 24), dtype=np.float32)
    pt = np.ones((3, 24), dtype=np.float32)
    p = prediction_path(tmp_path, "lstm", "test", "clean")
    save_predictions(p, y_true=y, point=pt, meta={})
    d = load_predictions(p)
    assert "quantiles" not in d
    np.testing.assert_array_equal(d["point"], pt)


def test_quantiles_require_levels(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        save_predictions(tmp_path / "x.npz", y_true=np.zeros((2, 4)),
                         quantiles=np.zeros((2, 4, 3)))


def test_list_settings_clean_first(tmp_path):
    y = np.zeros((2, 4), dtype=np.float32)
    for s in ("level_shift_4.0", "clean", "fgsm_1.0"):
        save_predictions(prediction_path(tmp_path, "m", "test", s), y_true=y, point=y)
    assert list_settings(tmp_path, "m", "test") == ["clean", "fgsm_1.0", "level_shift_4.0"]
