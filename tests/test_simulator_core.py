import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulator import (
    _compute_deltas,
    _flip_to_target,
    _simulate_single_variable,
    band_from_value,
    generate_numeric_values_from_bands,
    simulate_final_default_indicators,
)


def test_compute_deltas_preserves_global_bad_rate():
    p = np.array([0.4, 0.4, 0.2], dtype=float)
    gamma = np.array([2.0, 1.2, 1.0], dtype=float)
    d = 0.1

    delta = _compute_deltas(p, gamma, d)

    assert np.isclose(np.sum(p * delta), d, atol=1e-6)
    assert np.all(delta > 0)
    assert np.all(delta < 1)


def test_flip_to_target_hits_exact_count():
    rng = np.random.default_rng(42)
    y = np.array([1, 1, 0, 0, 0, 1, 0, 0], dtype=int)
    out = _flip_to_target(y, target_ones=4, rng=rng)
    assert int(out.sum()) == 4


def test_generate_numeric_values_and_band_mapping_integer():
    rng = np.random.default_rng(11)
    var = {
        "name": "tenure",
        "type": "numeric",
        "expected_range": {"min": 0, "max": 60},
        "bands": [
            {"band": "<6", "distribution_pct": 25.0, "bad_rate_ratio": 2.0},
            {"band": "6-24", "distribution_pct": 50.0, "bad_rate_ratio": 1.4},
            {"band": ">24", "distribution_pct": 25.0, "bad_rate_ratio": 1.0},
        ],
    }
    labels = np.array(["<6", "6-24", ">24", "6-24", "<6", ">24"], dtype=object)

    values = generate_numeric_values_from_bands(labels, var, rng)

    assert np.issubdtype(values.dtype, np.integer)
    mapped = [band_from_value(v, var) for v in values]
    assert mapped == labels.tolist()


def test_band_from_value_supports_never_sentinel():
    var = {
        "name": "months_since_last_delinquency",
        "type": "numeric",
        "expected_range": {"min": 0, "max": 240},
        "bands": [
            {"band": "never", "distribution_pct": 50.0, "bad_rate_ratio": 1.0},
            {"band": "<6", "distribution_pct": 20.0, "bad_rate_ratio": 4.0},
            {"band": "6-12", "distribution_pct": 15.0, "bad_rate_ratio": 3.0},
            {"band": "12-24", "distribution_pct": 10.0, "bad_rate_ratio": 2.0},
            {"band": ">24", "distribution_pct": 5.0, "bad_rate_ratio": 1.5},
        ],
    }
    assert band_from_value(240, var) == "never"
    assert band_from_value(5, var) == "<6"


def test_simulate_single_variable_returns_expected_shapes():
    var = {
        "name": "status",
        "bands": [
            {"band": "owner", "distribution_pct": 60.0, "bad_rate_ratio": 1.0},
            {"band": "rent", "distribution_pct": 40.0, "bad_rate_ratio": 2.0},
        ],
    }
    rng = np.random.default_rng(3)
    computed, X, Y = _simulate_single_variable(var, n=200, global_bad_rate=0.1, rng=rng)
    assert computed.name == "status"
    assert len(X) == 200 and len(Y) == 200
    assert set(np.unique(X)) == {"owner", "rent"}
    # Sorted descending by Y in implementation.
    assert Y[: int(Y.sum())].sum() == int(Y.sum())


def test_simulate_final_default_indicators_enforces_rate():
    df_X = pd.DataFrame(
        {
            "x1": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "x2": ["u", "u", "v", "v", "u", "v", "u", "v"],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_enc = encoder.fit_transform(df_X)
    model = LogisticRegression(max_iter=500).fit(X_enc, y)
    rng = np.random.default_rng(9)

    result = simulate_final_default_indicators(
        df_X=df_X,
        encoder=encoder,
        model=model,
        rng=rng,
        enforce_global_bad_rate=0.25,
    )
    assert int(result.y_final.sum()) == 2
    assert np.isclose(result.realized_bad_rate, 0.25)


def test_generate_numeric_values_requires_expected_range():
    rng = np.random.default_rng(0)
    var = {
        "name": "bad_numeric",
        "type": "numeric",
        "bands": [
            {"band": "0-1", "distribution_pct": 100.0, "bad_rate_ratio": 1.0},
        ],
    }
    with pytest.raises(ValueError, match="Missing expected_range"):
        generate_numeric_values_from_bands(np.array(["0-1"], dtype=object), var, rng)
