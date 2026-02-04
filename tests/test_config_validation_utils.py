import json
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from input_parameters.edit_variables import _as_float, _validate_all, _validate_global_params
from io_utils import load_spec, validate_spec


def test_load_spec_raises_for_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_spec(str(tmp_path / "missing.json"))


def test_validate_spec_accepts_valid_structure():
    spec = {
        "dataset_spec": {
            "variables": [
                {
                    "name": "x",
                    "bands": [
                        {"band": "a", "distribution_pct": 60, "bad_rate_ratio": 1.0},
                        {"band": "b", "distribution_pct": 40, "bad_rate_ratio": 2.0},
                    ],
                }
            ]
        }
    }
    variables = validate_spec(spec)
    assert len(variables) == 1
    assert variables[0]["name"] == "x"


def test_validate_spec_rejects_bad_distribution():
    spec = {
        "dataset_spec": {
            "variables": [
                {
                    "name": "x",
                    "bands": [
                        {"band": "a", "distribution_pct": 50, "bad_rate_ratio": 1.0},
                        {"band": "b", "distribution_pct": 30, "bad_rate_ratio": 2.0},
                    ],
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="sum to 100"):
        validate_spec(spec)


def test_edit_validation_detects_duplicate_names_and_invalid_global():
    cfg = {
        "dataset_spec": {
            "variables": [
                {
                    "name": "dup",
                    "bands": [
                        {"band": "a", "distribution_pct": 50.0, "bad_rate_ratio": 1.0},
                        {"band": "b", "distribution_pct": 50.0, "bad_rate_ratio": 1.0},
                    ],
                },
                {
                    "name": "dup",
                    "bands": [
                        {"band": "a", "distribution_pct": 70.0, "bad_rate_ratio": 1.0},
                        {"band": "b", "distribution_pct": 30.0, "bad_rate_ratio": 1.0},
                    ],
                },
            ]
        }
    }
    errs = _validate_all(cfg)
    assert any("Duplicate variable names" in e for e in errs)

    global_cfg = {
        "simulation_population": -1,
        "global_bad_rate_pct": 120,
        "train_set_pct": 60,
        "test_set_pct": 30,
    }
    g_errs = _validate_global_params(global_cfg)
    assert any("simulation_population must be > 0" in e for e in g_errs)
    assert any("between 0 and 100" in e for e in g_errs)
    assert any("must sum to 100" in e for e in g_errs)


def test_as_float_rejects_invalid_value():
    with pytest.raises(ValueError):
        _as_float("not-a-number", "distribution_pct")
