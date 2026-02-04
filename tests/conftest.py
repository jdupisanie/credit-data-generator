import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(f"testmod_{uuid4().hex}", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mini_variables_config() -> dict:
    return {
        "dataset_spec": {
            "horizon_months": 12,
            "variables": [
                {
                    "name": "age_years",
                    "type": "numeric",
                    "expected_range": {"min": 18, "max": 80},
                    "bands": [
                        {"band": "<30", "distribution_pct": 40.0, "bad_rate_ratio": 2.0},
                        {"band": "30-50", "distribution_pct": 40.0, "bad_rate_ratio": 1.2},
                        {"band": ">50", "distribution_pct": 20.0, "bad_rate_ratio": 1.0},
                    ],
                },
                {
                    "name": "residential_status",
                    "type": "categorical",
                    "bands": [
                        {"band": "owner", "distribution_pct": 50.0, "bad_rate_ratio": 1.0},
                        {"band": "rent", "distribution_pct": 50.0, "bad_rate_ratio": 1.8},
                    ],
                },
            ],
        }
    }


@pytest.fixture
def mini_global_config() -> dict:
    return {
        "global_bad_rate_pct": 10.0,
        "simulation_population": 200,
        "final_population": 200,
        "train_set_pct": 80.0,
        "test_set_pct": 20.0,
    }


@pytest.fixture
def mini_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 120
    ages = rng.integers(18, 80, size=n)
    status = rng.choice(["owner", "rent"], size=n, p=[0.55, 0.45])
    logits = -3.0 + 0.05 * (ages < 30) + 0.8 * (status == "rent")
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(int)
    # Ensure both classes exist in tiny samples.
    if y.sum() == 0:
        y[0] = 1
    if y.sum() == len(y):
        y[0] = 0
    return pd.DataFrame(
        {
            "age_years": ages,
            "residential_status": status,
            "default": y,
        }
    )


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def write_json():
    def _write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return _write
