import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

def load_spec(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spec file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def validate_spec(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expected structure (flexible):
      spec["dataset_spec"]["variables"] -> list of variables
      each variable: {"name": ..., "bands": [{"band": ..., "distribution_pct": ..., "bad_rate_ratio": ...}, ...]}
    Returns list of variable dicts.
    """
    if "dataset_spec" in spec and "variables" in spec["dataset_spec"]:
        variables = spec["dataset_spec"]["variables"]
    elif "variables" in spec:
        variables = spec["variables"]
    else:
        raise ValueError("Spec must contain either spec['dataset_spec']['variables'] or spec['variables'].")

    if not isinstance(variables, list) or len(variables) == 0:
        raise ValueError("Spec variables must be a non-empty list.")

    for v in variables:
        if "name" not in v:
            raise ValueError("Each variable must have a 'name'.")
        if "bands" not in v or not isinstance(v["bands"], list) or len(v["bands"]) < 2:
            raise ValueError(f"Variable '{v.get('name')}' must have a non-empty 'bands' list (>=2 bands).")

        s = 0.0
        for b in v["bands"]:
            if "band" not in b:
                raise ValueError(f"Variable '{v['name']}' has a band without 'band' label.")
            if "distribution_pct" not in b:
                raise ValueError(f"Variable '{v['name']}' band '{b.get('band')}' missing 'distribution_pct'.")
            if "bad_rate_ratio" not in b:
                raise ValueError(f"Variable '{v['name']}' band '{b.get('band')}' missing 'bad_rate_ratio'.")
            dp = float(b["distribution_pct"])
            br = float(b["bad_rate_ratio"])
            if dp <= 0:
                raise ValueError(f"Variable '{v['name']}' band '{b['band']}' has non-positive distribution_pct.")
            if br <= 0:
                raise ValueError(f"Variable '{v['name']}' band '{b['band']}' has non-positive bad_rate_ratio.")
            s += dp

        # Allow minor rounding error
        if abs(s - 100.0) > 1e-6:
            raise ValueError(f"Variable '{v['name']}' distributions must sum to 100. Found {s:.6f}.")

    return variables
