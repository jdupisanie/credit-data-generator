import json
from pathlib import Path


def test_distribution_pct_sums_to_100():
    variables_path = Path("input_parameters") / "variables.json"
    data = json.loads(variables_path.read_text(encoding="utf-8"))

    variables = data["dataset_spec"]["variables"]
    errors = []

    for variable in variables:
        bands = variable.get("bands", [])
        total = sum(band.get("distribution_pct", 0.0) for band in bands)
        if abs(total - 100.0) > 1e-6:
            errors.append(f"{variable.get('name', '<unknown>')}: {total}")

    assert not errors, "distribution_pct totals not 100: " + "; ".join(errors)
