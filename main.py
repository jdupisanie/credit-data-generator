import json
from pathlib import Path
import sys

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from simulator import (
    _simulate_single_variable,
    simulate_final_default_indicators,
    generate_numeric_values_from_bands,
)


def main() -> None:
    variables_path = Path("input_parameters") / "variables.json"
    globals_path = Path("input_parameters") / "global_parameters.json"

    variables = json.loads(variables_path.read_text(encoding="utf-8"))
    globals_cfg = json.loads(globals_path.read_text(encoding="utf-8"))

    n = int(globals_cfg["simulation_population"])
    global_bad_rate = float(globals_cfg["global_bad_rate_pct"]) / 100.0

    rng = np.random.default_rng(42)

    output_dir = Path("analytics") / "data_analysis" / "artifacts" / "01_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_model = {}
    data_output = {}
    target = None
    computed_meta = []

    all_variables = variables["dataset_spec"]["variables"]
    selected_variables = [
        var for var in all_variables if bool(var.get("include_in_data_creation", True))
    ]

    for idx, var in enumerate(selected_variables):
        computed, X, Y = _simulate_single_variable(var, n, global_bad_rate, rng)
        data_model[computed.name] = X
        if var.get("type") == "numeric":
            data_output[computed.name] = generate_numeric_values_from_bands(X, var, rng)
        else:
            data_output[computed.name] = X

        if idx == 0:
            target = Y

        computed_meta.append(
            {
                "name": computed.name,
                "bands": computed.bands,
                "p": computed.p.tolist(),
                "gamma": computed.gamma.tolist(),
                "delta": computed.delta.tolist(),
            }
        )

    if target is None:
        raise ValueError(
            "No enabled variables found to simulate. "
            "Set include_in_data_creation=true for at least one variable."
        )

    df_X = pd.DataFrame(data_model)

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # Older scikit-learn versions use `sparse`
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
    X_enc = encoder.fit_transform(df_X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_enc, target)

    final_result = simulate_final_default_indicators(
        df_X=df_X,
        encoder=encoder,
        model=model,
        rng=rng,
        enforce_global_bad_rate=global_bad_rate,
    )

    df = pd.DataFrame(data_output)
    df["default"] = final_result.y_final

    data_path = output_dir / "simulated_dataset_total.csv"
    df.to_csv(data_path, index=False)

    meta_path = output_dir / "simulated_metadata.json"
    meta_payload = {
        "variables": computed_meta,
        "final_default": {
            "realized_bad_rate": final_result.realized_bad_rate,
            "enforced_bad_rate": global_bad_rate,
        },
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
