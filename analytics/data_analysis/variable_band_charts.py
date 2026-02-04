import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from simulator import band_from_value

def main() -> None:
    output_dir = Path("analytics") / "data_analysis" / "artifacts" / "06_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path("outputs") / "simulator" / "simulated_dataset.csv"
    metadata_path = Path("outputs") / "simulator" / "simulated_metadata.json"

    if not dataset_path.exists():
        raise FileNotFoundError("Missing outputs/simulator/simulated_dataset.csv. Run main.py first.")
    if not metadata_path.exists():
        raise FileNotFoundError("Missing outputs/simulator/simulated_metadata.json. Run main.py first.")

    df = pd.read_csv(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    variables_cfg = json.loads((Path("input_parameters") / "variables.json").read_text(encoding="utf-8"))
    meta_items = metadata["variables"] if isinstance(metadata, dict) else metadata
    var_by_name = {item["name"]: item for item in variables_cfg["dataset_spec"]["variables"]}

    for item in meta_items:
        name = item["name"]
        bands = item["bands"]

        var_cfg = var_by_name.get(name, {})
        if var_cfg.get("type") == "numeric":
            band_series = df[name].apply(lambda v: band_from_value(v, var_cfg))
        else:
            band_series = df[name]

        counts = band_series.value_counts().reindex(bands).fillna(0).astype(int)
        dist_pct = (counts / len(df)) * 100.0
        bad_rate = df.groupby(band_series)["default"].mean().reindex(bands)

        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax2 = ax1.twinx()

        ax1.bar(bands, dist_pct, color="#4C78A8", alpha=0.85, label="Distribution %")
        ax2.plot(bands, bad_rate, color="#F58518", marker="o", label="Bad rate")

        ax1.set_ylabel("Distribution (%)")
        ax2.set_ylabel("Bad rate")
        ax1.set_title(f"{name}: distribution and bad rate by band")
        ax1.tick_params(axis="x", rotation=30)

        # Combined legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
