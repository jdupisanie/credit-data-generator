import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from simulator import band_from_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WOE encode features for logistic regression.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("analytics")
        / "data_analysis"
        / "artifacts"
        / "01_datasets"
        / "simulated_dataset_train.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("analytics")
        / "data_analysis"
        / "artifacts"
        / "01_datasets"
        / "simulated_dataset_test.csv",
        help="Path to test CSV (optional).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="default",
        help="Target column name.",
    )
    parser.add_argument(
        "--variables-config",
        type=Path,
        default=Path("input_parameters") / "variables.json",
        help="Variables configuration used for banding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analytics") / "data_analysis" / "artifacts" / "03_preprocessing",
        help="Directory to write WOE datasets and mappings.",
    )
    return parser.parse_args()


def _band_series(series: pd.Series, cfg: dict | None) -> pd.Series:
    if cfg and cfg.get("type") == "numeric":
        return series.apply(lambda v: band_from_value(v, cfg))
    return series


def _compute_woe_table(
    df: pd.DataFrame,
    target: str,
    variables_cfg: dict,
) -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    var_by_name = {item["name"]: item for item in variables_cfg["dataset_spec"]["variables"]}
    y = df[target].astype(int)
    good = (y == 0).sum()
    bad = (y == 1).sum()
    good = max(good, 1)
    bad = max(bad, 1)

    woe_tables: dict[str, dict[str, float]] = {}
    woe_details: dict[str, dict] = {}

    for name in [c for c in df.columns if c != target]:
        cfg = var_by_name.get(name)
        banded = _band_series(df[name], cfg)

        bands = []
        if cfg:
            bands = [b["band"] for b in cfg["bands"]]
        else:
            bands = sorted(banded.dropna().unique().tolist())

        woe_by_band = {}
        band_details = []
        for band in bands:
            mask = banded == band
            total_count = int(mask.sum())
            bad_count = int(y[mask].sum())
            good_count = int(total_count - bad_count)

            bad_rate = (bad_count + 0.5) / (bad + 1.0)
            good_rate = (good_count + 0.5) / (good + 1.0)
            dist_total = total_count / len(df) if len(df) else 0.0
            dist_bad = bad_count / bad if bad else 0.0
            dist_good = good_count / good if good else 0.0
            woe_value = float(np.log(bad_rate / good_rate))

            woe_by_band[band] = woe_value
            band_details.append(
                {
                    "band": band,
                    "count_total": total_count,
                    "count_bad": bad_count,
                    "count_good": good_count,
                    "dist_total": float(dist_total),
                    "dist_bad": float(dist_bad),
                    "dist_good": float(dist_good),
                    "woe": woe_value,
                }
            )

        woe_tables[name] = woe_by_band
        woe_details[name] = {
            "total_rows": int(len(df)),
            "total_bad": int(bad),
            "total_good": int(good),
            "bands": band_details,
        }

    return woe_tables, woe_details


def _apply_woe(
    df: pd.DataFrame,
    target: str,
    variables_cfg: dict,
    woe_tables: dict[str, dict[str, float]],
) -> pd.DataFrame:
    var_by_name = {item["name"]: item for item in variables_cfg["dataset_spec"]["variables"]}

    woe_df = pd.DataFrame()
    for name in [c for c in df.columns if c != target]:
        cfg = var_by_name.get(name)
        banded = _band_series(df[name], cfg)
        mapping = woe_tables.get(name, {})
        woe_df[name] = banded.map(mapping).astype(float)

    woe_df[target] = df[target].astype(int)
    return woe_df


def main() -> None:
    args = parse_args()

    if not args.train.exists():
        raise FileNotFoundError(f"Missing train dataset: {args.train}")
    if not args.variables_config.exists():
        raise FileNotFoundError(f"Missing variables config: {args.variables_config}")

    variables_cfg = json.loads(args.variables_config.read_text(encoding="utf-8"))
    train_df = pd.read_csv(args.train)
    if args.target not in train_df.columns:
        raise ValueError(f"Target column '{args.target}' not found in train dataset.")

    woe_tables, woe_details = _compute_woe_table(train_df, args.target, variables_cfg)
    train_woe = _apply_woe(train_df, args.target, variables_cfg, woe_tables)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "simulated_dataset_train_woe.csv"
    train_woe.to_csv(train_out, index=False)

    if args.test.exists():
        test_df = pd.read_csv(args.test)
        if args.target in test_df.columns:
            test_woe = _apply_woe(test_df, args.target, variables_cfg, woe_tables)
            test_out = args.output_dir / "simulated_dataset_test_woe.csv"
            test_woe.to_csv(test_out, index=False)

    mapping_out = args.output_dir / "woe_mappings.json"
    mapping_payload = {
        "source": {
            "train_dataset": str(args.train),
            "target": args.target,
        },
        "variables": woe_details,
    }
    mapping_out.write_text(json.dumps(mapping_payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
