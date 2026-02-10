import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split simulated dataset into train/test sets.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analytics") / "data_analysis" / "artifacts" / "01_datasets" / "simulated_dataset_total.csv",
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analytics") / "data_analysis" / "artifacts" / "01_datasets",
        help="Directory to write train/test CSVs.",
    )
    parser.add_argument(
        "--global-params",
        type=Path,
        default=Path("input_parameters") / "global_parameters.json",
        help="Path to global_parameters.json.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Optional override for test fraction (0-1). If omitted, uses global parameters.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input dataset: {args.input}")
    if not args.global_params.exists():
        raise FileNotFoundError(f"Missing global parameters file: {args.global_params}")

    df = pd.read_csv(args.input)
    global_cfg = json.loads(args.global_params.read_text(encoding="utf-8"))

    if args.test_size is not None:
        test_size = float(args.test_size)
    else:
        train_pct = float(global_cfg.get("train_set_pct", 80.0))
        test_pct = float(global_cfg.get("test_set_pct", 20.0))
        if train_pct <= 0 or test_pct <= 0:
            raise ValueError("train_set_pct and test_set_pct must be > 0.")
        if abs((train_pct + test_pct) - 100.0) > 1e-6:
            raise ValueError("train_set_pct and test_set_pct must sum to 100.")
        test_size = test_pct / 100.0

    stratify_col = df["default"] if "default" in df.columns else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=args.random_state,
        stratify=stratify_col,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "simulated_dataset_train.csv"
    test_path = args.output_dir / "simulated_dataset_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


if __name__ == "__main__":
    main()
