import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split simulated dataset into train/test sets.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "simulator" / "simulated_dataset.csv",
        help="Path to the input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analytics") / "data_analysis" / "artifacts" / "01_datasets",
        help="Directory to write train/test CSVs.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows to use for the test set.",
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

    df = pd.read_csv(args.input)

    stratify_col = df["default"] if "default" in df.columns else None
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
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
