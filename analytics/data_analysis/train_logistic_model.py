import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model on the training dataset."
    )
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
        "--woe-train",
        type=Path,
        default=Path("analytics")
        / "data_analysis"
        / "artifacts"
        / "03_preprocessing"
        / "simulated_dataset_train_woe.csv",
        help="Path to WOE-encoded training CSV (optional).",
    )
    parser.add_argument(
        "--woe-test",
        type=Path,
        default=Path("analytics")
        / "data_analysis"
        / "artifacts"
        / "03_preprocessing"
        / "simulated_dataset_test_woe.csv",
        help="Path to WOE-encoded test CSV (optional).",
    )
    parser.add_argument(
        "--use-woe",
        action="store_true",
        help="Use WOE-encoded datasets when available.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="default",
        help="Target column name.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        help="Comma-separated feature list. Default: all non-target columns.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="logit_all_features",
        help="Model name for tracking comparisons.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analytics") / "data_analysis" / "artifacts" / "04_models",
        help="Base directory to store model artifacts.",
    )
    parser.add_argument(
        "--submodel",
        type=str,
        default="logistic_regression",
        help="Subfolder name under output-dir to store this model's artifacts.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("analytics")
        / "data_analysis"
        / "artifacts"
        / "05_model_comparison"
        / "model_registry.csv",
        help="CSV file for model comparison results.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def _build_preprocessor(X: pd.DataFrame, woe_mode: bool) -> ColumnTransformer:
    if woe_mode:
        return ColumnTransformer(
            transformers=[("num", StandardScaler(with_mean=False), X.columns.tolist())]
        )

    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ]
    )


def _select_features(df: pd.DataFrame, target: str, features: list[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c != target]
    if features:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Features not found in dataset: {missing}")
        cols = [c for c in cols if c in features]
    return df[cols]


def _evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, probs)),
        "accuracy": float(accuracy_score(y, preds)),
        "log_loss": float(log_loss(y, probs)),
    }


def main() -> None:
    args = parse_args()

    train_path = args.woe_train if args.use_woe and args.woe_train.exists() else args.train
    test_path = args.woe_test if args.use_woe and args.woe_test.exists() else args.test

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}")

    train_df = pd.read_csv(train_path)
    if args.target not in train_df.columns:
        raise ValueError(f"Target column '{args.target}' not found in train dataset.")

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    X_train = _select_features(train_df, args.target, features)
    y_train = train_df[args.target].astype(int)

    woe_mode = train_path == args.woe_train
    preprocessor = _build_preprocessor(X_train, woe_mode)
    model = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.0,
        solver="saga",
        max_iter=4000,
        random_state=args.random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    train_metrics = _evaluate(pipeline, X_train, y_train)

    test_metrics = {"auc": np.nan, "accuracy": np.nan, "log_loss": np.nan}
    test_rows = 0
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        if args.target in test_df.columns:
            X_test = _select_features(test_df, args.target, features)
            y_test = test_df[args.target].astype(int)
            test_metrics = _evaluate(pipeline, X_test, y_test)
            test_rows = len(test_df)

    model_dir = args.output_dir / args.submodel
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.model_name}.joblib"
    metadata_path = model_dir / f"{args.model_name}.json"

    joblib.dump(pipeline, model_path)
    metadata = {
        "model_name": args.model_name,
        "model_type": "logistic_regression",
        "features": X_train.columns.tolist(),
        "woe_mode": bool(woe_mode),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": len(train_df),
        "test_rows": test_rows,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    result_row = {
        "model_name": args.model_name,
        "model_type": "logistic_regression",
        "features": "|".join(X_train.columns.tolist()),
        "train_rows": len(train_df),
        "test_rows": test_rows,
        "train_auc": train_metrics["auc"],
        "train_accuracy": train_metrics["accuracy"],
        "train_log_loss": train_metrics["log_loss"],
        "test_auc": test_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_log_loss": test_metrics["log_loss"],
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "model_dir": str(model_dir),
    }

    if args.results_csv.exists():
        results_df = pd.read_csv(args.results_csv)
        results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        results_df = pd.DataFrame([result_row])

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results_csv, index=False)


if __name__ == "__main__":
    main()
