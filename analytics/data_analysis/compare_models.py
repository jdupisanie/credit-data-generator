import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare performance across trained models (logistic, Cox, and score-based models)."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("analytics") / "data_analysis" / "artifacts" / "04_models",
        help="Root directory that contains model subfolders and metadata JSON files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Which dataset split to evaluate, using paths from each model metadata.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="default",
        help="Target/event column used for binary comparison metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analytics") / "data_analysis" / "artifacts" / "05_model_comparison",
        help="Directory to save comparison outputs.",
    )
    parser.add_argument(
        "--threshold-mode",
        type=str,
        choices=["youden", "fixed"],
        default="youden",
        help="Threshold strategy for confusion matrix from scores.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help="Used when --threshold-mode=fixed.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_artifact(path: Path) -> Any:
    if path.suffix.lower() == ".joblib":
        if joblib is None:
            raise ImportError("joblib is required to load .joblib model artifacts.")
        return joblib.load(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _infer_model_path(meta_path: Path, meta: dict[str, Any]) -> Path:
    model_name = str(meta.get("model_name", meta_path.stem))
    candidates = [
        meta_path.parent / f"{model_name}.joblib",
        meta_path.parent / f"{model_name}.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No model artifact found for metadata: {meta_path}")


def _dataset_path_from_meta(meta: dict[str, Any], split: str) -> Path:
    key = "test_path" if split == "test" else "train_path"
    if key not in meta:
        raise ValueError(f"Metadata missing '{key}'.")
    return Path(meta[key])


def _select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in evaluation dataset: {missing[:5]}")
    return df[features]


def _scores_from_logistic(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("Unsupported logistic artifact: no predict_proba/decision_function.")


def _scores_from_cox(model_bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    preprocessor = model_bundle["preprocessor"]
    beta = np.asarray(model_bundle["beta"], dtype=float)
    X_enc = preprocessor.transform(X)
    X_enc = X_enc.toarray() if hasattr(X_enc, "toarray") else np.asarray(X_enc)
    return X_enc @ beta


def _best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])


def _evaluate_binary(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, roc_thr = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    return {
        "auc": float(auc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thr.tolist(),
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(args.models_dir.glob("*/*.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No model metadata found under: {args.models_dir}")

    comparison_rows = []
    roc_payload = {}
    cm_payload = {}

    for meta_path in meta_paths:
        meta = _load_json(meta_path)
        if "model_name" not in meta:
            continue

        model_name = str(meta["model_name"])
        model_type = str(meta.get("model_type", "logistic_regression"))
        features = list(meta.get("features", []))

        eval_path = _dataset_path_from_meta(meta, args.split)
        if not eval_path.exists():
            continue
        df = pd.read_csv(eval_path)
        if args.target not in df.columns:
            continue

        y_true = df[args.target].astype(int).to_numpy()
        X_df = _select_features(df, features)

        model_path = _infer_model_path(meta_path, meta)
        artifact = _load_model_artifact(model_path)

        if model_type == "cox_ph":
            scores = _scores_from_cox(artifact, X_df)
        else:
            scores = _scores_from_logistic(artifact, X_df)

        if args.threshold_mode == "youden":
            threshold = _best_threshold(y_true, scores)
        else:
            threshold = float(args.fixed_threshold)

        metrics = _evaluate_binary(y_true, scores, threshold)
        cm = metrics.pop("confusion_matrix")
        roc_curve_data = metrics.pop("roc_curve")

        comparison_rows.append(
            {
                "model_name": model_name,
                "model_type": model_type,
                "split": args.split,
                "rows": len(df),
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "threshold": metrics["threshold"],
                "tn": cm["tn"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "tp": cm["tp"],
                "model_path": str(model_path),
                "metadata_path": str(meta_path),
                "eval_dataset": str(eval_path),
            }
        )
        roc_payload[model_name] = roc_curve_data
        cm_payload[model_name] = cm

    if not comparison_rows:
        raise ValueError("No models could be evaluated. Check metadata paths and datasets.")

    result_df = pd.DataFrame(comparison_rows).sort_values("auc", ascending=False)
    result_csv = args.output_dir / f"comparison_{args.split}.csv"
    result_df.to_csv(result_csv, index=False)

    (args.output_dir / f"roc_curves_{args.split}.json").write_text(
        json.dumps(roc_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / f"confusion_matrices_{args.split}.json").write_text(
        json.dumps(cm_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 6))
    for row in comparison_rows:
        name = row["model_name"]
        roc_info = roc_payload[name]
        plt.plot(roc_info["fpr"], roc_info["tpr"], label=f"{name} (AUC={row['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Comparison ({args.split})")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(args.output_dir / f"roc_comparison_{args.split}.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
