import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Cox proportional hazards model on the training dataset."
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
        help="Binary event column used when --event-col is not provided.",
    )
    parser.add_argument(
        "--event-col",
        type=str,
        default="",
        help="Event indicator column (1=event, 0=censored).",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="",
        help="Duration/time-to-event column. If missing, synthetic duration is created.",
    )
    parser.add_argument(
        "--horizon-months",
        type=float,
        default=12.0,
        help="Max horizon for synthetic duration when --time-col is missing.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="",
        help="Comma-separated feature list. Default: all non target/event/time columns.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cox_all_features",
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
        default="cox_hazard",
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
        "--max-iter",
        type=int,
        default=500,
        help="Maximum optimizer iterations.",
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


def _select_features(
    df: pd.DataFrame,
    exclude_cols: list[str],
    features: list[str],
) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in exclude_cols]
    if features:
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Features not found in dataset: {missing}")
        cols = [c for c in cols if c in features]
    return df[cols]


def _prepare_survival_targets(
    df: pd.DataFrame,
    target_col: str,
    event_col: str,
    time_col: str,
    horizon_months: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if event_col:
        if event_col not in df.columns:
            raise ValueError(f"Event column '{event_col}' not found.")
        event = df[event_col].astype(int).to_numpy()
        event_source = event_col
    else:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")
        event = df[target_col].astype(int).to_numpy()
        event_source = target_col

    if time_col:
        if time_col not in df.columns:
            raise ValueError(f"Time column '{time_col}' not found.")
        duration = df[time_col].astype(float).to_numpy()
        time_source = time_col
    else:
        # Synthetic time-to-event: defaults happen before horizon, non-defaults censored at horizon.
        rng = np.random.default_rng(random_state)
        duration = np.full(len(df), float(horizon_months), dtype=float)
        event_idx = np.where(event == 1)[0]
        if len(event_idx) > 0:
            duration[event_idx] = np.maximum(
                1e-6,
                rng.uniform(1e-6, float(horizon_months), size=len(event_idx)),
            )
        time_source = "synthetic_from_default"

    duration = np.clip(duration.astype(float), 1e-6, None)
    event = np.where(event > 0, 1, 0).astype(int)
    return duration, event, f"event={event_source},time={time_source}"


def _cox_partial_loglik_and_grad(
    beta: np.ndarray,
    X: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
) -> tuple[float, np.ndarray]:
    order = np.argsort(-duration)
    Xs = X[order]
    ts = duration[order]
    es = event[order]

    eta = Xs @ beta
    eta = np.clip(eta, -50, 50)
    r = np.exp(eta)

    cum_r = np.cumsum(r)
    cum_rx = np.cumsum(r[:, None] * Xs, axis=0)

    ll = 0.0
    grad = np.zeros_like(beta)

    i = 0
    n = len(ts)
    while i < n:
        t_val = ts[i]
        j = i
        while j + 1 < n and ts[j + 1] == t_val:
            j += 1

        event_slice = es[i : j + 1]
        d = int(event_slice.sum())
        if d > 0:
            event_rows = Xs[i : j + 1][event_slice == 1]
            sum_x_event = event_rows.sum(axis=0)
            denom = cum_r[j]
            ll += float(sum_x_event @ beta) - d * np.log(max(denom, 1e-12))
            grad += sum_x_event - d * (cum_rx[j] / max(denom, 1e-12))

        i = j + 1

    return ll, grad


def _fit_cox_ph(
    X: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, float, bool, int]:
    p = X.shape[1]
    beta0 = np.zeros(p, dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        ll, grad = _cox_partial_loglik_and_grad(beta, X, duration, event)
        return -ll, -grad

    result = minimize(
        fun=lambda b: objective(b)[0],
        x0=beta0,
        jac=lambda b: objective(b)[1],
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    beta_hat = result.x
    ll_final, _ = _cox_partial_loglik_and_grad(beta_hat, X, duration, event)
    return beta_hat, float(ll_final), bool(result.success), int(result.nit)


def _concordance_index(
    duration: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
) -> float:
    concordant = 0.0
    comparable = 0.0

    for i in range(len(duration)):
        if event[i] != 1:
            continue
        mask = duration > duration[i]
        n_comp = int(mask.sum())
        if n_comp == 0:
            continue
        comparable += n_comp
        other_scores = risk_score[mask]
        concordant += float((risk_score[i] > other_scores).sum())
        concordant += 0.5 * float((risk_score[i] == other_scores).sum())

    return float(concordant / comparable) if comparable > 0 else float("nan")


def _evaluate_cox(
    beta: np.ndarray,
    X: np.ndarray,
    duration: np.ndarray,
    event: np.ndarray,
) -> dict[str, float]:
    ll, _ = _cox_partial_loglik_and_grad(beta, X, duration, event)
    risk = X @ beta
    c_idx = _concordance_index(duration, event, risk)
    return {
        "partial_log_likelihood": float(ll),
        "c_index": float(c_idx),
    }


def main() -> None:
    args = parse_args()

    train_path = args.woe_train if args.use_woe and args.woe_train.exists() else args.train
    test_path = args.woe_test if args.use_woe and args.woe_test.exists() else args.test
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train dataset: {train_path}")

    train_df = pd.read_csv(train_path)
    duration_train, event_train, survival_source = _prepare_survival_targets(
        train_df,
        target_col=args.target,
        event_col=args.event_col,
        time_col=args.time_col,
        horizon_months=args.horizon_months,
        random_state=args.random_state,
    )

    exclude_cols = [args.target]
    if args.event_col:
        exclude_cols.append(args.event_col)
    if args.time_col:
        exclude_cols.append(args.time_col)

    feature_list = [f.strip() for f in args.features.split(",") if f.strip()]
    X_train_df = _select_features(train_df, exclude_cols=exclude_cols, features=feature_list)

    woe_mode = train_path == args.woe_train
    preprocessor = _build_preprocessor(X_train_df, woe_mode=woe_mode)
    X_train = preprocessor.fit_transform(X_train_df)
    X_train = X_train.toarray() if hasattr(X_train, "toarray") else np.asarray(X_train)

    beta, ll_train, converged, n_iter = _fit_cox_ph(
        X_train,
        duration_train,
        event_train,
        max_iter=args.max_iter,
    )
    train_metrics = _evaluate_cox(beta, X_train, duration_train, event_train)

    test_metrics = {"partial_log_likelihood": np.nan, "c_index": np.nan}
    test_rows = 0
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        if args.target in test_df.columns or args.event_col:
            duration_test, event_test, _ = _prepare_survival_targets(
                test_df,
                target_col=args.target,
                event_col=args.event_col,
                time_col=args.time_col,
                horizon_months=args.horizon_months,
                random_state=args.random_state,
            )
            X_test_df = _select_features(test_df, exclude_cols=exclude_cols, features=feature_list)
            X_test = preprocessor.transform(X_test_df)
            X_test = X_test.toarray() if hasattr(X_test, "toarray") else np.asarray(X_test)
            test_metrics = _evaluate_cox(beta, X_test, duration_test, event_test)
            test_rows = len(test_df)

    model_dir = args.output_dir / args.submodel
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{args.model_name}.pkl"
    metadata_path = model_dir / f"{args.model_name}.json"
    coef_path = model_dir / f"{args.model_name}_coefficients.csv"

    feature_names = preprocessor.get_feature_names_out().tolist()
    model_bundle = {
        "model_type": "cox_ph",
        "beta": beta,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "survival_source": survival_source,
    }
    with model_path.open("wb") as handle:
        pickle.dump(model_bundle, handle)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": beta,
            "hazard_ratio": np.exp(beta),
        }
    ).sort_values("hazard_ratio", ascending=False)
    coef_df.to_csv(coef_path, index=False)

    metadata = {
        "model_name": args.model_name,
        "model_type": "cox_ph",
        "features": X_train_df.columns.tolist(),
        "feature_count_encoded": len(feature_names),
        "train_rows": len(train_df),
        "test_rows": test_rows,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "woe_mode": bool(woe_mode),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "survival_source": survival_source,
        "optimizer": {
            "converged": converged,
            "iterations": n_iter,
            "max_iter": args.max_iter,
            "train_partial_log_likelihood": ll_train,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    result_row = {
        "model_name": args.model_name,
        "model_type": "cox_ph",
        "features": "|".join(X_train_df.columns.tolist()),
        "train_rows": len(train_df),
        "test_rows": test_rows,
        "train_auc": np.nan,
        "train_accuracy": np.nan,
        "train_log_loss": np.nan,
        "test_auc": np.nan,
        "test_accuracy": np.nan,
        "test_log_loss": np.nan,
        "train_c_index": train_metrics["c_index"],
        "test_c_index": test_metrics["c_index"],
        "train_partial_log_likelihood": train_metrics["partial_log_likelihood"],
        "test_partial_log_likelihood": test_metrics["partial_log_likelihood"],
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
