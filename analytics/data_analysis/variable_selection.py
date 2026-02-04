import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import chi2, norm
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from simulator import band_from_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Variable selection on the training dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("analytics")
        / "data_analysis"
        / "artifacts"
        / "01_datasets"
        / "simulated_dataset_train.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="default",
        help="Target column name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analytics")
        / "data_analysis"
        / "artifacts"
        / "02_feature_selection"
        / "variable_selection_report.csv",
        help="Output CSV report path.",
    )
    parser.add_argument(
        "--variables-config",
        type=Path,
        default=Path("input_parameters") / "variables.json",
        help="Variables configuration used for banding and IV.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for CV AUC in the report.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ]
    )


def _fit_logit_and_ll(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> tuple[LogisticRegression, ColumnTransformer, np.ndarray, float]:
    preprocessor = _build_preprocessor(X)
    X_enc = preprocessor.fit_transform(X)

    model = LogisticRegression(
        solver="lbfgs",
        penalty=None,
        max_iter=4000,
        random_state=random_state,
    )

    model.fit(X_enc, y)
    probs = model.predict_proba(X_enc)[:, 1]
    probs = np.clip(probs, 1e-9, 1 - 1e-9)
    ll = float(np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs)))
    return model, preprocessor, X_enc, ll


def _feature_to_variable_name(feature: str, columns: list[str]) -> str:
    raw = feature
    if feature.startswith("cat__"):
        raw = feature[5:]
    elif feature.startswith("num__"):
        raw = feature[5:]

    for col in sorted(columns, key=len, reverse=True):
        if raw == col or raw.startswith(col + "_"):
            return col
    return raw


def _compute_information_value(
    df: pd.DataFrame,
    target: str,
    variables_cfg: dict,
) -> dict[str, float]:
    var_by_name = {item["name"]: item for item in variables_cfg["dataset_spec"]["variables"]}
    y = df[target].astype(int)
    good = (y == 0).sum()
    bad = (y == 1).sum()
    good = max(good, 1)
    bad = max(bad, 1)

    iv_by_var: dict[str, float] = {}

    for name in [c for c in df.columns if c != target]:
        cfg = var_by_name.get(name)
        series = df[name]

        if cfg and cfg.get("type") == "numeric":
            banded = series.apply(lambda v: band_from_value(v, cfg))
            bands = [b["band"] for b in cfg["bands"]]
        elif cfg:
            banded = series
            bands = [b["band"] for b in cfg["bands"]]
        else:
            banded = series
            bands = sorted(series.dropna().unique().tolist())

        iv = 0.0
        for band in bands:
            mask = banded == band
            bad_count = int(y[mask].sum())
            good_count = int(mask.sum() - bad_count)

            bad_rate = (bad_count + 0.5) / (bad + 1.0)
            good_rate = (good_count + 0.5) / (good + 1.0)
            woe = np.log(bad_rate / good_rate)
            iv += (bad_rate - good_rate) * woe

        iv_by_var[name] = float(iv)

    return iv_by_var


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing input dataset: {args.input}")

    df = pd.read_csv(args.input)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset.")

    X = df.drop(columns=[args.target])
    y = df[args.target].astype(int)

    preprocessor = _build_preprocessor(X)
    X_enc = preprocessor.fit_transform(X)

    feature_names = preprocessor.get_feature_names_out()
    feature_to_var = {
        f: _feature_to_variable_name(f, X.columns.tolist()) for f in feature_names
    }

    # 1) Mutual information per original column
    mi_scores = mutual_info_classif(X_enc, y, random_state=args.random_state)
    mi_df = pd.DataFrame({"feature": feature_names, "mi": mi_scores})
    mi_df["variable"] = mi_df["feature"].map(feature_to_var)
    mi_by_var = mi_df.groupby("variable")["mi"].sum().sort_values(ascending=False)

    # 2) L1 logistic regression for sparsity and ranking
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )
    model = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=1.0,
        solver="saga",
        max_iter=4000,
        random_state=args.random_state,
    )
    model.fit(X_enc, y)
    coef = np.abs(model.coef_).ravel()
    l1_df = pd.DataFrame({"feature": feature_names, "coef_abs": coef})
    l1_df["variable"] = l1_df["feature"].map(feature_to_var)
    l1_by_var = l1_df.groupby("variable")["coef_abs"].sum().sort_values(ascending=False)

    # 3) Wald test (full model)
    full_model, full_preprocessor, full_X_enc, ll_full = _fit_logit_and_ll(
        X, y, args.random_state
    )

    probs_full = full_model.predict_proba(full_X_enc)[:, 1]
    probs_full = np.clip(probs_full, 1e-9, 1 - 1e-9)

    if sparse.issparse(full_X_enc):
        X_enc_sparse = full_X_enc.tocsr()
    else:
        X_enc_sparse = sparse.csr_matrix(full_X_enc)

    ones = sparse.csr_matrix(np.ones((X_enc_sparse.shape[0], 1)))
    X_design = sparse.hstack([ones, X_enc_sparse], format="csr")
    weights = probs_full * (1 - probs_full)
    X_weighted = X_design.multiply(weights[:, None])
    fisher = (X_design.T @ X_weighted).toarray()

    try:
        cov = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(fisher)

    se = np.sqrt(np.diag(cov))
    coef_full = np.concatenate(([full_model.intercept_[0]], full_model.coef_.ravel()))
    z_scores = coef_full / se
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))

    feature_names_full = ["intercept"] + list(full_preprocessor.get_feature_names_out())
    feature_to_var_full = {
        f: _feature_to_variable_name(f, X.columns.tolist()) for f in feature_names_full
    }

    wald_by_var = {}
    for feature, z in zip(feature_names_full, z_scores):
        var_name = feature_to_var_full.get(feature)
        if var_name is None or var_name == "intercept":
            continue
        wald_by_var.setdefault(var_name, []).append(z * z)

    wald_stats = {}
    for var_name, stats in wald_by_var.items():
        chi2_stat = float(np.sum(stats))
        df_w = len(stats)
        p_val = float(1 - chi2.cdf(chi2_stat, df_w))
        wald_stats[var_name] = (chi2_stat, df_w, p_val)

    # 4) Likelihood Ratio test per variable
    lr_stats = {}
    for var_name in X.columns:
        X_reduced = X.drop(columns=[var_name])
        if X_reduced.shape[1] == 0:
            continue
        _, _, _, ll_reduced = _fit_logit_and_ll(X_reduced, y, args.random_state)
        df_diff = len(
            [f for f in feature_names_full if feature_to_var_full.get(f) == var_name]
        )
        lr_chi2 = 2.0 * (ll_full - ll_reduced)
        lr_p = float(1 - chi2.cdf(lr_chi2, df_diff))
        lr_stats[var_name] = (float(lr_chi2), df_diff, lr_p)

    # 5) Cross-validated AUC using full set (reference)
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        X_train_enc = preprocessor.fit_transform(X_train)
        X_test_enc = preprocessor.transform(X_test)

        model_cv = LogisticRegression(
            penalty="elasticnet",
            l1_ratio=0.0,
            solver="saga",
            max_iter=4000,
            random_state=args.random_state,
        )
        model_cv.fit(X_train_enc, y_train)
        probs = model_cv.predict_proba(X_test_enc)[:, 1]
        aucs.append(roc_auc_score(y_test, probs))

    # 6) Information Value
    if not args.variables_config.exists():
        raise FileNotFoundError(f"Missing variables config: {args.variables_config}")
    variables_cfg = json.loads(args.variables_config.read_text(encoding="utf-8"))
    iv_by_var = _compute_information_value(df, args.target, variables_cfg)

    report = pd.DataFrame(
        {
            "variable": sorted(X.columns),
            "mutual_info": [mi_by_var.get(v, 0.0) for v in sorted(X.columns)],
            "l1_coef_abs": [l1_by_var.get(v, 0.0) for v in sorted(X.columns)],
            "wald_chi2": [wald_stats.get(v, (0.0, 0, 1.0))[0] for v in sorted(X.columns)],
            "wald_df": [wald_stats.get(v, (0.0, 0, 1.0))[1] for v in sorted(X.columns)],
            "wald_pvalue": [wald_stats.get(v, (0.0, 0, 1.0))[2] for v in sorted(X.columns)],
            "lr_chi2": [lr_stats.get(v, (0.0, 0, 1.0))[0] for v in sorted(X.columns)],
            "lr_df": [lr_stats.get(v, (0.0, 0, 1.0))[1] for v in sorted(X.columns)],
            "lr_pvalue": [lr_stats.get(v, (0.0, 0, 1.0))[2] for v in sorted(X.columns)],
            "information_value": [iv_by_var.get(v, 0.0) for v in sorted(X.columns)],
        }
    )
    report["rank_mi"] = report["mutual_info"].rank(ascending=False, method="min").astype(int)
    report["rank_l1"] = report["l1_coef_abs"].rank(ascending=False, method="min").astype(int)
    report = report.sort_values(["rank_mi", "rank_l1", "variable"])

    report["cv_auc_mean_full_model"] = float(np.mean(aucs))
    report["cv_auc_std_full_model"] = float(np.std(aucs))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
