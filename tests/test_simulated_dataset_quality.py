import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simulator import band_from_value


DISTRIBUTION_PCT_TOL = 2.0  # percentage points
BAD_RATE_TOL = 0.05         # absolute probability
GAMMA_TOL = 10.15            # absolute ratio difference


def _load_dataset(dataset_path: Path):
    with dataset_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise AssertionError("simulated_dataset_total.csv is empty.")
    return rows


def test_simulated_dataset_matches_metadata():
    dataset_path = Path("analytics") / "data_analysis" / "artifacts" / "01_datasets" / "simulated_dataset_total.csv"
    metadata_path = Path("analytics") / "data_analysis" / "artifacts" / "01_datasets" / "simulated_metadata.json"
    report_path = Path("analytics") / "data_analysis" / "artifacts" / "01_datasets" / "simulated_quality_report.json"
    variables_path = Path("input_parameters") / "variables.json"

    if not dataset_path.exists():
        pytest.skip("Missing analytics/data_analysis/artifacts/01_datasets/simulated_dataset_total.csv. Run main.py to run this quality test.")
    if not metadata_path.exists():
        pytest.skip("Missing analytics/data_analysis/artifacts/01_datasets/simulated_metadata.json. Run main.py to run this quality test.")

    rows = _load_dataset(dataset_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    variables_cfg = json.loads(variables_path.read_text(encoding="utf-8"))
    meta_items = metadata["variables"] if isinstance(metadata, dict) else metadata
    var_by_name = {
        item["name"]: item for item in variables_cfg["dataset_spec"]["variables"]
    }

    meta_by_name = {item["name"]: item for item in meta_items}
    n = len(rows)

    failures = []
    report_payload = {"variables": []}
    failure_lines = []

    for name, meta in meta_by_name.items():
        if name not in rows[0]:
            failures.append(f"{name}: column missing in dataset.")
            continue

        bands = meta["bands"]
        expected_p = meta["p"]
        expected_delta = meta["delta"]
        expected_gamma = meta["gamma"]

        counts = {band: 0 for band in bands}
        bad_counts = {band: 0 for band in bands}

        for row in rows:
            value = row[name]
            var_cfg = var_by_name.get(name, {})
            if var_cfg.get("type") == "numeric":
                band = band_from_value(value, var_cfg)
                if band is None:
                    failures.append(f"{name}: value '{value}' could not be mapped to a band.")
                    continue
            else:
                band = value
            if band not in counts:
                failures.append(f"{name}: unexpected band '{band}'.")
                continue
            counts[band] += 1
            if row["default"] == "1":
                bad_counts[band] += 1

        dist_diffs = []
        bad_rate_diffs = []
        per_band = []

        for idx, band in enumerate(bands):
            count = counts[band]
            expected_dist_pct = expected_p[idx] * 100.0
            actual_dist_pct = (count / n) * 100.0
            dist_diff = abs(actual_dist_pct - expected_dist_pct)
            dist_diffs.append(dist_diff)

            if count > 0:
                actual_bad_rate = bad_counts[band] / count
                bad_rate_diff = abs(actual_bad_rate - expected_delta[idx])
                bad_rate_diffs.append(bad_rate_diff)
            else:
                actual_bad_rate = None
                bad_rate_diff = float("inf")
                bad_rate_diffs.append(bad_rate_diff)

            per_band.append(
                {
                    "band": band,
                    "count": count,
                    "expected_dist_pct": expected_dist_pct,
                    "actual_dist_pct": actual_dist_pct,
                    "dist_diff_pct_points": dist_diff,
                    "expected_bad_rate": expected_delta[idx],
                    "actual_bad_rate": actual_bad_rate,
                    "bad_rate_diff": bad_rate_diff,
                }
            )

        # Gamma comparison (ratio-normalized)
        min_expected_gamma = min(expected_gamma)
        expected_gamma_norm = [g / min_expected_gamma for g in expected_gamma]

        actual_bad_rates = []
        for band in bands:
            count = counts[band]
            if count > 0:
                actual_bad_rates.append(bad_counts[band] / count)
            else:
                actual_bad_rates.append(0.0)

        min_actual_bad = min([r for r in actual_bad_rates if r > 0.0] or [0.0])
        gamma_diffs = []
        if min_actual_bad > 0.0:
            for idx, rate in enumerate(actual_bad_rates):
                gamma_diffs.append(abs((rate / min_actual_bad) - expected_gamma_norm[idx]))
        else:
            gamma_diffs = [float("inf")] * len(bands)

        max_dist = max(dist_diffs)
        max_bad = max(bad_rate_diffs)
        max_gamma = max(gamma_diffs)

        report_payload["variables"].append(
            {
                "name": name,
                "max_dist_diff_pct_points": max_dist,
                "max_bad_rate_diff": max_bad,
                "max_gamma_diff": max_gamma,
                "bands": per_band,
                "expected_gamma_norm": expected_gamma_norm,
                "actual_bad_rates": actual_bad_rates,
                "actual_gamma_norm": (
                    [r / min_actual_bad for r in actual_bad_rates] if min_actual_bad > 0.0 else []
                ),
            }
        )

        if max_dist > DISTRIBUTION_PCT_TOL:
            failures.append(name)
            failing_bands = [
                f"{b['band']} (dist {b['dist_diff_pct_points']:.2f}pp)"
                for b in per_band
                if b["dist_diff_pct_points"] > DISTRIBUTION_PCT_TOL
            ]
            failure_lines.append(
                f"{name}: distribution out of tolerance -> " + ", ".join(failing_bands)
            )
        if max_bad > BAD_RATE_TOL:
            failures.append(name)
            failing_bands = [
                f"{b['band']} (bad rate diff {b['bad_rate_diff']:.4f})"
                for b in per_band
                if b["bad_rate_diff"] > BAD_RATE_TOL
            ]
            failure_lines.append(
                f"{name}: bad rate out of tolerance -> " + ", ".join(failing_bands)
            )
        if max_gamma > GAMMA_TOL:
            failures.append(name)
            failing_bands = [
                f"{band} (gamma diff {gamma_diffs[idx]:.3f})"
                for idx, band in enumerate(bands)
                if gamma_diffs[idx] > GAMMA_TOL
            ]
            failure_lines.append(
                f"{name}: gamma out of tolerance -> " + ", ".join(failing_bands)
            )

    report_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")

    if failure_lines:
        print("\nSimulation fit failures:\n" + "\n".join(failure_lines))

    assert not failures, "Simulation mismatches:\n" + "\n".join(failure_lines)
