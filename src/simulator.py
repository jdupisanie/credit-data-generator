from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


@dataclass
class VariableComputed:
    name: str
    bands: List[str]
    p: np.ndarray            # probabilities per band (sum=1)
    gamma: np.ndarray        # bad_rate_ratio per band
    delta: np.ndarray        # computed bad rate per band (probabilities)


@dataclass
class BandInterval:
    band: str
    low: float
    high: float
    include_low: bool
    include_high: bool


_NUMERIC_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _band_contains_decimal(band: str) -> bool:
    return any("." in token for token in _NUMERIC_TOKEN_RE.findall(band))


def _parse_band_interval(band: str, expected_min: float, expected_max: float) -> BandInterval:
    band = band.strip()
    band_lower = band.lower()

    if band_lower == "never":
        return BandInterval(
            band=band,
            low=expected_max,
            high=expected_max,
            include_low=True,
            include_high=True,
        )

    if band.endswith("+"):
        low = float(band[:-1])
        return BandInterval(band=band, low=low, high=expected_max, include_low=True, include_high=True)

    if band.startswith(">="):
        low = float(band[2:])
        return BandInterval(band=band, low=low, high=expected_max, include_low=True, include_high=True)

    if band.startswith(">"):
        low = float(band[1:])
        return BandInterval(band=band, low=low, high=expected_max, include_low=False, include_high=True)

    if band.startswith("<="):
        high = float(band[2:])
        return BandInterval(band=band, low=expected_min, high=high, include_low=True, include_high=True)

    if band.startswith("<"):
        high = float(band[1:])
        return BandInterval(band=band, low=expected_min, high=high, include_low=True, include_high=False)

    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$", band)
    if match:
        low = float(match.group(1))
        high = float(match.group(2))
        return BandInterval(band=band, low=low, high=high, include_low=True, include_high=True)

    try:
        value = float(band)
    except ValueError as exc:
        raise ValueError(f"Unrecognized numeric band format: '{band}'") from exc

    return BandInterval(band=band, low=value, high=value, include_low=True, include_high=True)


def _build_band_intervals(
    bands: List[str],
    expected_min: float,
    expected_max: float,
) -> List[BandInterval]:
    intervals = [_parse_band_interval(band, expected_min, expected_max) for band in bands]

    # If a sentinel band maps to the max value (e.g., "never"),
    # ensure other ranges do not also include the max boundary.
    max_only_bands = {
        interval.band
        for interval in intervals
        if np.isclose(interval.low, expected_max) and np.isclose(interval.high, expected_max)
    }
    if max_only_bands:
        for interval in intervals:
            if interval.band in max_only_bands:
                continue
            if interval.include_high and np.isclose(interval.high, expected_max):
                interval.include_high = False

    # Avoid overlap for adjacent numeric ranges (e.g., 6-12 followed by 12-24).
    for idx in range(len(intervals) - 1):
        curr = intervals[idx]
        next_interval = intervals[idx + 1]
        if curr.include_high and next_interval.include_low:
            if np.isclose(curr.high, next_interval.low):
                curr.include_high = False

    return intervals


def _variable_is_integer(var: Dict[str, Any]) -> bool:
    expected_range = var.get("expected_range") or {}
    min_val = expected_range.get("min")
    max_val = expected_range.get("max")
    if min_val is None or max_val is None:
        return False

    min_float = float(min_val)
    max_float = float(max_val)
    if not (min_float.is_integer() and max_float.is_integer()):
        return False

    for band in var.get("bands", []):
        if _band_contains_decimal(str(band.get("band", ""))):
            return False

    return True


def _sample_from_interval(
    interval: BandInterval,
    size: int,
    rng: np.random.Generator,
    as_int: bool,
) -> np.ndarray:
    low = float(interval.low)
    high = float(interval.high)
    include_low = interval.include_low
    include_high = interval.include_high

    if as_int:
        low_i = int(np.floor(low)) if include_low else int(np.floor(low)) + 1
        high_i = int(np.floor(high)) if include_high else int(np.ceil(high)) - 1
        if high_i < low_i:
            high_i = low_i
        return rng.integers(low_i, high_i + 1, size=size)

    low_f = low if include_low else np.nextafter(low, high)
    high_f = high if include_high else np.nextafter(high, low)
    if not (low_f < high_f):
        return np.full(size, low_f)
    return rng.uniform(low_f, high_f, size=size)


def generate_numeric_values_from_bands(
    band_labels: np.ndarray,
    var: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    expected_range = var.get("expected_range") or {}
    if "min" not in expected_range or "max" not in expected_range:
        raise ValueError(f"Missing expected_range for numeric variable '{var.get('name', '')}'.")

    expected_min = float(expected_range["min"])
    expected_max = float(expected_range["max"])
    bands = [b["band"] for b in var["bands"]]
    intervals = _build_band_intervals(bands, expected_min, expected_max)
    interval_by_band = {interval.band: interval for interval in intervals}

    as_int = _variable_is_integer(var)
    values = np.full(len(band_labels), np.nan, dtype=float)

    for band, interval in interval_by_band.items():
        idx = np.where(band_labels == band)[0]
        if len(idx) == 0:
            continue
        values[idx] = _sample_from_interval(interval, len(idx), rng, as_int)

    if np.isnan(values).any():
        raise ValueError(f"Numeric sampling failed for variable '{var.get('name', '')}'.")

    return values.astype(int) if as_int else values


def band_from_value(value: Any, var: Dict[str, Any]) -> Optional[str]:
    expected_range = var.get("expected_range") or {}
    if "min" not in expected_range or "max" not in expected_range:
        return None

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None

    bands = [b["band"] for b in var["bands"]]
    intervals = _build_band_intervals(bands, float(expected_range["min"]), float(expected_range["max"]))

    for interval in intervals:
        low_ok = numeric_value >= interval.low if interval.include_low else numeric_value > interval.low
        high_ok = numeric_value <= interval.high if interval.include_high else numeric_value < interval.high
        if low_ok and high_ok:
            return interval.band

    return None

def _compute_deltas(p: np.ndarray, gamma: np.ndarray, global_bad_rate: float) -> np.ndarray:
    """
    Compute per-band bad rates delta_j from p_j and gamma_j using the methodology:
      delta_j = k * gamma_j
      k = d / sum(p_j * gamma_j)
    This preserves the ratios between delta values and scales to the global bad rate.
    """
    if not (0 < global_bad_rate < 1):
        raise ValueError("global_bad_rate must be in (0,1).")

    sum_pg = np.sum(p * gamma)
    if sum_pg <= 0:
        raise ValueError("Sum of p * gamma must be positive.")

    k = global_bad_rate / sum_pg
    delta = k * gamma

    # Safety clamp (rare numeric edge cases)
    delta = np.clip(delta, 1e-9, 1 - 1e-9)

    # Sanity: ensure unconditional is close to global_bad_rate
    implied = float(np.sum(p * delta))
    if abs(implied - global_bad_rate) > 1e-3:
        # Not fatal; warn via exception message only if you want strictness
        pass

    return delta


def _flip_to_target(y: np.ndarray, target_ones: int, rng: np.random.Generator) -> np.ndarray:
    """
    Adjust a binary array to have exactly target_ones ones by random flipping.
    """
    y = y.copy()
    k = int(y.sum())
    n = len(y)

    if target_ones < 0 or target_ones > n:
        raise ValueError("target_ones out of bounds.")

    if k > target_ones:
        idx_ones = np.where(y == 1)[0]
        flip = rng.choice(idx_ones, size=(k - target_ones), replace=False)
        y[flip] = 0
    elif k < target_ones:
        idx_zeros = np.where(y == 0)[0]
        flip = rng.choice(idx_zeros, size=(target_ones - k), replace=False)
        y[flip] = 1
    return y


def _simulate_single_variable(
    var: Dict[str, Any],
    n: int,
    global_bad_rate: float,
    rng: np.random.Generator
) -> Tuple[VariableComputed, np.ndarray, np.ndarray]:
    """
    Simulate X (band labels) and Y (defaults) for one variable.
    Returns (computed, X, Y).
    """
    name = var["name"]
    bands = [b["band"] for b in var["bands"]]
    p = np.array([float(b["distribution_pct"]) for b in var["bands"]], dtype=float) / 100.0
    gamma = np.array([float(b["bad_rate_ratio"]) for b in var["bands"]], dtype=float)

    delta = _compute_deltas(p, gamma, global_bad_rate)

    # Draw X
    x_idx = rng.choice(len(bands), size=n, p=p)
    X = np.array([bands[i] for i in x_idx], dtype=object)

    # Draw Y conditional on X via delta
    probs = delta[x_idx]
    Y = rng.binomial(1, probs, size=n).astype(int)

    # Order by Y descending so defaults appear first
    order = np.argsort(-Y)
    X = X[order]
    Y = Y[order]

    computed = VariableComputed(
        name=name,
        bands=bands,
        p=p,
        gamma=gamma,
        delta=delta
    )

    return computed, X, Y

@dataclass
class FinalDefaultSimulationResult:
    p_hat: np.ndarray      # conditional PDs from the fitted model
    y_final: np.ndarray    # re-simulated default indicator (0/1)
    realized_bad_rate: float


def simulate_final_default_indicators(
    df_X: pd.DataFrame,
    encoder: OneHotEncoder,
    model: LogisticRegression,
    rng: np.random.Generator,
    enforce_global_bad_rate: Optional[float] = None,
) -> FinalDefaultSimulationResult:
    """
    Section 3.5 (paper): Replace the initial default indicator with an indicator simulated
    from the conditional distribution given the attributes.

    Steps:
      1) Compute p_hat = P(Y=1 | X) from fitted logistic regression.
      2) Simulate y_final ~ Bernoulli(p_hat) independently per row.
      3) (Optional) Enforce exact overall bad rate by flipping to target count.

    Parameters
    ----------
    df_X : pd.DataFrame
        Attribute columns only (categorical band labels).
    encoder : OneHotEncoder
        Fitted encoder used for the logistic regression.
    model : LogisticRegression
        Fitted logistic regression model.
    rng : np.random.Generator
        RNG for reproducibility.
    enforce_global_bad_rate : float, optional
        If provided, force the final y to have exactly round(n * bad_rate) defaults
        (mirrors the "force to nd" idea used earlier, but applied to final Y).

    Returns
    -------
    FinalDefaultSimulationResult
    """
    # 1) Transform attributes using the same encoder used in model fitting
    X_enc = encoder.transform(df_X)

    # 2) Conditional default probabilities from fitted model
    # sklearn returns two columns: P(Y=0), P(Y=1)
    p_hat = model.predict_proba(X_enc)[:, 1]
    p_hat = np.clip(p_hat, 1e-9, 1 - 1e-9)

    # 3) Simulate final defaults from Bernoulli(p_hat)
    y_final = rng.binomial(1, p_hat, size=len(p_hat)).astype(int)

    # 4) Optional: enforce exact global bad rate
    if enforce_global_bad_rate is not None:
        if not (0 < enforce_global_bad_rate < 1):
            raise ValueError("enforce_global_bad_rate must be in (0,1).")
        target_ones = int(round(len(y_final) * enforce_global_bad_rate))
        y_final = _flip_to_target(y_final, target_ones, rng)

    realized_bad_rate = float(y_final.mean())

    return FinalDefaultSimulationResult(
        p_hat=p_hat,
        y_final=y_final,
        realized_bad_rate=realized_bad_rate
    )
