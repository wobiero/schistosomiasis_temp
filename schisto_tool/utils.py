
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from .config import CI_LOWER_Q, CI_UPPER_Q

def _clamp_probability(value: float, eps: float = 1e-6) -> float:
    """Clamp a probability to the open interval (0, 1)."""
    return float(np.clip(float(value), eps, 1.0 - eps))

def _positive_normal(mu: float, sigma: float, rng: np.random.Generator) -> float:
    """Draw a non-negative truncated normal value."""
    mu = float(mu)
    sigma = float(sigma)
    if sigma <= 0:
        return max(mu, 0.0)
    a = (0.0 - mu) / sigma
    return float(truncnorm.rvs(a, np.inf, loc=mu, scale=sigma, random_state=rng))

def _bounded_normal(
    mu: float,
    sigma: float,
    rng: np.random.Generator,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """Draw from a normal distribution truncated to [lower, upper]."""
    mu = float(mu)
    sigma = float(sigma)
    lower = float(lower)
    upper = float(upper)
    if sigma <= 0 or upper <= lower:
        return float(np.clip(mu, lower, upper))
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    return float(truncnorm.rvs(a, b, loc=mu, scale=sigma, random_state=rng))

def _gamma(mu: float, sigma: float, rng: np.random.Generator) -> float:
    """Draw from a Gamma distribution parameterized by mean and SD."""
    mu = float(mu)
    sigma = float(sigma)
    if mu <= 0:
        return 0.0
    if sigma <= 0:
        return mu
    shape = (mu * mu) / (sigma * sigma)
    scale = (sigma * sigma) / mu
    return float(rng.gamma(shape, scale))

def _truncated_beta(mu: float, sigma: float, rng: np.random.Generator) -> float:
    """Draw from a beta distribution using mean and SD."""
    mu = _clamp_probability(mu)
    sigma = max(float(sigma), 1e-9)
    max_sigma = np.sqrt(mu * (1.0 - mu)) * 0.999
    sigma = min(sigma, max_sigma)
    factor = mu * (1.0 - mu) / (sigma * sigma) - 1.0
    alpha = max(mu * factor, 1e-9)
    beta = max((1.0 - mu) * factor, 1e-9)
    return float(rng.beta(alpha, beta))

def _format_ci(mean: float, lo: float, hi: float, fmt: str = ",.0f") -> str:
    """Format a mean and 95% interval, returning NA for non-finite values."""
    vals = np.array([mean, lo, hi], dtype=float)
    if not np.all(np.isfinite(vals)):
        return "NA"
    return f"{mean:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"

def _format_ci_or_na(
    mean: float,
    lo: float,
    hi: float,
    fmt: str = ",.0f",
    na_text: str = "NA",
) -> str:
    """Format a mean and 95% interval, returning na_text for non-finite values."""
    vals = np.array([mean, lo, hi], dtype=float)
    if not np.all(np.isfinite(vals)):
        return na_text
    return f"{mean:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]"

def _finite_series(values: object) -> pd.Series:
    """Return finite numeric values as a pandas Series."""
    ser = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return ser.astype(float)

def _ci_bounds(values: object, lower_q: float = CI_LOWER_Q, upper_q: float = CI_UPPER_Q) -> tuple[float, float, float]:
    """Return mean, lower percentile, and upper percentile for numeric draws."""
    ser = _finite_series(values)
    if ser.empty:
        return (np.nan, np.nan, np.nan)
    return (float(ser.mean()), float(ser.quantile(lower_q)), float(ser.quantile(upper_q)))

def _discount_array(values: object, horizon: int, rate: float = 0.03) -> np.ndarray:
    """Present value of each annual draw over a fixed horizon."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    horizon = int(horizon)
    rate = float(rate)
    if horizon <= 0:
        return np.zeros_like(arr, dtype=float)
    if rate == 0:
        return arr * horizon
    factor = (1.0 - (1.0 + rate) ** (-horizon)) / rate
    return arr * factor

def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    total = int(total)
    successes = int(successes)
    if total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    lo = (centre - half) / denom
    hi = (centre + half) / denom
    return (float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0)))

def _same_length_sum(arrays: list[object]) -> np.ndarray:
    """Sum draw arrays after trimming all non-empty arrays to a common length."""
    clean = [pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float) for arr in arrays if arr is not None and len(arr) > 0]
    if not clean:
        return np.array([], dtype=float)
    min_len = min(arr.size for arr in clean)
    if min_len <= 0:
        return np.array([], dtype=float)
    stacked = np.vstack([np.nan_to_num(arr[:min_len], nan=0.0, posinf=0.0, neginf=0.0) for arr in clean])
    return stacked.sum(axis=0)

def _discount(annual_value: float, horizon: int, rate: float = 0.03) -> float:
    """Present value of a constant annual value over a fixed horizon."""
    annual_value = float(annual_value)
    horizon = int(horizon)
    rate = float(rate)
    if horizon <= 0:
        return 0.0
    if rate == 0:
        return annual_value * horizon
    return annual_value * (1.0 - (1.0 + rate) ** (-horizon)) / rate

def weighted_mean(
    df: pd.DataFrame,
    value_col: str,
    weight_col: str = "PopReq",
    default: float = 0.0,
) -> float:
    """Weighted mean with robust handling of empty data and missing weights."""
    if df.empty or value_col not in df.columns:
        return float(default)

    values = pd.to_numeric(df[value_col], errors="coerce")
    weights = (
        pd.to_numeric(df[weight_col], errors="coerce")
        if weight_col in df.columns
        else pd.Series(1.0, index=df.index)
    )
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        fallback = values.dropna()
        return float(fallback.mean()) if not fallback.empty else float(default)

    values = values[valid]
    weights = weights[valid]
    denom = float(weights.sum())
    return float((values * weights).sum() / denom) if denom > 0 else float(default)

def _widget_scope_key(*parts: object, max_len: int = 180) -> str:
    """Return a stable Streamlit widget-key segment for scoped defaults.

    Streamlit preserves keyed widget values across reruns. Population and
    prevalence widgets need keys that change with the selected geography and
    denominator assumptions; otherwise a national value can remain in the
    disease-input box after the user selects an ADMIN1/ADMIN2/IU. A compact
    digest keeps keys short while avoiding collisions from truncated long names.
    """
    raw = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=8).hexdigest()
    return digest[:max_len]

def species_subset(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Return ESPEN rows relevant to a species based on observed prevalence."""
    if df.empty:
        return df.iloc[0:0].copy()

    sh_prev = pd.to_numeric(
        df.get("sh_prev_pct", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    sm_prev = pd.to_numeric(
        df.get("sm_prev_pct", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)

    if species == "mansoni":
        mask = sm_prev > 0
    elif species == "haematobium":
        mask = sh_prev > 0
    else:
        mask = pd.Series(True, index=df.index)
    return df.loc[mask].copy()
