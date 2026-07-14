"""Schistosomiasis endgame / elimination projections.

Adds a coverage- and frequency-responsive prevalence engine on top of the
existing burden model, plus WHO-target detection and a PSA-based probability
of reaching elimination as a public health problem (EPHP).

Design notes
------------
* Prevalence is handled in PERCENTAGE POINTS (0-100), matching the rest of the
  package (caseloads.py, prevalence.py). The source projector used fractions.
* Decay follows a closed-form geometric model:
      P[t] = p_min + (P0 - p_min) * exp(-delta_eff * t)
  where delta_eff scales the per-standard-round species reduction rate by
  MDA intensity (rounds/year and coverage relative to a 75% reference round).
* EPHP is defined on the prevalence of HEAVY-INTENSITY infection in
  school-age children (SAC), NOT overall prevalence. Overall-prevalence
  targets are provided only for legacy morbidity-control framing.
* This engine is intentionally OPTIMISTIC in the low-prevalence tail (no
  reinfection rebound, groups modelled independently). Use `p_min_pct` to set
  a transmission reservoir floor for realistic endgame scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .cache import cache_data

# --- WHO 2021-2030 roadmap targets (percentage points) ----------------------
# EPHP: elimination as a public health problem.
EPHP_HEAVY_SAC_THRESHOLD_PCT = 1.0
# Legacy morbidity-control context (heavy-intensity), retained for comparison.
MORBIDITY_CONTROL_HEAVY_PCT = 5.0
# Elimination of transmission (interruption) proxy: near-zero overall prevalence.
INTERRUPTION_OVERALL_PCT = 0.5

WHO_EPHP_TARGET_YEAR = 2030

# --- Per-standard-round population prevalence-reduction rate delta -----------
# delta = -ln(1 - per_round_relative_reduction), for a standard annual round at
# the reference coverage. Values from programmatic decay of overall prevalence.
SPECIES_DELTA = {
    "mansoni": float(-np.log(0.67)),      # ~33% relative reduction per round
    "haematobium": float(-np.log(0.54)),  # ~46% relative reduction per round
}

REFERENCE_COVERAGE = 0.75  # coverage defining a "standard" round

# Heavy-intensity infections clear preferentially under repeated PZQ, so the
# heavy-intensity trajectory declines faster than overall prevalence. This is a
# PLACEHOLDER default; calibrate against sentinel-site heavy-intensity data.
DEFAULT_HEAVY_INTENSITY_ACCEL = 1.6


@dataclass(frozen=True)
class EliminationTarget:
    """A WHO-style prevalence target for the endgame view."""

    label: str
    metric: str          # "heavy_sac" or "overall"
    threshold_pct: float
    target_year: int = WHO_EPHP_TARGET_YEAR


def ephp_target(target_year: int = WHO_EPHP_TARGET_YEAR) -> EliminationTarget:
    return EliminationTarget(
        label="EPHP (heavy-intensity SAC < 1%)",
        metric="heavy_sac",
        threshold_pct=EPHP_HEAVY_SAC_THRESHOLD_PCT,
        target_year=int(target_year),
    )


def interruption_target(target_year: int = WHO_EPHP_TARGET_YEAR) -> EliminationTarget:
    return EliminationTarget(
        label="Elimination of transmission (overall ~0%)",
        metric="overall",
        threshold_pct=INTERRUPTION_OVERALL_PCT,
        target_year=int(target_year),
    )


# --- Core mechanics ---------------------------------------------------------

def _frequency_to_rounds_per_year(frequency: str) -> float:
    """Annual -> 1.0 round/yr; Biennial -> 0.5 round/yr."""
    return 0.5 if str(frequency).strip().lower().startswith("bien") else 1.0


def _annual_equivalent_frequency_factor(
    frequency: str,
    frequency_effect_factor: Optional[float] = None,
) -> float:
    """Annual-equivalent delivery factor used when supplied by the app."""
    if not str(frequency).strip().lower().startswith("bien"):
        return 1.0
    if frequency_effect_factor is None:
        return _frequency_to_rounds_per_year(frequency)
    return float(np.clip(float(frequency_effect_factor), 0.0, 1.5))


def _effective_decay(
    species: str,
    coverage_pct: float,
    rounds_per_year: float,
    delta: Optional[float] = None,
) -> float:
    base = SPECIES_DELTA.get(species, SPECIES_DELTA["mansoni"]) if delta is None else float(delta)
    coverage_frac = max(float(coverage_pct), 0.0) / 100.0
    intensity = float(rounds_per_year) * (coverage_frac / REFERENCE_COVERAGE)
    return max(base * intensity, 0.0)


def project_prevalence_geometric(
    years: np.ndarray,
    initial_prev_pct: float,
    species: str,
    coverage_pct: float,
    frequency: str = "Annual",
    p_min_pct: float = 0.0,
    delta: Optional[float] = None,
    decay_accel: float = 1.0,
    frequency_effect_factor: Optional[float] = None,
) -> np.ndarray:
    """Coverage/frequency-responsive geometric prevalence decline (percent)."""
    t = np.atleast_1d(np.asarray(years, dtype=float))
    P0 = float(np.clip(initial_prev_pct, 0.0, 100.0))
    p_min = float(np.clip(p_min_pct, 0.0, P0)) if P0 > 0 else 0.0
    delivery_factor = _annual_equivalent_frequency_factor(frequency, frequency_effect_factor)
    delta_eff = _effective_decay(species, coverage_pct, delivery_factor, delta) * float(decay_accel)
    prev = p_min + (P0 - p_min) * np.exp(-delta_eff * t)
    return np.clip(prev, 0.0, 100.0)


def project_elimination(
    *,
    species: str,
    prev_sac_pct: float,
    prev_adult_pct: float,
    sac_population: float,
    adult_population: float,
    heavy_share: float,
    coverage_pct: float,
    frequency: str = "Annual",
    years: int = 15,
    base_year: int = 2026,
    p_min_pct: float = 0.0,
    heavy_intensity_accel: float = DEFAULT_HEAVY_INTENSITY_ACCEL,
    treat_adults: bool = True,
    delta: Optional[float] = None,
    frequency_effect_factor: Optional[float] = None,
) -> pd.DataFrame:
    """Year-by-year overall and heavy-intensity prevalence for SAC and adults."""
    t = np.arange(0, int(years) + 1, dtype=float)
    adult_cov = float(coverage_pct) if treat_adults else 0.0
    hs = float(np.clip(heavy_share, 0.0, 1.0))

    sac_overall = project_prevalence_geometric(
        t, prev_sac_pct, species, coverage_pct, frequency, p_min_pct, delta,
        frequency_effect_factor=frequency_effect_factor,
    )
    adult_overall = project_prevalence_geometric(
        t, prev_adult_pct, species, adult_cov, frequency, p_min_pct, delta,
        frequency_effect_factor=frequency_effect_factor,
    )
    sac_heavy = project_prevalence_geometric(
        t, prev_sac_pct * hs, species, coverage_pct, frequency,
        p_min_pct * hs, delta, decay_accel=heavy_intensity_accel,
        frequency_effect_factor=frequency_effect_factor,
    )
    adult_heavy = project_prevalence_geometric(
        t, prev_adult_pct * hs, species, adult_cov, frequency,
        p_min_pct * hs, delta, decay_accel=heavy_intensity_accel,
        frequency_effect_factor=frequency_effect_factor,
    )

    total_pop = max(float(sac_population) + float(adult_population), 1.0)
    combined_overall = (
        sac_overall * float(sac_population) + adult_overall * float(adult_population)
    ) / total_pop

    df = pd.DataFrame({
        "Year": (base_year + t).astype(int),
        "t": t.astype(int),
        "SAC_overall_prev_pct": sac_overall,
        "Adult_overall_prev_pct": adult_overall,
        "Combined_overall_prev_pct": combined_overall,
        "SAC_heavy_prev_pct": sac_heavy,
        "Adult_heavy_prev_pct": adult_heavy,
        "SAC_infected": sac_overall / 100.0 * float(sac_population),
        "Adult_infected": adult_overall / 100.0 * float(adult_population),
    })
    df["Total_infected"] = df["SAC_infected"] + df["Adult_infected"]
    return df


def evaluate_target(proj_df: pd.DataFrame, target: EliminationTarget) -> dict:
    """Detect first year the target metric crosses its threshold."""
    col = "SAC_heavy_prev_pct" if target.metric == "heavy_sac" else "Combined_overall_prev_pct"
    below = (proj_df[col] <= target.threshold_pct).to_numpy()
    if below.any():
        first_idx = int(np.argmax(below))
        year_reached: Optional[int] = int(proj_df["Year"].iloc[first_idx])
        years_to: Optional[int] = int(proj_df["t"].iloc[first_idx])
    else:
        year_reached = None
        years_to = None
    on_or_before = proj_df.loc[proj_df["Year"] <= target.target_year, col]
    reached_by_target = bool((on_or_before <= target.threshold_pct).any())
    return {
        "metric_col": col,
        "year_reached": year_reached,
        "years_to_target": years_to,
        "reached_by_target_year": reached_by_target,
        "final_metric_pct": float(proj_df[col].iloc[-1]),
        "target": target,
    }


def cases_averted_summary(
    *,
    species: str,
    prev_sac_pct: float,
    prev_adult_pct: float,
    sac_population: float,
    adult_population: float,
    coverage_pct: float,
    frequency: str,
    years: int,
    treat_adults: bool = True,
    p_min_pct: float = 0.0,
    delta: Optional[float] = None,
    frequency_effect_factor: Optional[float] = None,
) -> dict:
    """Infection-years and final cases averted vs a constant no-MDA counterfactual."""
    t = np.arange(0, int(years) + 1, dtype=float)
    adult_cov = float(coverage_pct) if treat_adults else 0.0

    sac_mda = project_prevalence_geometric(
        t, prev_sac_pct, species, coverage_pct, frequency, p_min_pct, delta,
        frequency_effect_factor=frequency_effect_factor,
    ) / 100.0
    adult_mda = project_prevalence_geometric(
        t, prev_adult_pct, species, adult_cov, frequency, p_min_pct, delta,
        frequency_effect_factor=frequency_effect_factor,
    ) / 100.0
    sac_cf = np.full_like(t, float(np.clip(prev_sac_pct, 0.0, 100.0)) / 100.0)
    adult_cf = np.full_like(t, float(np.clip(prev_adult_pct, 0.0, 100.0)) / 100.0)

    sac_iya = float(sac_population) * float(np.trapz(sac_cf - sac_mda, t))
    adult_iya = float(adult_population) * float(np.trapz(adult_cf - adult_mda, t))
    final_averted = (
        float(sac_population) * float(sac_cf[-1] - sac_mda[-1])
        + float(adult_population) * float(adult_cf[-1] - adult_mda[-1])
    )
    return {
        "infection_years_averted": sac_iya + adult_iya,
        "infection_years_averted_sac": sac_iya,
        "infection_years_averted_adult": adult_iya,
        "final_cases_averted": final_averted,
    }


# --- PSA: probability of reaching the target --------------------------------

def _beta_ab_from_mean_sd(mean: float, sd: float) -> tuple[float, float]:
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    max_sd = np.sqrt(mean * (1.0 - mean)) * 0.999
    sd = float(np.clip(sd, 1e-9, max_sd))
    k = mean * (1.0 - mean) / (sd * sd) - 1.0
    return max(mean * k, 1e-9), max((1.0 - mean) * k, 1e-9)


@cache_data(show_spinner=False)
def probability_of_target(
    species: str,
    prev_sac_pct: float,
    heavy_share: float,
    coverage_pct: float,
    frequency: str,
    target: EliminationTarget,
    base_year: int = 2026,
    n_iter: int = 1_000,
    seed: int = 42,
    per_round_reduction_sd: float = 0.06,
    p_min_pct: float = 0.0,
    heavy_intensity_accel: float = DEFAULT_HEAVY_INTENSITY_ACCEL,
    frequency_effect_factor: Optional[float] = None,
) -> dict:
    """Fraction of PSA draws that reach the target on/before target_year.

    Uncertainty is placed on the per-round relative reduction (Beta), which is
    the dominant driver of whether the low-prevalence target is crossed.
    """
    rng = np.random.default_rng(int(seed))
    base_reduction = 1.0 - float(np.exp(-SPECIES_DELTA.get(species, SPECIES_DELTA["mansoni"])))
    a, b = _beta_ab_from_mean_sd(base_reduction, per_round_reduction_sd)
    red_draws = rng.beta(a, b, size=int(n_iter))
    delta_draws = -np.log(np.clip(1.0 - red_draws, 1e-9, 1.0))

    delivery_factor = _annual_equivalent_frequency_factor(frequency, frequency_effect_factor)
    intensity = (float(coverage_pct) / 100.0) / REFERENCE_COVERAGE * delivery_factor

    use_heavy = target.metric == "heavy_sac"
    accel = heavy_intensity_accel if use_heavy else 1.0
    delta_eff = delta_draws * intensity * accel                      # (n,)

    horizon = max(int(target.target_year) - int(base_year), 0)
    t = np.arange(0, horizon + 1, dtype=float)                        # (T,)

    P0 = float(np.clip(prev_sac_pct, 0.0, 100.0))
    if use_heavy:
        P0 = P0 * float(np.clip(heavy_share, 0.0, 1.0))
    floor = float(np.clip(p_min_pct, 0.0, P0)) * (float(np.clip(heavy_share, 0.0, 1.0)) if use_heavy else 1.0)

    prev = floor + (P0 - floor) * np.exp(-np.outer(delta_eff, t))     # (n, T)
    reached = (prev.min(axis=1) <= target.threshold_pct)
    p = float(reached.mean()) if reached.size else float("nan")

    years_to = np.full(int(n_iter), np.nan)
    below = prev <= target.threshold_pct
    any_below = below.any(axis=1)
    years_to[any_below] = below[any_below].argmax(axis=1)

    return {
        "prob_reached_by_target": p,
        "n_iter": int(n_iter),
        "median_years_to_target": float(np.nanmedian(years_to)) if any_below.any() else float("nan"),
        "target": target,
    }


def compare_elimination_scenarios(
    *,
    species: str,
    prev_sac_pct: float,
    heavy_share: float,
    coverage_options: list[float],
    frequency_options: list[str],
    target: EliminationTarget,
    base_year: int = 2026,
    years: int = 15,
    n_iter: int = 1_000,
    seed: int = 42,
    p_min_pct: float = 0.0,
    heavy_intensity_accel: float = DEFAULT_HEAVY_INTENSITY_ACCEL,
    frequency_effect_factor: Optional[float] = None,
) -> pd.DataFrame:
    """Grid of coverage x frequency: deterministic year reached + P(target)."""
    rows = []
    for freq in frequency_options:
        for cov in coverage_options:
            proj = project_elimination(
                species=species, prev_sac_pct=prev_sac_pct, prev_adult_pct=0.0,
                sac_population=1.0, adult_population=0.0, heavy_share=heavy_share,
                coverage_pct=cov, frequency=freq, years=years, base_year=base_year,
                p_min_pct=p_min_pct, heavy_intensity_accel=heavy_intensity_accel,
                treat_adults=False, frequency_effect_factor=frequency_effect_factor,
            )
            ev = evaluate_target(proj, target)
            pr = probability_of_target(
                species, prev_sac_pct, heavy_share, cov, freq, target,
                base_year=base_year, n_iter=n_iter, seed=seed,
                p_min_pct=p_min_pct, heavy_intensity_accel=heavy_intensity_accel,
                frequency_effect_factor=frequency_effect_factor,
            )
            rows.append({
                "Frequency": freq,
                "Coverage (%)": cov,
                "Frequency effect factor": _annual_equivalent_frequency_factor(freq, frequency_effect_factor),
                "Year target reached": ev["year_reached"],
                "Reached by target year": ev["reached_by_target_year"],
                f"Final {target.metric} (%)": ev["final_metric_pct"],
                f"P(reach by {target.target_year})": pr["prob_reached_by_target"],
            })
    return pd.DataFrame(rows)