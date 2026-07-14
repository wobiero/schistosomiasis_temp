from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

# This module provides deterministic prevalence scenario projections for the
# costing tool. It is intentionally transparent and should not be presented as a
# calibrated dynamic transmission model. Coverage, delivery frequency/effect,
# adult targeting, no-MDA assumptions, species response, and residual floors are
# explicit.
MIN_PREVALENCE_PCT = 0.0
DEFAULT_PROJECTION_FLOOR_PCT = 0.0
REFERENCE_MDA_COVERAGE_PCT = 75.0
REFERENCE_COVERAGE_PCT = REFERENCE_MDA_COVERAGE_PCT  # backward-compatible name
DEFAULT_ADULT_TO_SAC_RATIO = 0.50

# v1.3 default species response convention:
# - Mansoni preserves the v1.2 transparent scenario coefficient.
# - Haematobium uses the same relative faster-response ratio as the elimination
#   module, while keeping the mansoni anchor stable for existing analyses.
DEFAULT_MANSONI_TRAJECTORY_DECAY_SAC = 0.305
DEFAULT_MANSONI_TRAJECTORY_DECAY_ADULT = 0.305
_M_ELIM_DELTA = float(-np.log(0.67))
_H_ELIM_DELTA = float(-np.log(0.54))
_HAEMATOBIUM_TO_MANSONI_RESPONSE_RATIO = _H_ELIM_DELTA / _M_ELIM_DELTA
DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_SAC = float(
    DEFAULT_MANSONI_TRAJECTORY_DECAY_SAC * _HAEMATOBIUM_TO_MANSONI_RESPONSE_RATIO
)
DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_ADULT = float(
    DEFAULT_MANSONI_TRAJECTORY_DECAY_ADULT * _HAEMATOBIUM_TO_MANSONI_RESPONSE_RATIO
)
DEFAULT_TRAJECTORY_DECAY_SAC = DEFAULT_MANSONI_TRAJECTORY_DECAY_SAC
DEFAULT_TRAJECTORY_DECAY_ADULT = DEFAULT_MANSONI_TRAJECTORY_DECAY_ADULT
DEFAULT_SAC_ANNUAL_DECAY = DEFAULT_TRAJECTORY_DECAY_SAC
DEFAULT_ADULT_ANNUAL_DECAY = DEFAULT_TRAJECTORY_DECAY_ADULT
SPECIES_TRAJECTORY_DECAY = {
    "mansoni": {
        "sac": DEFAULT_MANSONI_TRAJECTORY_DECAY_SAC,
        "adult": DEFAULT_MANSONI_TRAJECTORY_DECAY_ADULT,
    },
    "haematobium": {
        "sac": DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_SAC,
        "adult": DEFAULT_HAEMATOBIUM_TRAJECTORY_DECAY_ADULT,
    },
}

DEFAULT_TRAJECTORY_OSCILLATION_PERIOD_YEARS = 1.7
OSCILLATION_AMPLITUDE = 0.18
OSCILLATION_PERIOD_YEARS = DEFAULT_TRAJECTORY_OSCILLATION_PERIOD_YEARS

# None means adult prevalence is allowed to differ from, and exceed, SAC
# prevalence when the data say so. A numeric value can still be passed to apply
# a scenario cap.
ADULT_TO_SAC_PREVALENCE_CAP: float | None = None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _species_key(species: str | None) -> str:
    text = str(species or "mansoni").strip().lower()
    if "haemat" in text or "hemat" in text or text in {"sh", "s_haematobium", "s_hematobium"}:
        return "haematobium"
    return "mansoni"


def rounds_per_year_from_frequency(frequency: str) -> float:
    """Return literal treatment rounds per year for annual vs biennial delivery.

    This helper is retained for legacy code and for contexts where literal
    treatment rounds are needed. The main costing trajectory can instead use an
    analyst-selected annual-equivalent frequency effect so the biennial slider
    drives the same assumption in PSA, sensitivity, and trajectory views.
    """
    return 0.5 if str(frequency).strip().lower().startswith("bien") else 1.0


def _frequency_to_rounds_per_year(frequency: str) -> float:
    """Backward-compatible private alias."""
    return rounds_per_year_from_frequency(frequency)


def annual_equivalent_frequency_factor(
    frequency: str,
    biennial_effect_factor: float | None = None,
) -> float:
    """Return annual-equivalent delivery effect for the selected frequency.

    Annual MDA always has factor 1.0. Biennial MDA uses the analyst-selected
    effect-vs-annual factor when provided. If no factor is provided, the legacy
    literal-rounds assumption is used, so biennial defaults to 0.5.
    """
    if not str(frequency).strip().lower().startswith("bien"):
        return 1.0
    if biennial_effect_factor is None:
        return rounds_per_year_from_frequency(frequency)
    return float(np.clip(_safe_float(biennial_effect_factor, 0.5), 0.0, 1.5))


def frequency_effect_factor_from_frequency(
    frequency: str,
    biennial_effect_factor: float | None = None,
) -> float:
    """Public alias for annual-equivalent frequency effects."""
    return annual_equivalent_frequency_factor(frequency, biennial_effect_factor)


def species_trajectory_decay(species: str | None, age_group: str = "sac") -> float:
    """Return the default trajectory decay parameter for a species and age group."""
    species_key = _species_key(species)
    age_key = "adult" if "adult" in str(age_group).lower() else "sac"
    return float(SPECIES_TRAJECTORY_DECAY.get(species_key, SPECIES_TRAJECTORY_DECAY["mansoni"])[age_key])


def trajectory_decay_at_reference(
    species: str | None = "mansoni",
    age_group: str = "sac",
    fallback: float = DEFAULT_TRAJECTORY_DECAY_SAC,
) -> float:
    """Return the default annual-equivalent decay parameter at 75% annual MDA."""
    try:
        return species_trajectory_decay(species, age_group)
    except Exception:
        return float(fallback)


def _coverage_factor(coverage_pct: float, reference_coverage_pct: float = REFERENCE_MDA_COVERAGE_PCT) -> float:
    coverage = np.clip(_safe_float(coverage_pct), 0.0, 100.0)
    reference = max(_safe_float(reference_coverage_pct, REFERENCE_MDA_COVERAGE_PCT), 1e-9)
    return float(coverage / reference)


def effective_prevalence_decay(
    base_decay: float,
    coverage_pct: float,
    frequency: str = "Annual",
    *,
    reference_coverage_pct: float = REFERENCE_MDA_COVERAGE_PCT,
    frequency_effect_factor: float | None = None,
) -> float:
    """Return coverage- and frequency-adjusted annual prevalence decline.

    A 75% annual MDA scenario returns ``base_decay``. Biennial delivery uses the
    same annual-equivalent effect factor that drives the PSA and sensitivity
    analysis when supplied. Values above 75% coverage increase the decline
    proportionally but do not force elimination because the residual floor is
    applied separately.
    """
    base = max(_safe_float(base_decay), 0.0)
    delivery_factor = annual_equivalent_frequency_factor(frequency, frequency_effect_factor)
    return float(base * _coverage_factor(coverage_pct, reference_coverage_pct) * delivery_factor)


def constrain_initial_adult_prevalence(
    initial_prev_sac: float,
    initial_prev_adult: float,
    max_adult_sac_ratio: float | None = ADULT_TO_SAC_PREVALENCE_CAP,
    *,
    fallback_adult_sac_ratio: float = DEFAULT_ADULT_TO_SAC_RATIO,
) -> float:
    """Return an adult starting prevalence without forcing it to mirror SAC."""
    sac = float(np.clip(_safe_float(initial_prev_sac), 0.0, 100.0))
    adult = float(np.clip(_safe_float(initial_prev_adult), 0.0, 100.0))
    fallback_ratio = max(0.0, _safe_float(fallback_adult_sac_ratio, DEFAULT_ADULT_TO_SAC_RATIO))

    if adult <= 0.0 and sac > 0.0:
        adult = sac * fallback_ratio

    if max_adult_sac_ratio is not None and sac > 0.0:
        cap = sac * max(0.0, _safe_float(max_adult_sac_ratio, 1.0))
        adult = min(adult, cap)

    return float(np.clip(adult, 0.0, 100.0))


def adult_to_sac_ratio(
    initial_prev_sac: float,
    initial_prev_adult: float,
    *,
    fallback_ratio: float = DEFAULT_ADULT_TO_SAC_RATIO,
    max_ratio: float | None = ADULT_TO_SAC_PREVALENCE_CAP,
) -> float:
    """Return the adult:SAC prevalence ratio implied by the inputs."""
    sac = float(np.clip(_safe_float(initial_prev_sac), 0.0, 100.0))
    adult = constrain_initial_adult_prevalence(
        sac,
        initial_prev_adult,
        max_adult_sac_ratio=max_ratio,
        fallback_adult_sac_ratio=fallback_ratio,
    )
    if sac <= 0.0:
        ratio = max(0.0, _safe_float(fallback_ratio, DEFAULT_ADULT_TO_SAC_RATIO))
    else:
        ratio = adult / sac
    if max_ratio is not None:
        ratio = min(ratio, max(0.0, _safe_float(max_ratio, 1.0)))
    return float(max(0.0, ratio))


def split_effective_prevalence(
    effective_prev: float,
    target_multiplier: float,
    adult_sac_ratio: float = DEFAULT_ADULT_TO_SAC_RATIO,
    *,
    max_adult_sac_ratio: float | None = ADULT_TO_SAC_PREVALENCE_CAP,
) -> tuple[float, float]:
    """Split an effective prevalence into SAC and adult starting prevalences.

    This preserves the weighted effective prevalence used for caseloads while
    retaining the requested adult:SAC age profile. Adult prevalence is not capped
    by SAC prevalence unless ``max_adult_sac_ratio`` is explicitly supplied.
    """
    effective = float(np.clip(_safe_float(effective_prev), 0.0, 100.0))
    multiplier = max(_safe_float(target_multiplier, 1.0), 1.0)
    adult_units = multiplier - 1.0
    ratio = max(0.0, _safe_float(adult_sac_ratio, DEFAULT_ADULT_TO_SAC_RATIO))
    if max_adult_sac_ratio is not None:
        ratio = min(ratio, max(0.0, _safe_float(max_adult_sac_ratio, 1.0)))

    if np.isclose(adult_units, 0.0):
        sac = effective
        adult = effective * ratio if ratio > 0 else 0.0
        return float(np.clip(sac, 0.0, 100.0)), float(np.clip(adult, 0.0, 100.0))

    denominator = 1.0 + adult_units * ratio
    sac = effective * multiplier / max(denominator, 1e-12)
    adult = sac * ratio

    # Keep percentages within [0, 100] while preserving the effective prevalence
    # as closely as possible for extreme ratios or high prevalence values.
    if adult > 100.0:
        adult = 100.0
        sac = multiplier * effective - adult_units * adult
    if sac > 100.0:
        sac = 100.0
        adult = (multiplier * effective - sac) / max(adult_units, 1e-12)

    sac = float(np.clip(sac, 0.0, 100.0))
    adult = float(np.clip(adult, 0.0, 100.0))
    return sac, adult


def derive_projection_prevalence_inputs(
    effective_prev: float,
    default_prev_sac: float,
    default_prev_adult: float,
    target_multiplier: float,
    max_adult_sac_ratio: float | None = ADULT_TO_SAC_PREVALENCE_CAP,
) -> tuple[float, float]:
    """Reconcile one effective prevalence with separate SAC/adult inputs."""
    ratio = adult_to_sac_ratio(
        default_prev_sac,
        default_prev_adult,
        max_ratio=max_adult_sac_ratio,
    )
    return split_effective_prevalence(
        effective_prev,
        target_multiplier,
        ratio,
        max_adult_sac_ratio=max_adult_sac_ratio,
    )


def _trajectory_floor(initial_prev_pct: float, floor_pct: float) -> float:
    """Use a floor only below the initial value so trajectories never rise to a floor."""
    initial = float(np.clip(_safe_float(initial_prev_pct), 0.0, 100.0))
    floor = float(np.clip(_safe_float(floor_pct), 0.0, 100.0))
    return float(min(initial, floor))


def project_no_mda_prevalence(
    years: np.ndarray,
    initial_prev_pct: float,
    annual_change_pct: float = 0.0,
) -> np.ndarray:
    """Project the no-MDA comparator prevalence trajectory.

    The default is constant prevalence. Analysts can apply a positive or
    negative annual percentage change for explicit scenario analysis. The change
    is interpreted as ordinary year-on-year compounding, so -5% for 10 years is
    P0 * 0.95^10.
    """
    t = np.atleast_1d(np.asarray(years, dtype=float))
    p0 = float(np.clip(_safe_float(initial_prev_pct), 0.0, 100.0))
    annual_change = _safe_float(annual_change_pct) / 100.0
    annual_factor = max(1.0 + annual_change, 0.0)
    prev = p0 * np.power(annual_factor, t)
    return np.clip(prev, 0.0, 100.0)


def project_mda_prevalence(
    years: np.ndarray,
    initial_prev_pct: float,
    coverage_pct: float,
    frequency: str,
    annual_decay_at_reference: float = DEFAULT_TRAJECTORY_DECAY_SAC,
    reference_coverage_pct: float = REFERENCE_MDA_COVERAGE_PCT,
    floor_pct: float = 0.0,
    oscillation_amplitude: float = 0.0,
    oscillation_period_years: float = DEFAULT_TRAJECTORY_OSCILLATION_PERIOD_YEARS,
    frequency_effect_factor: float | None = None,
) -> np.ndarray:
    """Project prevalence under an MDA scenario using a transparent decay model.

    Formula:
        P_mda(t) = floor + (P0 - floor) * exp(-lambda * I * t)

    where I = annual_equivalent_frequency_effect * coverage / reference_coverage.
    Annual MDA has frequency effect 1.0. Biennial MDA should use the same
    analyst-specified relative effect as the PSA path when supplied. This is a
    deterministic scenario projection for costing and reviewer transparency; it
    is not a calibrated transmission model and should not be labelled as a
    direct SCHISTOX simulation.
    """
    t = np.atleast_1d(np.asarray(years, dtype=float))
    p0 = float(np.clip(_safe_float(initial_prev_pct), 0.0, 100.0))
    floor = _trajectory_floor(p0, floor_pct)
    coverage = float(np.clip(_safe_float(coverage_pct), 0.0, 100.0))
    reference = max(float(reference_coverage_pct), 1e-9)
    delivery_factor = annual_equivalent_frequency_factor(frequency, frequency_effect_factor)
    intensity = delivery_factor * (coverage / reference)
    decay = max(_safe_float(annual_decay_at_reference), 0.0) * intensity

    prev = floor + (p0 - floor) * np.exp(-decay * t)

    amplitude = float(np.clip(_safe_float(oscillation_amplitude), 0.0, 0.50))
    if amplitude > 0.0 and intensity > 0.0:
        period = max(float(oscillation_period_years), 0.25)
        osc = 1.0 + amplitude * min(intensity, 1.5) * np.sin(2.0 * np.pi * t / period)
        prev = floor + np.maximum(prev - floor, 0.0) * osc

    prev = np.clip(prev, floor, 100.0)
    if prev.size:
        prev[0] = p0
    return prev


def project_prevalence_exponential(
    t: np.ndarray,
    initial_prev: float,
    base_decay: float,
    *,
    coverage_pct: float = REFERENCE_MDA_COVERAGE_PCT,
    frequency: str = "Annual",
    p_min_pct: float = DEFAULT_PROJECTION_FLOOR_PCT,
    include_oscillations: bool = False,
    oscillation_amplitude: float = OSCILLATION_AMPLITUDE,
    frequency_effect_factor: float | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper around ``project_mda_prevalence``."""
    return project_mda_prevalence(
        t,
        initial_prev,
        coverage_pct=coverage_pct,
        frequency=frequency,
        annual_decay_at_reference=base_decay,
        floor_pct=p_min_pct,
        oscillation_amplitude=oscillation_amplitude if include_oscillations else 0.0,
        frequency_effect_factor=frequency_effect_factor,
    )


def project_prevalence_sac(
    t: np.ndarray,
    initial_prev: float,
    include_oscillations: bool = False,
    min_prev: float = DEFAULT_PROJECTION_FLOOR_PCT,
    *,
    coverage_pct: float = REFERENCE_MDA_COVERAGE_PCT,
    frequency: str = "Annual",
    species: str = "mansoni",
    annual_decay_at_reference: float | None = None,
    frequency_effect_factor: float | None = None,
) -> np.ndarray:
    """Project SAC prevalence under the selected MDA scenario."""
    decay = (
        species_trajectory_decay(species, "sac")
        if annual_decay_at_reference is None
        else float(annual_decay_at_reference)
    )
    return project_mda_prevalence(
        t,
        initial_prev,
        coverage_pct=coverage_pct,
        frequency=frequency,
        annual_decay_at_reference=decay,
        floor_pct=min_prev,
        oscillation_amplitude=OSCILLATION_AMPLITUDE if include_oscillations else 0.0,
        frequency_effect_factor=frequency_effect_factor,
    )


def _broadcast_like(values: object, target: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 1:
        return np.full(target.shape, float(arr.reshape(-1)[0]), dtype=float)
    return np.reshape(arr, target.shape).astype(float)


def project_prevalence_adult(
    t: np.ndarray,
    initial_prev: float,
    sac_prevalence: np.ndarray | float | None = None,
    *,
    max_adult_sac_ratio: float | None = ADULT_TO_SAC_PREVALENCE_CAP,
    min_prev: float = DEFAULT_PROJECTION_FLOOR_PCT,
    coverage_pct: float = REFERENCE_MDA_COVERAGE_PCT,
    frequency: str = "Annual",
    species: str = "mansoni",
    annual_decay_at_reference: float | None = None,
    frequency_effect_factor: float | None = None,
) -> np.ndarray:
    """Project adult prevalence under the selected MDA scenario."""
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    decay = (
        species_trajectory_decay(species, "adult")
        if annual_decay_at_reference is None
        else float(annual_decay_at_reference)
    )
    prev = project_mda_prevalence(
        t_arr,
        initial_prev,
        coverage_pct=coverage_pct,
        frequency=frequency,
        annual_decay_at_reference=decay,
        floor_pct=min_prev,
        frequency_effect_factor=frequency_effect_factor,
    )

    if sac_prevalence is None or max_adult_sac_ratio is None:
        return prev

    sac_arr = _broadcast_like(sac_prevalence, t_arr)
    sac_arr = np.clip(np.nan_to_num(sac_arr, nan=0.0, posinf=100.0, neginf=0.0), 0.0, 100.0)
    cap = np.clip(sac_arr * max(0.0, _safe_float(max_adult_sac_ratio, 1.0)), 0.0, 100.0)
    local_floor = np.minimum(float(min_prev), cap)
    return np.clip(np.minimum(np.maximum(prev, local_floor), cap), 0.0, 100.0)


def build_prevalence_scenario_df(
    years: np.ndarray,
    initial_prev_sac: float,
    initial_prev_adult: float,
    coverage_pct: float,
    frequency: str,
    target_multiplier: float = 1.0,
    treat_adults: bool = True,
    annual_decay_sac_at_reference: float | None = None,
    annual_decay_adult_at_reference: float | None = None,
    no_mda_annual_change_pct: float = 0.0,
    floor_pct: float = 0.0,
    reference_coverage_pct: float = REFERENCE_MDA_COVERAGE_PCT,
    include_oscillations: bool = False,
    oscillation_amplitude: float = 0.10,
    species: str = "mansoni",
    frequency_effect_factor: float | None = None,
) -> pd.DataFrame:
    """Return no-MDA and MDA prevalence trajectories for SAC, adults, and combined target."""
    years = np.atleast_1d(np.asarray(years, dtype=float))
    multiplier = max(_safe_float(target_multiplier, 1.0), 1.0)
    adult_units = max(multiplier - 1.0, 0.0)
    adult_coverage = coverage_pct if treat_adults else 0.0
    amp = oscillation_amplitude if include_oscillations else 0.0
    sac_decay = (
        species_trajectory_decay(species, "sac")
        if annual_decay_sac_at_reference is None
        else float(annual_decay_sac_at_reference)
    )
    adult_decay = (
        species_trajectory_decay(species, "adult")
        if annual_decay_adult_at_reference is None
        else float(annual_decay_adult_at_reference)
    )
    delivery_factor = annual_equivalent_frequency_factor(frequency, frequency_effect_factor)
    nominal_rounds = rounds_per_year_from_frequency(frequency)

    sac_no = project_no_mda_prevalence(years, initial_prev_sac, no_mda_annual_change_pct)
    adult_no = project_no_mda_prevalence(years, initial_prev_adult, no_mda_annual_change_pct)
    sac_mda = project_mda_prevalence(
        years,
        initial_prev_sac,
        coverage_pct=coverage_pct,
        frequency=frequency,
        annual_decay_at_reference=sac_decay,
        reference_coverage_pct=reference_coverage_pct,
        floor_pct=floor_pct,
        oscillation_amplitude=amp,
        frequency_effect_factor=delivery_factor,
    )
    adult_mda = project_mda_prevalence(
        years,
        initial_prev_adult,
        coverage_pct=adult_coverage,
        frequency=frequency,
        annual_decay_at_reference=adult_decay,
        reference_coverage_pct=reference_coverage_pct,
        floor_pct=floor_pct,
        oscillation_amplitude=amp,
        frequency_effect_factor=delivery_factor if adult_coverage > 0 else 0.0,
    )

    def _combined(sac: np.ndarray, adult: np.ndarray) -> np.ndarray:
        if adult_units <= 0.0:
            return sac
        return (sac + adult * adult_units) / multiplier

    scenario_specs = [
        ("No-MDA counterfactual", sac_no, adult_no, 0.0, 0.0, 0.0),
        (
            "MDA scenario",
            sac_mda,
            adult_mda,
            delivery_factor * float(coverage_pct) / max(reference_coverage_pct, 1e-9),
            delivery_factor * float(adult_coverage) / max(reference_coverage_pct, 1e-9),
            delivery_factor,
        ),
    ]

    rows = []
    for scenario, sac, adult, sac_intensity, adult_intensity, scenario_delivery_factor in scenario_specs:
        combined = _combined(sac, adult)
        for year, p_sac, p_adult, p_combined in zip(years, sac, adult, combined):
            rows.append(
                {
                    "Year": float(year),
                    "Scenario": scenario,
                    "Prev_SAC": float(p_sac),
                    "Prev_Adult": float(p_adult),
                    "Prev_Combined": float(p_combined),
                    "MDA_coverage_pct": float(coverage_pct),
                    "Adult_coverage_pct": float(adult_coverage),
                    "Frequency": str(frequency),
                    "Species": _species_key(species),
                    "Nominal_rounds_per_year": float(nominal_rounds),
                    "Frequency_effect_factor": float(scenario_delivery_factor),
                    "Trajectory_decay_SAC": float(sac_decay),
                    "Trajectory_decay_Adult": float(adult_decay),
                    "SAC_decay_at_reference": float(sac_decay),
                    "Adult_decay_at_reference": float(adult_decay),
                    "SAC_MDA_intensity": float(sac_intensity),
                    "Adult_MDA_intensity": float(adult_intensity),
                    "No_MDA_annual_change_pct": float(no_mda_annual_change_pct),
                    "Residual_floor_pct": float(floor_pct),
                }
            )
    return pd.DataFrame(rows)


def build_prevalence_projection_df(
    years: np.ndarray,
    initial_prev_sac: float,
    initial_prev_adult: float,
    include_oscillations: bool = False,
    *,
    max_adult_sac_ratio: float | None = ADULT_TO_SAC_PREVALENCE_CAP,
    coverage_pct: float = REFERENCE_MDA_COVERAGE_PCT,
    frequency: str = "Annual",
    species: str = "mansoni",
    p_min_pct: float = DEFAULT_PROJECTION_FLOOR_PCT,
    target_multiplier: float = 1.0,
    include_no_mda: bool = False,
    frequency_effect_factor: float | None = None,
) -> pd.DataFrame:
    """Build a legacy wide dataframe for selected MDA and optional no-MDA trajectories."""
    initial_prev_sac = float(np.clip(_safe_float(initial_prev_sac), 0.0, 100.0))
    initial_prev_adult = constrain_initial_adult_prevalence(
        initial_prev_sac,
        initial_prev_adult,
        max_adult_sac_ratio=max_adult_sac_ratio,
    )
    scenario_df = build_prevalence_scenario_df(
        years=years,
        initial_prev_sac=initial_prev_sac,
        initial_prev_adult=initial_prev_adult,
        coverage_pct=coverage_pct,
        frequency=frequency,
        target_multiplier=target_multiplier,
        treat_adults=True,
        annual_decay_sac_at_reference=None,
        annual_decay_adult_at_reference=None,
        floor_pct=p_min_pct,
        include_oscillations=include_oscillations,
        species=species,
        frequency_effect_factor=frequency_effect_factor,
    )
    mda = scenario_df.loc[scenario_df["Scenario"] == "MDA scenario"].copy()
    out = mda[["Year", "Prev_SAC", "Prev_Adult", "Prev_Combined"]].reset_index(drop=True)
    if include_no_mda:
        no_mda = scenario_df.loc[scenario_df["Scenario"] == "No-MDA counterfactual"].reset_index(drop=True)
        out["Prev_SAC_no_mda"] = no_mda["Prev_SAC"]
        out["Prev_Adult_no_mda"] = no_mda["Prev_Adult"]
        out["Prev_Combined_no_mda"] = no_mda["Prev_Combined"]
    return out


def project_caseloads_from_prevalence_scenario(
    prevalence_df: pd.DataFrame,
    scenario: str,
    at_risk_pop: float,
    caseload_fn: Callable[..., dict[str, Any]],
    caseload_params: Any,
) -> pd.DataFrame:
    """Project caseloads from the combined prevalence in a scenario dataframe."""
    if prevalence_df is None or prevalence_df.empty:
        return pd.DataFrame()
    sub = prevalence_df.loc[prevalence_df["Scenario"] == scenario].copy()
    rows = []
    for _, row in sub.iterrows():
        cl = caseload_fn(float(at_risk_pop), float(row["Prev_Combined"]), caseload_params)
        cl["Year"] = row["Year"]
        cl["Scenario"] = scenario
        cl["Prev_SAC"] = row["Prev_SAC"]
        cl["Prev_Adult"] = row["Prev_Adult"]
        cl["Prev_Combined"] = row["Prev_Combined"]
        rows.append(cl)
    return pd.DataFrame(rows)


def project_caseloads_over_time(
    years: np.ndarray,
    at_risk_pop: float,
    initial_prev_sac: float,
    initial_prev_adult: float,
    caseload_fn: Callable[..., dict[str, Any]],
    caseload_params: Any,
    include_oscillations: bool = False,
    target_multiplier: float = 1.0,
    *,
    max_adult_sac_ratio: float | None = ADULT_TO_SAC_PREVALENCE_CAP,
    coverage_pct: float = REFERENCE_MDA_COVERAGE_PCT,
    frequency: str = "Annual",
    species: str = "mansoni",
    p_min_pct: float = DEFAULT_PROJECTION_FLOOR_PCT,
    frequency_effect_factor: float | None = None,
) -> pd.DataFrame:
    """Project caseloads year-by-year using age-specific time-varying prevalence."""
    prev_df = build_prevalence_projection_df(
        years,
        initial_prev_sac,
        initial_prev_adult,
        include_oscillations,
        max_adult_sac_ratio=max_adult_sac_ratio,
        coverage_pct=coverage_pct,
        frequency=frequency,
        species=species,
        p_min_pct=p_min_pct,
        target_multiplier=target_multiplier,
        include_no_mda=False,
        frequency_effect_factor=frequency_effect_factor,
    )

    rows = []
    for _, row in prev_df.iterrows():
        cl = caseload_fn(at_risk_pop, row["Prev_Combined"], caseload_params)
        cl["Year"] = row["Year"]
        cl["Prev_SAC"] = row["Prev_SAC"]
        cl["Prev_Adult"] = row["Prev_Adult"]
        cl["Prev_Combined"] = row["Prev_Combined"]
        rows.append(cl)

    return pd.DataFrame(rows)
