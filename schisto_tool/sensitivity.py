from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .health_sector import extract_mean_utilization
from .prevalence import (
    DEFAULT_PROJECTION_FLOOR_PCT,
    project_prevalence_sac,
    annual_equivalent_frequency_factor,
    species_trajectory_decay,
)


def adjust_mda_effect_multiplier(
    coverage_pct: float,
    frequency_factor: float,
) -> float:
    """Compute adjusted MDA intensity for sensitivity analysis.

    This remains as a backward-compatible helper for charts and tests. Values
    are coverage fraction x annual-equivalent delivery factor.
    """
    return float(np.clip((coverage_pct / 100.0) * frequency_factor, 0.0, 1.0))


def _frequency_name_from_factor(frequency_factor: float) -> str:
    return "Biennial" if float(frequency_factor) <= 0.75 else "Annual"


def _scenario_annualized_programme_cost(
    frequency: str,
    fallback_annualized_cost: float,
    mda_year_prog_cost: float | None = None,
    off_year_fixed_cost: float = 0.0,
) -> float:
    """Return scenario-specific annualized programme cost."""
    if mda_year_prog_cost is None:
        return float(fallback_annualized_cost)
    if str(frequency).strip().lower().startswith("bien"):
        return (float(mda_year_prog_cost) + float(off_year_fixed_cost)) / 2.0
    return float(mda_year_prog_cost)


def project_prevalence_with_multiplier(
    t: np.ndarray,
    initial_prev: float,
    mda_effect_multiplier: float,
    *,
    p_min_pct: float = DEFAULT_PROJECTION_FLOOR_PCT,
    species: str = "mansoni",
    annual_decay_at_reference: float | None = None,
) -> np.ndarray:
    """Project prevalence from an annual-equivalent MDA intensity.

    A multiplier of 0.75 corresponds to the reference 75% annual MDA scenario
    and therefore returns the default SAC decline rate. This replaces the older
    ad-hoc 0.05-to-0.305 interpolation used in the sensitivity plot.
    """
    coverage_pct = 100.0 * float(np.clip(mda_effect_multiplier, 0.0, 1.0))
    decay = species_trajectory_decay(species, "sac") if annual_decay_at_reference is None else annual_decay_at_reference
    return project_prevalence_sac(
        t,
        initial_prev,
        include_oscillations=False,
        min_prev=p_min_pct,
        coverage_pct=coverage_pct,
        frequency="Annual",
        species=species,
        annual_decay_at_reference=decay,
    )


def run_sensitivity_analysis(
    coverage_range: List[float],
    frequency_scenarios: Dict[str, float],
    time_horizon: int,
    at_risk_pop: float,
    initial_prev_sac: float,
    caseload_fn,
    caseload_params,
    annual_prog_cost: float,
    psa_df,
    opd_cost: float,
    ipd_cost: float,
    discount_rate: float = 0.03,
    p_min_pct: float = DEFAULT_PROJECTION_FLOOR_PCT,
    species: str = "mansoni",
    annual_decay_at_reference: float | None = None,
    mda_year_prog_cost: float | None = None,
    off_year_fixed_cost: float = 0.0,
) -> pd.DataFrame:
    """Run sensitivity analysis across coverage levels and frequency scenarios.

    ``frequency_scenarios`` maps frequency labels to annual-equivalent effect
    factors. For example, {"Biennial": 0.70} makes the biennial trajectory use
    70% of the annual prevalence-response effect, matching the sidebar slider
    and the PSA effect multiplier. Scenario-specific annualized programme costs
    are used when MDA-year and off-year costs are supplied.
    """
    results = []
    decay = species_trajectory_decay(species, "sac") if annual_decay_at_reference is None else annual_decay_at_reference

    for freq_name, freq_factor in frequency_scenarios.items():
        frequency = str(freq_name) if freq_name else _frequency_name_from_factor(freq_factor)
        delivery_factor = annual_equivalent_frequency_factor(frequency, freq_factor)
        scenario_prog_cost = _scenario_annualized_programme_cost(
            frequency,
            annual_prog_cost,
            mda_year_prog_cost=mda_year_prog_cost,
            off_year_fixed_cost=off_year_fixed_cost,
        )
        for cov in coverage_range:
            years = np.arange(0, time_horizon + 1, 1)
            prev = project_prevalence_sac(
                years,
                initial_prev_sac,
                include_oscillations=False,
                min_prev=p_min_pct,
                coverage_pct=cov,
                frequency=frequency,
                species=species,
                annual_decay_at_reference=decay,
                frequency_effect_factor=delivery_factor,
            )

            for yr in [5, 10]:
                if yr > time_horizon:
                    continue

                prev_yr = prev[yr]
                cl = caseload_fn(at_risk_pop, prev_yr, caseload_params)
                infected = cl.get("infected", 0.0)

                util = extract_mean_utilization(psa_df, species)
                opd_visits = infected * util.get("opd_visits_per_infected", 0.0)
                ipd_days = infected * util.get("ipd_days_per_infected", 0.0)
                hs_cost_yr = opd_visits * opd_cost + ipd_days * ipd_cost

                infected_baseline = at_risk_pop * (initial_prev_sac / 100.0)
                baseline_opd_visits = infected_baseline * util.get("opd_visits_per_infected", 0.0)
                baseline_ipd_days = infected_baseline * util.get("ipd_days_per_infected", 0.0)
                baseline_hs_cost_yr = baseline_opd_visits * opd_cost + baseline_ipd_days * ipd_cost

                disc = (1.0 + discount_rate) ** yr
                hs_cost_disc = hs_cost_yr / disc
                hs_savings_disc = (baseline_hs_cost_yr - hs_cost_yr) / disc
                prog_cost_yr = scenario_prog_cost / disc
                cases_averted = infected_baseline - infected

                results.append({
                    "Frequency": frequency,
                    "Frequency_effect_factor": delivery_factor,
                    "Coverage (%)": cov,
                    "Year": yr,
                    "Prevalence (%)": prev_yr,
                    "Infected": infected,
                    "Cases_averted": cases_averted,
                    "HS_cost_discounted_USD": hs_cost_disc,
                    "HS_savings_discounted_USD": hs_savings_disc,
                    "Annualized_programme_cost_USD": scenario_prog_cost,
                    "Programme_cost_discounted_USD": prog_cost_yr,
                    "Net_benefit_USD": hs_savings_disc - prog_cost_yr,
                })

    return pd.DataFrame(results)


def sensitivity_summary_table(
    sensitivity_df: pd.DataFrame,
    year: int = 10,
) -> pd.DataFrame:
    """Create a summary comparison table at target year."""
    df = sensitivity_df[sensitivity_df["Year"] == year].copy()

    summary = df[
        [
            "Frequency",
            "Frequency_effect_factor",
            "Coverage (%)",
            "Prevalence (%)",
            "Cases_averted",
            "HS_cost_discounted_USD",
            "HS_savings_discounted_USD",
            "Programme_cost_discounted_USD",
            "Net_benefit_USD",
        ]
    ].copy()

    summary.columns = [
        "Frequency",
        "Frequency effect factor",
        "Coverage (%)",
        "Prevalence (%)",
        "Cases averted",
        "HS cost with MDA (USD)",
        "HS savings (USD)",
        "Prog cost (USD)",
        "Net benefit (USD)",
    ]

    return summary.sort_values(["Frequency", "Coverage (%)"])


def identify_optimal_scenario(
    sensitivity_df: pd.DataFrame,
    year: int = 10,
    criterion: str = "net_benefit",
) -> dict:
    """Identify the optimal coverage/frequency scenario based on a criterion."""
    df = sensitivity_df[sensitivity_df["Year"] == year].copy()

    if criterion == "net_benefit":
        optimal = df.loc[df["Net_benefit_USD"].idxmax()]
    elif criterion == "cases_averted":
        optimal = df.loc[df["Cases_averted"].idxmax()]
    else:
        optimal = df.iloc[0]

    return {
        "frequency": optimal["Frequency"],
        "coverage": optimal["Coverage (%)"],
        "cases_averted": optimal["Cases_averted"],
        "net_benefit": optimal["Net_benefit_USD"],
        "prev_final": optimal["Prevalence (%)"],
    }
