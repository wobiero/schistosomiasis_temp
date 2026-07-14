
from __future__ import annotations

import numpy as np
import pandas as pd

from .parameters import HaematobiumInputs, MansoniInputs
from .utils import _clamp_probability

def partitioned_species_defaults(
    df: pd.DataFrame,
    species: str,
    both_species_mansoni_share: float = 0.50,
) -> dict:
    """Return allocated PopReq and prevalence defaults for one species module."""
    empty = {
        "rows": 0, "pop_req": 0.0, "prev_sac": 0.0,
        "prev_adults": 0.0, "both_pop_allocated": 0.0,
    }
    if df is None or df.empty:
        return empty

    sh_prev = pd.to_numeric(
        df.get("sh_prev_pct", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    sm_prev = pd.to_numeric(
        df.get("sm_prev_pct", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)

    has_sh = sh_prev > 0
    has_sm = sm_prev > 0
    both_mask = has_sh & has_sm

    if species == "mansoni":
        base_mask = has_sm & ~both_mask
        species_share = pd.to_numeric(
            df.get("sm_share", pd.Series(1.0, index=df.index)), errors="coerce"
        ).fillna(1.0)
        prevalence_col = "sm_prev_pct"
    elif species == "haematobium":
        base_mask = has_sh & ~both_mask
        species_share = pd.to_numeric(
            df.get("sh_share", pd.Series(1.0, index=df.index)), errors="coerce"
        ).fillna(1.0)
        prevalence_col = "sh_prev_pct"
    else:
        base_mask = pd.Series(True, index=df.index)
        species_share = pd.Series(1.0, index=df.index)
        prevalence_col = None

    pop_req = pd.to_numeric(
        df.get("PopReq", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    weights = pd.Series(0.0, index=df.index, dtype=float)

    weights.loc[base_mask] = pop_req.loc[base_mask] * species_share.loc[base_mask]

    if both_mask.any():
        # Rows with evidence for both species do not identify a species-specific
        # population denominator. Use the UI/manual split for population
        # allocation, while retaining species-specific prevalence columns below
        # for each module's prevalence default.
        mansoni_both_share = float(np.clip(both_species_mansoni_share, 0.0, 1.0))
        if species == "mansoni":
            both_share = mansoni_both_share
        elif species == "haematobium":
            both_share = 1.0 - mansoni_both_share
        else:
            both_share = 1.0
        weights.loc[both_mask] = pop_req.loc[both_mask] * both_share

    valid = weights > 0
    pop_total = float(weights.loc[valid].sum())
    both_allocated = (
        float(weights.loc[both_mask & valid].sum()) if both_mask.any() else 0.0
    )

    def _weighted(col: str, default: float = 0.0) -> float:
        if col not in df.columns or not valid.any():
            return float(default)
        values = pd.to_numeric(df[col], errors="coerce")
        ok = valid & values.notna()
        denom = float(weights.loc[ok].sum())
        if denom <= 0:
            fallback = values.loc[ok].dropna()
            return float(fallback.mean()) if not fallback.empty else float(default)
        return float((values.loc[ok] * weights.loc[ok]).sum() / denom)

    if prevalence_col and prevalence_col in df.columns:
        prev_sac = _weighted(prevalence_col, 0.0)
    else:
        prev_sac = _weighted("Prev_SAC", 0.0)
    prev_adults = _weighted("Prev_Adults", prev_sac)

    if not np.isfinite(prev_sac):
        prev_sac = 0.0
    if not np.isfinite(prev_adults):
        prev_adults = prev_sac
    prev_sac = float(np.clip(prev_sac, 0.0, 100.0))
    prev_adults = float(np.clip(prev_adults, 0.0, 100.0))
    if not np.isfinite(pop_total):
        pop_total = 0.0

    return {
        "rows": int(valid.sum()),
        "pop_req": pop_total,
        "prev_sac": prev_sac,
        "prev_adults": prev_adults,
        "both_pop_allocated": both_allocated,
    }

def intensity_morbidity_effect(egg_reduction_rate: float) -> float:
    """Return the ERR-based effect used for infection-intensity-mediated morbidity."""
    return float(np.clip(float(egg_reduction_rate), 0.0, 1.0))

def effective_prevalence(
    sac_prev: float,
    adult_prev: float,
    target_multiplier: float,
) -> float:
    """Combine SAC and adult prevalence when adult treatment is selected."""
    target_multiplier = max(float(target_multiplier), 1.0)
    if np.isclose(target_multiplier, 1.0):
        return float(sac_prev)
    adult_units = target_multiplier - 1.0
    return float((sac_prev + adult_prev * adult_units) / target_multiplier)

def threshold_message(icer_mean: float, cet: float, annual_ppp: float) -> str:
    if not np.isfinite(icer_mean):
        return "not evaluable because DALYs averted are zero or negative"
    lower = min(cet, annual_ppp)
    upper = max(cet, annual_ppp)
    if icer_mean < lower:
        return "cost-effective under both threshold definitions"
    if icer_mean < upper:
        return "cost-effective under one threshold definition"
    return "not cost-effective under either threshold definition"

def estimate_caseloads_mansoni(
    at_risk_pop: float,
    prev_pct: float,
    m_params: MansoniInputs,
) -> dict:
    infected = float(at_risk_pop) * (float(prev_pct) / 100.0)
    heavy_share = _clamp_probability(m_params.pct_heavy)
    heavy_infected = infected * heavy_share
    light_infected = max(infected - heavy_infected, 0.0)
    anemia_cases = (
        light_infected * m_params.pct_light_anemia
        + heavy_infected * m_params.pct_heavy_anemia
    )
    hepatomegaly_cases = heavy_infected * m_params.pct_hepatomegaly
    fibrosis_cases = hepatomegaly_cases * m_params.pct_fibrosis
    portal_htn_cases = fibrosis_cases * m_params.pct_portal_htn
    varices_cases = portal_htn_cases * m_params.pct_varices

    return {
        "infected": infected,
        "light_infected": light_infected,
        "heavy_infected": heavy_infected,
        "anemia": anemia_cases,
        "hepatomegaly": hepatomegaly_cases,
        "fibrosis": fibrosis_cases,
        "portal_htn": portal_htn_cases,
        "varices": varices_cases,
    }

def _bladder_cancer_case_components(
    at_risk_pop: float,
    prevalence_fraction: float,
    bladder_cancer_rate_per_100k: float,
    relative_risk: float,
    primary_share: float,
    effect_reduction: float = 0.0,
) -> dict:
    """Return no-MDA and MDA bladder-cancer case components.

    The UI input is an observed all-cause bladder-cancer incidence rate per
    100,000 population, consistent with a GLOBOCAN-style population rate. The
    S. haematobium-attributable share is estimated with the Levin PAF:

        PAF = P_e * (RR - 1) / (1 + P_e * (RR - 1))

    MDA is then applied only to the attributable component. Non-attributable
    bladder-cancer cases remain unchanged.
    """
    pop = max(float(at_risk_pop), 0.0)
    pe = float(np.clip(float(prevalence_fraction), 0.0, 1.0))
    rr = max(float(relative_risk), 1.001)
    rate_per_person = max(float(bladder_cancer_rate_per_100k), 0.0) / 100_000.0
    primary_share = float(np.clip(float(primary_share), 0.0, 1.0))
    effect_reduction = float(np.clip(float(effect_reduction), 0.0, 1.0))

    paf = pe * (rr - 1.0) / (1.0 + pe * (rr - 1.0))
    total_ca = pop * rate_per_person
    attributable_ca = total_ca * paf
    nonattributable_ca = max(total_ca - attributable_ca, 0.0)

    attributable_ca_mda = attributable_ca * (1.0 - effect_reduction)
    nonattributable_ca_mda = nonattributable_ca
    total_ca_mda = nonattributable_ca_mda + attributable_ca_mda
    cancer_cases_averted = attributable_ca - attributable_ca_mda

    return {
        "paf": paf,
        "total_ca": total_ca,
        "total_ca_mda": total_ca_mda,
        "attributable_ca": attributable_ca,
        "attributable_ca_mda": attributable_ca_mda,
        "nonattributable_ca": nonattributable_ca,
        "nonattributable_ca_mda": nonattributable_ca_mda,
        "ca_primary": attributable_ca * primary_share,
        "ca_meta": attributable_ca * (1.0 - primary_share),
        "ca_primary_mda": attributable_ca_mda * primary_share,
        "ca_meta_mda": attributable_ca_mda * (1.0 - primary_share),
        "cancer_cases_averted": cancer_cases_averted,
        "cancer_cases_averted_primary": cancer_cases_averted * primary_share,
        "cancer_cases_averted_metastatic": cancer_cases_averted * (1.0 - primary_share),
    }

def estimate_caseloads_haematobium(
    at_risk_pop: float,
    prev_pct: float,
    h_params: HaematobiumInputs,
    female_fraction: float = 0.50,
) -> dict:
    infected = float(at_risk_pop) * (float(prev_pct) / 100.0)
    pe = infected / max(float(at_risk_pop), 1.0)
    cancer = _bladder_cancer_case_components(
        at_risk_pop=at_risk_pop,
        prevalence_fraction=pe,
        bladder_cancer_rate_per_100k=h_params.bg_bladder_cancer_rate,
        relative_risk=h_params.bladder_cancer_rr,
        primary_share=h_params.pct_cancer_primary,
        effect_reduction=0.0,
    )

    return {
        "infected": infected,
        "hematuria": infected * h_params.pct_hematuria,
        "hydronephrosis": infected * h_params.pct_hydronephrosis,
        "fgs": infected * float(female_fraction) * h_params.pct_fgs,
        "paf": cancer["paf"],
        "bladder_cancer_total": cancer["total_ca"],
        "bladder_cancer_nonattributable": cancer["nonattributable_ca"],
        "bladder_cancer_attributable": cancer["attributable_ca"],
        "cancer_primary": cancer["ca_primary"],
        "cancer_metastatic": cancer["ca_meta"],
    }
