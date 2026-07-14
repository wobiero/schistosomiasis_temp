
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .cache import cache_data
from .caseloads import (
    _bladder_cancer_case_components,
    estimate_caseloads_haematobium,
    estimate_caseloads_mansoni,
    intensity_morbidity_effect,
)
from .config import DEFAULT_SEED, WORKING_DAYS_PER_YEAR
from .parameters import HaematobiumInputs, MansoniInputs
from .utils import _bounded_normal, _gamma, _positive_normal, _truncated_beta

def _single_run_mansoni(
    m: MansoniInputs,
    caseloads: dict,
    rng: np.random.Generator,
    mda_effect_multiplier: float,
) -> dict:
    ph = _truncated_beta(m.pct_heavy, m.pct_heavy_std, rng)
    pl = 1.0 - ph
    pla = _truncated_beta(m.pct_light_anemia, m.pct_light_anemia_std, rng)
    pha = _truncated_beta(m.pct_heavy_anemia, m.pct_heavy_anemia_std, rng)
    mild_light_share = _truncated_beta(m.pct_light_anemia_mild, m.pct_light_anemia_mild_std, rng)
    mild_heavy_share = _truncated_beta(m.pct_heavy_anemia_mild, m.pct_heavy_anemia_mild_std, rng)
    phep = _truncated_beta(m.pct_hepatomegaly, m.pct_hepatomegaly_std, rng)
    pfib = _truncated_beta(m.pct_fibrosis, m.pct_fibrosis_std, rng)
    ppht = _truncated_beta(m.pct_portal_htn, m.pct_portal_htn_std, rng)
    pvar = _truncated_beta(m.pct_varices, m.pct_varices_std, rng)

    dw_am = _bounded_normal(m.dw_anemia_mild, m.dw_anemia_mild_std, rng)
    dw_amod = _bounded_normal(m.dw_anemia_moderate, m.dw_anemia_moderate_std, rng)
    dw_hep = _bounded_normal(m.dw_hepatomegaly, m.dw_hepatomegaly_std, rng)
    dw_fib = _bounded_normal(m.dw_fibrosis, m.dw_fibrosis_std, rng)
    dw_asc = _bounded_normal(m.dw_ascites, m.dw_ascites_std, rng)
    dw_var = _bounded_normal(m.dw_varices, m.dw_varices_std, rng)

    pl_prod = _bounded_normal(m.prod_loss_anemia, m.prod_loss_anemia_std, rng)
    ph_prod = _bounded_normal(m.prod_loss_hepatomegaly, m.prod_loss_hepatomegaly_std, rng)
    pph_prod = _bounded_normal(m.prod_loss_portal_htn, m.prod_loss_portal_htn_std, rng)
    pv_prod = _bounded_normal(m.prod_loss_varices, m.prod_loss_varices_std, rng)

    cr = _truncated_beta(m.cure_rate, m.cure_rate_std, rng)
    err = _truncated_beta(m.egg_reduction_rate, m.err_std, rng)
    mrh = _truncated_beta(m.morbidity_reduction_hepatic, m.mrh_std, rng)

    ov_an = _positive_normal(m.opd_visits_anemia, m.opd_visits_anemia_std, rng)
    ov_hep = _positive_normal(m.opd_visits_hepatic, m.opd_visits_hepatic_std, rng)
    id_var = _positive_normal(m.ipd_days_varices, m.ipd_days_varices_std, rng)
    pvb = _truncated_beta(m.pct_varices_bleed_pa, m.pvb_std, rng)

    infected = caseloads["infected"]
    light_inf = infected * pl
    heavy_inf = infected * ph
    anemia_light = light_inf * pla
    anemia_heavy = heavy_inf * pha
    anemia = anemia_light + anemia_heavy
    hepatomeg = heavy_inf * phep
    fibrosis = hepatomeg * pfib
    portal_htn = fibrosis * ppht
    varices = portal_htn * pvar

    anemia_mild = anemia_light * mild_light_share + anemia_heavy * mild_heavy_share
    anemia_mod = max(anemia - anemia_mild, 0.0)

    daly_anemia = anemia_mild * dw_am + anemia_mod * dw_amod
    daly_hepatomeg = hepatomeg * dw_hep
    daly_fibrosis = fibrosis * dw_fib
    daly_portal = portal_htn * dw_asc
    daly_varices = varices * dw_var
    daly_total = daly_anemia + daly_hepatomeg + daly_fibrosis + daly_portal + daly_varices

    eff_anemia = float(np.clip(mda_effect_multiplier * intensity_morbidity_effect(err), 0.0, 1.0))
    eff_hepatic = float(np.clip(mda_effect_multiplier * mrh, 0.0, 1.0))

    daly_anemia_mda = daly_anemia * (1.0 - eff_anemia)
    daly_hepatomeg_mda = daly_hepatomeg * (1.0 - eff_hepatic)
    daly_fibrosis_mda = daly_fibrosis * (1.0 - eff_hepatic)
    daly_portal_mda = daly_portal * (1.0 - eff_hepatic)
    daly_varices_mda = daly_varices * (1.0 - eff_hepatic)
    daly_total_mda = (
        daly_anemia_mda
        + daly_hepatomeg_mda
        + daly_fibrosis_mda
        + daly_portal_mda
        + daly_varices_mda
    )

    work_days_lost = (
        anemia * pl_prod * WORKING_DAYS_PER_YEAR
        + hepatomeg * ph_prod * WORKING_DAYS_PER_YEAR
        + portal_htn * pph_prod * WORKING_DAYS_PER_YEAR
        + varices * pv_prod * WORKING_DAYS_PER_YEAR
    )
    work_days_lost_mda = (
        anemia * (1.0 - eff_anemia) * pl_prod * WORKING_DAYS_PER_YEAR
        + hepatomeg * (1.0 - eff_hepatic) * ph_prod * WORKING_DAYS_PER_YEAR
        + portal_htn * (1.0 - eff_hepatic) * pph_prod * WORKING_DAYS_PER_YEAR
        + varices * (1.0 - eff_hepatic) * pv_prod * WORKING_DAYS_PER_YEAR
    )

    anemia_mda = anemia * (1.0 - eff_anemia)
    hepatomeg_mda = hepatomeg * (1.0 - eff_hepatic)
    fibrosis_mda = fibrosis * (1.0 - eff_hepatic)
    portal_htn_mda = portal_htn * (1.0 - eff_hepatic)
    varices_mda = varices * (1.0 - eff_hepatic)

    opd_anemia = anemia * ov_an
    opd_anemia_mda = anemia_mda * ov_an
    opd_hepatomeg = hepatomeg * ov_hep
    opd_hepatomeg_mda = hepatomeg_mda * ov_hep
    opd_fibrosis = fibrosis * ov_hep
    opd_fibrosis_mda = fibrosis_mda * ov_hep
    opd_portal_htn = portal_htn * ov_hep
    opd_portal_htn_mda = portal_htn_mda * ov_hep
    ipd_varices = varices * pvb * id_var
    ipd_varices_mda = varices_mda * pvb * id_var
    opd_hepatic = opd_hepatomeg + opd_fibrosis + opd_portal_htn
    opd_hepatic_mda = opd_hepatomeg_mda + opd_fibrosis_mda + opd_portal_htn_mda
    opd_total = opd_anemia + opd_hepatic
    opd_total_mda = opd_anemia_mda + opd_hepatic_mda
    ipd_total = ipd_varices
    ipd_total_mda = ipd_varices_mda

    return {
        "infected": infected,
        "anemia": anemia,
        "anemia_mda": anemia_mda,
        "hepatomeg": hepatomeg,
        "hepatomeg_mda": hepatomeg_mda,
        "fibrosis": fibrosis,
        "fibrosis_mda": fibrosis_mda,
        "portal_htn": portal_htn,
        "portal_htn_mda": portal_htn_mda,
        "varices": varices,
        "varices_mda": varices_mda,
        "daly_anemia": daly_anemia,
        "daly_hepatomeg": daly_hepatomeg,
        "daly_fibrosis": daly_fibrosis,
        "daly_portal": daly_portal,
        "daly_varices": daly_varices,
        "daly_total": daly_total,
        "daly_anemia_mda": daly_anemia_mda,
        "daly_hepatomeg_mda": daly_hepatomeg_mda,
        "daly_fibrosis_mda": daly_fibrosis_mda,
        "daly_portal_mda": daly_portal_mda,
        "daly_varices_mda": daly_varices_mda,
        "daly_total_mda": daly_total_mda,
        "work_days_lost": work_days_lost,
        "work_days_lost_mda": work_days_lost_mda,
        "opd_anemia": opd_anemia,
        "opd_anemia_mda": opd_anemia_mda,
        "opd_hepatomeg": opd_hepatomeg,
        "opd_hepatomeg_mda": opd_hepatomeg_mda,
        "opd_fibrosis": opd_fibrosis,
        "opd_fibrosis_mda": opd_fibrosis_mda,
        "opd_portal_htn": opd_portal_htn,
        "opd_portal_htn_mda": opd_portal_htn_mda,
        "ipd_varices": ipd_varices,
        "ipd_varices_mda": ipd_varices_mda,
        "opd_total": opd_total,
        "opd_total_mda": opd_total_mda,
        "ipd_total": ipd_total,
        "ipd_total_mda": ipd_total_mda,
        "cure_rate": cr,
        "egg_reduction": err,
        "egg_reduction_effect": err,
        "light_anemia_mild_share": mild_light_share,
        "heavy_anemia_mild_share": mild_heavy_share,
        "morbidity_reduction": mrh,
        "effective_anemia_reduction": eff_anemia,
        "effective_hepatic_reduction": eff_hepatic,
        "mda_effect_multiplier": mda_effect_multiplier,
    }

def _single_run_haematobium(
    h: HaematobiumInputs,
    caseloads: dict,
    life_exp: float,
    rng: np.random.Generator,
    mda_effect_multiplier: float,
) -> dict:
    ph_hem = _truncated_beta(h.pct_hematuria, h.pct_hematuria_std, rng)
    ph_hyd = _truncated_beta(h.pct_hydronephrosis, h.pct_hydronephrosis_std, rng)
    ph_fgs = _truncated_beta(h.pct_fgs, h.pct_fgs_std, rng)
    rr = max(_gamma(h.bladder_cancer_rr, h.bladder_cancer_rr_std, rng), 1.001)
    bg_ca = _positive_normal(h.bg_bladder_cancer_rate, h.bg_bladder_cancer_rate_std, rng)
    pcp = _truncated_beta(h.pct_cancer_primary, h.pct_cancer_primary_std, rng)
    survival_primary = _positive_normal(h.cancer_survival_primary, h.cancer_survival_primary_std, rng)
    survival_meta = _positive_normal(h.cancer_survival_meta, h.cancer_survival_meta_std, rng)
    cfr_primary = _truncated_beta(h.cfr_primary, h.cfr_primary_std, rng)
    cfr_meta = _truncated_beta(h.cfr_meta, h.cfr_meta_std, rng)
    mean_age_cancer = _positive_normal(h.mean_age_cancer, h.mean_age_cancer_std, rng)

    dw_hem = _bounded_normal(h.dw_hematuria, h.dw_hematuria_std, rng)
    dw_hyd = _bounded_normal(h.dw_hydronephrosis, h.dw_hydronephrosis_std, rng)
    dw_fgs = _bounded_normal(h.dw_fgs, h.dw_fgs_std, rng)
    dw_cap = _bounded_normal(h.dw_cancer_primary, h.dw_cancer_primary_std, rng)
    dw_cam = _bounded_normal(h.dw_cancer_metastatic, h.dw_cancer_metastatic_std, rng)

    pl_hem = _bounded_normal(h.prod_loss_hematuria, h.prod_loss_hematuria_std, rng)
    pl_hyd = _bounded_normal(h.prod_loss_hydronephrosis, h.prod_loss_hydronephrosis_std, rng)
    pl_fgs = _bounded_normal(h.prod_loss_fgs, h.prod_loss_fgs_std, rng)
    pl_ca = _bounded_normal(h.prod_loss_cancer, h.prod_loss_cancer_std, rng)

    cr = _truncated_beta(h.cure_rate, h.cure_rate_std, rng)
    err = _truncated_beta(h.egg_reduction_rate, h.err_std, rng)
    mru = _truncated_beta(h.morbidity_reduction_urinary, h.mru_std, rng)
    crm = _truncated_beta(h.cancer_reduction_mda, h.crm_std, rng)

    ov_hem = _positive_normal(h.opd_visits_hematuria, h.opd_visits_hematuria_std, rng)
    ov_hyd = _positive_normal(h.opd_visits_hydronephrosis, h.opd_visits_hydronephrosis_std, rng)
    id_hyd = _positive_normal(h.ipd_days_hydronephrosis, h.ipd_days_hydronephrosis_std, rng)
    ov_ca = _positive_normal(h.opd_visits_cancer, h.opd_visits_cancer_std, rng)
    id_ca = _positive_normal(h.ipd_days_cancer, h.ipd_days_cancer_std, rng)

    infected = caseloads["infected"]
    pe = infected / max(h.at_risk_pop, 1.0)
    hematuria = infected * ph_hem
    hydronepr = infected * ph_hyd
    fgs = infected * h.female_fraction * ph_fgs

    # Hematuria is tied to infection clearance; hydronephrosis, FGS, and
    # attributable cancer risk use ERR as the intensity-mediated antiparasitic
    # component. Cure rate and ERR are not averaged.
    eff_hematuria = float(np.clip(mda_effect_multiplier * cr, 0.0, 1.0))
    eff_urinary = float(np.clip(mda_effect_multiplier * intensity_morbidity_effect(err) * mru, 0.0, 1.0))
    eff_cancer = float(np.clip(mda_effect_multiplier * intensity_morbidity_effect(err) * crm, 0.0, 1.0))

    cancer = _bladder_cancer_case_components(
        at_risk_pop=h.at_risk_pop,
        prevalence_fraction=pe,
        bladder_cancer_rate_per_100k=bg_ca,
        relative_risk=rr,
        primary_share=pcp,
        effect_reduction=eff_cancer,
    )
    paf = cancer["paf"]
    total_ca = cancer["total_ca"]
    total_ca_mda = cancer["total_ca_mda"]
    attributable_ca = cancer["attributable_ca"]
    attributable_ca_mda = cancer["attributable_ca_mda"]
    nonattributable_ca = cancer["nonattributable_ca"]
    ca_primary = cancer["ca_primary"]
    ca_meta = cancer["ca_meta"]
    ca_primary_mda = cancer["ca_primary_mda"]
    ca_meta_mda = cancer["ca_meta_mda"]
    cancer_cases_averted = cancer["cancer_cases_averted"]

    daly_hem = hematuria * dw_hem
    daly_hyd = hydronepr * dw_hyd
    daly_fgs = fgs * dw_fgs

    daly_cancer_yld = ca_primary * dw_cap * survival_primary + ca_meta * dw_cam * survival_meta
    daly_cancer_yld_mda = ca_primary_mda * dw_cap * survival_primary + ca_meta_mda * dw_cam * survival_meta

    remaining_life_expectancy = max(float(life_exp) - mean_age_cancer, 0.0)
    yll_primary = ca_primary * cfr_primary * remaining_life_expectancy
    yll_meta = ca_meta * cfr_meta * remaining_life_expectancy
    yll_primary_mda = ca_primary_mda * cfr_primary * remaining_life_expectancy
    yll_meta_mda = ca_meta_mda * cfr_meta * remaining_life_expectancy

    daly_cancer_yll = yll_primary + yll_meta
    daly_cancer_yll_mda = yll_primary_mda + yll_meta_mda
    daly_cancer = daly_cancer_yld + daly_cancer_yll
    daly_total = daly_hem + daly_hyd + daly_fgs + daly_cancer

    daly_hem_mda = daly_hem * (1.0 - eff_hematuria)
    daly_hyd_mda = daly_hyd * (1.0 - eff_urinary)
    daly_fgs_mda = daly_fgs * (1.0 - eff_urinary)
    daly_cancer_mda = daly_cancer_yld_mda + daly_cancer_yll_mda
    daly_total_mda = daly_hem_mda + daly_hyd_mda + daly_fgs_mda + daly_cancer_mda

    work_days_lost = (
        hematuria * pl_hem * WORKING_DAYS_PER_YEAR
        + hydronepr * pl_hyd * WORKING_DAYS_PER_YEAR
        + fgs * pl_fgs * WORKING_DAYS_PER_YEAR
        + attributable_ca * pl_ca * WORKING_DAYS_PER_YEAR
    )
    work_days_lost_mda = (
        hematuria * (1.0 - eff_hematuria) * pl_hem * WORKING_DAYS_PER_YEAR
        + hydronepr * (1.0 - eff_urinary) * pl_hyd * WORKING_DAYS_PER_YEAR
        + fgs * (1.0 - eff_urinary) * pl_fgs * WORKING_DAYS_PER_YEAR
        + attributable_ca_mda * pl_ca * WORKING_DAYS_PER_YEAR
    )

    hematuria_mda = hematuria * (1.0 - eff_hematuria)
    hydronephrosis_mda = hydronepr * (1.0 - eff_urinary)
    fgs_mda = fgs * (1.0 - eff_urinary)

    opd_hematuria = hematuria * ov_hem
    opd_hematuria_mda = hematuria_mda * ov_hem
    opd_hydronephrosis = hydronepr * ov_hyd
    opd_hydronephrosis_mda = hydronephrosis_mda * ov_hyd
    ipd_hydronephrosis = hydronepr * id_hyd
    ipd_hydronephrosis_mda = hydronephrosis_mda * id_hyd
    opd_cancer = attributable_ca * ov_ca
    opd_cancer_mda = attributable_ca_mda * ov_ca
    ipd_cancer = attributable_ca * id_ca
    ipd_cancer_mda = attributable_ca_mda * id_ca

    opd_total = opd_hematuria + opd_hydronephrosis + opd_cancer
    ipd_total = ipd_hydronephrosis + ipd_cancer
    opd_total_mda = opd_hematuria_mda + opd_hydronephrosis_mda + opd_cancer_mda
    ipd_total_mda = ipd_hydronephrosis_mda + ipd_cancer_mda

    return {
        "infected": infected,
        "hematuria": hematuria,
        "hematuria_mda": hematuria_mda,
        "hydronephrosis": hydronepr,
        "hydronephrosis_mda": hydronephrosis_mda,
        "fgs": fgs,
        "fgs_mda": fgs_mda,
        "ca_primary": ca_primary,
        "ca_meta": ca_meta,
        "ca_primary_mda": ca_primary_mda,
        "ca_meta_mda": ca_meta_mda,
        "total_ca": total_ca,
        "total_ca_mda": total_ca_mda,
        "nonattributable_ca": nonattributable_ca,
        "attributable_ca": attributable_ca,
        "attributable_ca_mda": attributable_ca_mda,
        "cancer_cases_averted": cancer_cases_averted,
        "cancer_cases_averted_primary": cancer["cancer_cases_averted_primary"],
        "cancer_cases_averted_metastatic": cancer["cancer_cases_averted_metastatic"],
        "paf": paf,
        "daly_hem": daly_hem,
        "daly_hyd": daly_hyd,
        "daly_fgs": daly_fgs,
        "daly_cancer_yld": daly_cancer_yld,
        "daly_cancer_yll": daly_cancer_yll,
        "daly_cancer_yld_mda": daly_cancer_yld_mda,
        "daly_cancer_yll_mda": daly_cancer_yll_mda,
        "daly_cancer": daly_cancer,
        "daly_total": daly_total,
        "daly_hem_mda": daly_hem_mda,
        "daly_hyd_mda": daly_hyd_mda,
        "daly_fgs_mda": daly_fgs_mda,
        "daly_cancer_mda": daly_cancer_mda,
        "daly_total_mda": daly_total_mda,
        "work_days_lost": work_days_lost,
        "work_days_lost_mda": work_days_lost_mda,
        "opd_hematuria": opd_hematuria,
        "opd_hematuria_mda": opd_hematuria_mda,
        "opd_hydronephrosis": opd_hydronephrosis,
        "opd_hydronephrosis_mda": opd_hydronephrosis_mda,
        "ipd_hydronephrosis": ipd_hydronephrosis,
        "ipd_hydronephrosis_mda": ipd_hydronephrosis_mda,
        "opd_cancer": opd_cancer,
        "opd_cancer_mda": opd_cancer_mda,
        "ipd_cancer": ipd_cancer,
        "ipd_cancer_mda": ipd_cancer_mda,
        "opd_total": opd_total,
        "opd_total_mda": opd_total_mda,
        "ipd_total": ipd_total,
        "ipd_total_mda": ipd_total_mda,
        "cure_rate": cr,
        "egg_reduction": err,
        "egg_reduction_effect": err,
        "morbidity_reduction": mru,
        "cancer_reduction": crm,
        "effective_hematuria_reduction": eff_hematuria,
        "effective_urinary_reduction": eff_urinary,
        "effective_cancer_reduction": eff_cancer,
        "cfr_primary": cfr_primary,
        "cfr_meta": cfr_meta,
        "mda_effect_multiplier": mda_effect_multiplier,
        "yll_primary": yll_primary,
        "yll_meta": yll_meta,
        "yll_primary_mda": yll_primary_mda,
        "yll_meta_mda": yll_meta_mda,
        "cancer_survival_primary": survival_primary,
        "cancer_survival_meta": survival_meta,
        "mean_age_cancer": mean_age_cancer,
    }

@cache_data(show_spinner=False)
def run_monte_carlo_mansoni(
    n: int,
    at_risk_pop: float,
    prev_pct: float,
    params: MansoniInputs,
    mda_effect_multiplier: float,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    m = replace(
        params,
        n_iterations=int(n),
        at_risk_pop=float(at_risk_pop),
    )
    cl = estimate_caseloads_mansoni(at_risk_pop, prev_pct, m)
    rows = [_single_run_mansoni(m, cl, rng, mda_effect_multiplier) for _ in range(int(n))]
    return pd.DataFrame(rows)

@cache_data(show_spinner=False)
def run_monte_carlo_haematobium(
    n: int,
    at_risk_pop: float,
    prev_pct: float,
    female_fraction: float,
    life_exp: float,
    params: HaematobiumInputs,
    mda_effect_multiplier: float,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    h = replace(
        params,
        n_iterations=int(n),
        at_risk_pop=float(at_risk_pop),
        female_fraction=float(female_fraction),
    )
    cl = estimate_caseloads_haematobium(at_risk_pop, prev_pct, h, female_fraction)
    rows = [
        _single_run_haematobium(h, cl, life_exp, rng, mda_effect_multiplier)
        for _ in range(int(n))
    ]
    return pd.DataFrame(rows)
