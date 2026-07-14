
from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_N_ITER

@dataclass(frozen=True)
class MansoniInputs:
    """Parameters for intestinal and hepatosplenic schistosomiasis."""

    n_iterations: int = DEFAULT_N_ITER
    pct_heavy: float = 0.40
    pct_heavy_std: float = 0.08
    pct_light_anemia: float = 0.18
    pct_light_anemia_std: float = 0.05
    pct_heavy_anemia: float = 0.45
    pct_heavy_anemia_std: float = 0.08
    pct_light_anemia_mild: float = 0.60
    pct_light_anemia_mild_std: float = 0.08
    pct_heavy_anemia_mild: float = 0.30
    pct_heavy_anemia_mild_std: float = 0.08
    pct_hepatomegaly: float = 0.22
    pct_hepatomegaly_std: float = 0.06
    pct_fibrosis: float = 0.48
    pct_fibrosis_std: float = 0.10
    pct_portal_htn: float = 0.32
    pct_portal_htn_std: float = 0.10
    pct_varices: float = 0.58
    pct_varices_std: float = 0.12
    dw_anemia_mild: float = 0.006
    dw_anemia_mild_std: float = 0.001
    dw_anemia_moderate: float = 0.058
    dw_anemia_moderate_std: float = 0.010
    dw_hepatomegaly: float = 0.021
    dw_hepatomegaly_std: float = 0.006
    dw_fibrosis: float = 0.021
    dw_fibrosis_std: float = 0.006
    dw_ascites: float = 0.222
    dw_ascites_std: float = 0.030
    dw_varices: float = 0.190
    dw_varices_std: float = 0.025
    prod_loss_anemia: float = 0.12
    prod_loss_anemia_std: float = 0.04
    prod_loss_hepatomegaly: float = 0.10
    prod_loss_hepatomegaly_std: float = 0.04
    prod_loss_portal_htn: float = 0.45
    prod_loss_portal_htn_std: float = 0.10
    prod_loss_varices: float = 0.65
    prod_loss_varices_std: float = 0.12
    cure_rate: float = 0.85
    cure_rate_std: float = 0.07
    egg_reduction_rate: float = 0.90
    err_std: float = 0.05
    morbidity_reduction_hepatic: float = 0.70
    mrh_std: float = 0.10
    opd_visits_anemia: float = 2.0
    opd_visits_anemia_std: float = 0.5
    opd_visits_hepatic: float = 3.5
    opd_visits_hepatic_std: float = 0.8
    ipd_days_varices: float = 7.0
    ipd_days_varices_std: float = 2.0
    pct_varices_bleed_pa: float = 0.08
    pvb_std: float = 0.03
    at_risk_pop: float = 0.0

@dataclass(frozen=True)
class HaematobiumInputs:
    """Parameters for urogenital schistosomiasis and attributable cancer."""

    n_iterations: int = DEFAULT_N_ITER
    pct_hematuria: float = 0.62
    pct_hematuria_std: float = 0.08
    pct_hydronephrosis: float = 0.15
    pct_hydronephrosis_std: float = 0.05
    pct_fgs: float = 0.75
    pct_fgs_std: float = 0.08
    bladder_cancer_rr: float = 4.20
    bladder_cancer_rr_std: float = 1.00
    pct_cancer_primary: float = 0.72
    pct_cancer_primary_std: float = 0.08
    pct_cancer_metastatic: float = 0.28
    pct_cancer_metastatic_std: float = 0.08
    bg_bladder_cancer_rate: float = 3.5
    bg_bladder_cancer_rate_std: float = 0.8
    cancer_survival_primary: float = 5.0
    cancer_survival_primary_std: float = 1.0
    cancer_survival_meta: float = 1.5
    cancer_survival_meta_std: float = 0.5
    cfr_primary: float = 0.35
    cfr_primary_std: float = 0.08
    cfr_meta: float = 0.85
    cfr_meta_std: float = 0.06
    mean_age_cancer: float = 55.0
    mean_age_cancer_std: float = 5.0
    dw_hematuria: float = 0.020
    dw_hematuria_std: float = 0.005
    dw_hydronephrosis: float = 0.149
    dw_hydronephrosis_std: float = 0.020
    dw_fgs: float = 0.048
    dw_fgs_std: float = 0.012
    dw_cancer_primary: float = 0.288
    dw_cancer_primary_std: float = 0.035
    dw_cancer_metastatic: float = 0.540
    dw_cancer_metastatic_std: float = 0.050
    prod_loss_hematuria: float = 0.08
    prod_loss_hematuria_std: float = 0.03
    prod_loss_hydronephrosis: float = 0.25
    prod_loss_hydronephrosis_std: float = 0.08
    prod_loss_fgs: float = 0.12
    prod_loss_fgs_std: float = 0.04
    prod_loss_cancer: float = 0.75
    prod_loss_cancer_std: float = 0.10
    cure_rate: float = 0.87
    cure_rate_std: float = 0.06
    egg_reduction_rate: float = 0.91
    err_std: float = 0.04
    morbidity_reduction_urinary: float = 0.75
    mru_std: float = 0.10
    cancer_reduction_mda: float = 0.30
    crm_std: float = 0.12
    opd_visits_hematuria: float = 2.5
    opd_visits_hematuria_std: float = 0.6
    opd_visits_hydronephrosis: float = 4.0
    opd_visits_hydronephrosis_std: float = 1.0
    ipd_days_hydronephrosis: float = 5.0
    ipd_days_hydronephrosis_std: float = 1.5
    opd_visits_cancer: float = 12.0
    opd_visits_cancer_std: float = 3.0
    ipd_days_cancer: float = 14.0
    ipd_days_cancer_std: float = 4.0
    at_risk_pop: float = 0.0
    female_fraction: float = 0.50
