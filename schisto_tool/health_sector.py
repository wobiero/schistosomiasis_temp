
from __future__ import annotations

import numpy as np
import pandas as pd

def extract_mean_utilization(psa_df: pd.DataFrame, species: str = "mansoni") -> dict:
    """Extract per-infected utilization from PSA output."""
    defaults = {
        "opd_visits_per_infected": 2.5,
        "ipd_days_per_infected": 0.1,
    }
    if psa_df is None or psa_df.empty:
        return defaults
    required = {"infected", "opd_total", "ipd_total"}
    if not required.issubset(psa_df.columns):
        return defaults
    
    infected = pd.to_numeric(psa_df["infected"], errors="coerce").mean()
    if not np.isfinite(infected) or infected <= 0:
        return defaults
    
    opd_total = pd.to_numeric(psa_df["opd_total"], errors="coerce").mean()
    ipd_total = pd.to_numeric(psa_df["ipd_total"], errors="coerce").mean()
    if not np.isfinite(opd_total) or not np.isfinite(ipd_total):
        return defaults
    
    return {
        "opd_visits_per_infected": opd_total / infected,
        "ipd_days_per_infected": ipd_total / infected,
    }

def project_health_sector_costs(
    caseload_proj_df: pd.DataFrame,
    psa_df: pd.DataFrame,
    opd_unit_cost: float,
    ipd_unit_cost: float,
    discount_rate: float = 0.03,
    species: str = "mansoni",
) -> pd.DataFrame:
    """Project annual health sector costs by applying utilization rates to time-varying caseloads."""
    util = extract_mean_utilization(psa_df, species)
    
    opd_per_inf = util["opd_visits_per_infected"]
    ipd_per_inf = util["ipd_days_per_infected"]
    
    rows = []
    for year_step, (_, row) in enumerate(caseload_proj_df.iterrows()):
        year = row["Year"]
        infected = row["infected"]
        
        opd_visits = infected * opd_per_inf
        ipd_days = infected * ipd_per_inf
        
        hs_cost_undis = (opd_visits * float(opd_unit_cost) + 
                         ipd_days * float(ipd_unit_cost))
        
        disc_factor = (1.0 + float(discount_rate)) ** year_step
        hs_cost_dis = hs_cost_undis / disc_factor
        
        rows.append({
            "Year": year,
            "Infected": infected,
            "OPD_visits": opd_visits,
            "IPD_days": ipd_days,
            "HS_cost_undiscounted_USD": hs_cost_undis,
            "HS_cost_discounted_USD": hs_cost_dis,
        })
    
    return pd.DataFrame(rows)
