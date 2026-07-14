
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .cache import cache_data
from .config import (
    CI_LOWER_Q,
    CI_UPPER_Q,
    DEFAULT_HOURS_PER_WORKDAY,
    DEFAULT_SEED,
    WORKING_DAYS_PER_YEAR,
)
from .utils import _ci_bounds, _discount_array, _same_length_sum, _wilson_ci

DALY_AVERTED_PAIRS = (
    ("daly_total", "daly_total_mda", "dalys_averted"),
    ("daly_anemia", "daly_anemia_mda", "daly_anemia_averted"),
    ("daly_hepatomeg", "daly_hepatomeg_mda", "daly_hepatomeg_averted"),
    ("daly_fibrosis", "daly_fibrosis_mda", "daly_fibrosis_averted"),
    ("daly_portal", "daly_portal_mda", "daly_portal_averted"),
    ("daly_varices", "daly_varices_mda", "daly_varices_averted"),
    ("daly_hem", "daly_hem_mda", "daly_hem_averted"),
    ("daly_hyd", "daly_hyd_mda", "daly_hyd_averted"),
    ("daly_fgs", "daly_fgs_mda", "daly_fgs_averted"),
    ("daly_cancer", "daly_cancer_mda", "daly_cancer_averted"),
)

MORBIDITY_BUDGET_STATES = (
    {
        "species": "S. mansoni / japonicum",
        "state_key": "mansoni_anemia",
        "health_state": "Anemia",
        "case_no": "anemia",
        "case_mda": "anemia_mda",
        "opd_no": "opd_anemia",
        "opd_mda": "opd_anemia_mda",
        "ipd_no": None,
        "ipd_mda": None,
    },
    {
        "species": "S. mansoni / japonicum",
        "state_key": "mansoni_hepatomegaly",
        "health_state": "Hepatomegaly",
        "case_no": "hepatomeg",
        "case_mda": "hepatomeg_mda",
        "opd_no": "opd_hepatomeg",
        "opd_mda": "opd_hepatomeg_mda",
        "ipd_no": None,
        "ipd_mda": None,
    },
    {
        "species": "S. mansoni / japonicum",
        "state_key": "mansoni_fibrosis",
        "health_state": "Periportal fibrosis",
        "case_no": "fibrosis",
        "case_mda": "fibrosis_mda",
        "opd_no": "opd_fibrosis",
        "opd_mda": "opd_fibrosis_mda",
        "ipd_no": None,
        "ipd_mda": None,
    },
    {
        "species": "S. mansoni / japonicum",
        "state_key": "mansoni_portal_htn",
        "health_state": "Portal hypertension",
        "case_no": "portal_htn",
        "case_mda": "portal_htn_mda",
        "opd_no": "opd_portal_htn",
        "opd_mda": "opd_portal_htn_mda",
        "ipd_no": None,
        "ipd_mda": None,
    },
    {
        "species": "S. mansoni / japonicum",
        "state_key": "mansoni_varices",
        "health_state": "Esophageal varices",
        "case_no": "varices",
        "case_mda": "varices_mda",
        "opd_no": None,
        "opd_mda": None,
        "ipd_no": "ipd_varices",
        "ipd_mda": "ipd_varices_mda",
    },
    {
        "species": "S. haematobium",
        "state_key": "haematobium_hematuria",
        "health_state": "Hematuria",
        "case_no": "hematuria",
        "case_mda": "hematuria_mda",
        "opd_no": "opd_hematuria",
        "opd_mda": "opd_hematuria_mda",
        "ipd_no": None,
        "ipd_mda": None,
    },
    {
        "species": "S. haematobium",
        "state_key": "haematobium_hydronephrosis",
        "health_state": "Hydronephrosis",
        "case_no": "hydronephrosis",
        "case_mda": "hydronephrosis_mda",
        "opd_no": "opd_hydronephrosis",
        "opd_mda": "opd_hydronephrosis_mda",
        "ipd_no": "ipd_hydronephrosis",
        "ipd_mda": "ipd_hydronephrosis_mda",
    },
    {
        "species": "S. haematobium",
        "state_key": "haematobium_fgs",
        "health_state": "Female genital schistosomiasis",
        "case_no": "fgs",
        "case_mda": "fgs_mda",
        "opd_no": None,
        "opd_mda": None,
        "ipd_no": None,
        "ipd_mda": None,
    },
    {
        "species": "S. haematobium",
        "state_key": "haematobium_bladder_cancer",
        "health_state": "Attributable bladder cancer",
        "case_no": "attributable_ca",
        "case_mda": "attributable_ca_mda",
        "opd_no": "opd_cancer",
        "opd_mda": "opd_cancer_mda",
        "ipd_no": "ipd_cancer",
        "ipd_mda": "ipd_cancer_mda",
    },
)

DEFAULT_HEALTH_STATE_COSTS_USD = {
    state["state_key"]: 0.0 for state in MORBIDITY_BUDGET_STATES
}


def _numeric_draws(df: pd.DataFrame, col: str | None, template: np.ndarray | None = None) -> np.ndarray:
    """Return a numeric draw array, or zeros matching template when absent."""
    if col is None or df is None or df.empty or col not in df.columns:
        if template is not None:
            return np.zeros_like(template, dtype=float)
        return np.zeros(len(df) if df is not None else 0, dtype=float)
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def morbidity_budget_state_draws(
    sim_dfs: list[pd.DataFrame],
    opd_cost: float,
    ipd_cost: float,
    health_state_costs: Optional[dict[str, float]] = None,
    opd_staff_minutes: float = 15.0,
    ipd_staff_minutes: float = 120.0,
    staff_hourly_cost: float = 0.0,
    include_staff_time_value: bool = False,
) -> list[dict]:
    """Build draw-level budget offsets by morbidity state.

    Clinical-management offsets are estimated state by state from avoided OPD
    visits, avoided IPD days, and optional additional direct management cost per
    case averted. Staff time is reported separately in hours and may be included
    in the net budget impact only when include_staff_time_value is True.
    """
    health_state_costs = dict(health_state_costs or {})
    valid_dfs = [df for df in sim_dfs if df is not None and not df.empty]
    rows: list[dict] = []
    if not valid_dfs:
        return rows

    for df in valid_dfs:
        for state in MORBIDITY_BUDGET_STATES:
            case_no_col = state["case_no"]
            case_mda_col = state["case_mda"]
            if case_no_col not in df.columns or case_mda_col not in df.columns:
                continue

            cases_no = _numeric_draws(df, case_no_col)
            cases_mda = _numeric_draws(df, case_mda_col, cases_no)
            cases_averted = np.maximum(cases_no - cases_mda, 0.0)

            opd_no = _numeric_draws(df, state.get("opd_no"), cases_no)
            opd_mda = _numeric_draws(df, state.get("opd_mda"), cases_no)
            ipd_no = _numeric_draws(df, state.get("ipd_no"), cases_no)
            ipd_mda = _numeric_draws(df, state.get("ipd_mda"), cases_no)
            opd_averted = np.maximum(opd_no - opd_mda, 0.0)
            ipd_averted = np.maximum(ipd_no - ipd_mda, 0.0)

            extra_cost_per_case = float(health_state_costs.get(state["state_key"], 0.0) or 0.0)
            visit_day_offset = opd_averted * float(opd_cost) + ipd_averted * float(ipd_cost)
            additional_case_offset = cases_averted * extra_cost_per_case
            clinical_offset = visit_day_offset + additional_case_offset

            staff_hours_saved = (
                opd_averted * max(float(opd_staff_minutes), 0.0) / 60.0
                + ipd_averted * max(float(ipd_staff_minutes), 0.0) / 60.0
            )
            staff_time_value = staff_hours_saved * max(float(staff_hourly_cost), 0.0)
            staff_time_value_included = staff_time_value if include_staff_time_value else np.zeros_like(staff_time_value)
            budget_offset = clinical_offset + staff_time_value_included

            rows.append(
                {
                    "species": state["species"],
                    "state_key": state["state_key"],
                    "health_state": state["health_state"],
                    "extra_cost_per_case_usd": extra_cost_per_case,
                    "cases_averted": cases_averted,
                    "opd_visits_averted": opd_averted,
                    "ipd_days_averted": ipd_averted,
                    "visit_day_offset_usd": visit_day_offset,
                    "additional_case_offset_usd": additional_case_offset,
                    "clinical_offset_usd": clinical_offset,
                    "staff_hours_saved": staff_hours_saved,
                    "staff_time_value_usd": staff_time_value,
                    "staff_time_value_included_usd": staff_time_value_included,
                    "budget_offset_usd": budget_offset,
                }
            )
    return rows


def combined_morbidity_budget_draws(
    sim_dfs: list[pd.DataFrame],
    opd_cost: float,
    ipd_cost: float,
    health_state_costs: Optional[dict[str, float]] = None,
    opd_staff_minutes: float = 15.0,
    ipd_staff_minutes: float = 120.0,
    staff_hourly_cost: float = 0.0,
    include_staff_time_value: bool = False,
) -> dict:
    """Combine morbidity-state budget-offset draws across selected species."""
    state_rows = morbidity_budget_state_draws(
        sim_dfs=sim_dfs,
        opd_cost=opd_cost,
        ipd_cost=ipd_cost,
        health_state_costs=health_state_costs,
        opd_staff_minutes=opd_staff_minutes,
        ipd_staff_minutes=ipd_staff_minutes,
        staff_hourly_cost=staff_hourly_cost,
        include_staff_time_value=include_staff_time_value,
    )
    clinical = _same_length_sum([row["clinical_offset_usd"] for row in state_rows])
    staff_hours = _same_length_sum([row["staff_hours_saved"] for row in state_rows])
    staff_value = _same_length_sum([row["staff_time_value_usd"] for row in state_rows])
    staff_included = _same_length_sum([row["staff_time_value_included_usd"] for row in state_rows])
    if clinical.size:
        budget_offset = clinical + (staff_included[: clinical.size] if staff_included.size else 0.0)
    elif staff_included.size:
        budget_offset = staff_included
    else:
        budget_offset = np.array([], dtype=float)
    return {
        "state_rows": state_rows,
        "clinical_budget_offset_pa": clinical,
        "staff_hours_saved_pa": staff_hours,
        "staff_time_value_pa": staff_value,
        "staff_time_value_included_pa": staff_included,
        "budget_offset_pa": budget_offset,
    }


def morbidity_budget_summary_table(
    state_rows: list[dict],
    horizon: int,
    disc_rate: float,
    include_staff_time_value: bool = False,
) -> pd.DataFrame:
    """Summarize annual and discounted-horizon budget offsets by morbidity state."""
    if not state_rows:
        return pd.DataFrame()
    horizon = int(horizon)
    disc_rate = float(disc_rate)
    if horizon <= 0:
        pv_factor = 0.0
    else:
        pv_factor = sum(1.0 / ((1.0 + disc_rate) ** yr) for yr in range(1, horizon + 1))

    out_rows = []
    for row in state_rows:
        cases_mean, cases_lo, cases_hi = _ci_bounds(row["cases_averted"])
        opd_mean, opd_lo, opd_hi = _ci_bounds(row["opd_visits_averted"])
        ipd_mean, ipd_lo, ipd_hi = _ci_bounds(row["ipd_days_averted"])
        visit_mean, visit_lo, visit_hi = _ci_bounds(row["visit_day_offset_usd"])
        add_mean, add_lo, add_hi = _ci_bounds(row["additional_case_offset_usd"])
        clinical_mean, clinical_lo, clinical_hi = _ci_bounds(row["clinical_offset_usd"])
        staff_hours_mean, staff_hours_lo, staff_hours_hi = _ci_bounds(row["staff_hours_saved"])
        staff_value_mean, staff_value_lo, staff_value_hi = _ci_bounds(row["staff_time_value_usd"])
        included_mean, included_lo, included_hi = _ci_bounds(row["staff_time_value_included_usd"])
        budget_mean, budget_lo, budget_hi = _ci_bounds(row["budget_offset_usd"])
        pv_draws = row["budget_offset_usd"] * pv_factor
        pv_mean, pv_lo, pv_hi = _ci_bounds(pv_draws)

        out_rows.append(
            {
                "Species": row["species"],
                "Health state": row["health_state"],
                "Cases averted p.a.": cases_mean,
                "Cases averted p.a. 95% CI lower": cases_lo,
                "Cases averted p.a. 95% CI upper": cases_hi,
                "OPD visits averted p.a.": opd_mean,
                "OPD visits averted p.a. 95% CI lower": opd_lo,
                "OPD visits averted p.a. 95% CI upper": opd_hi,
                "IPD days averted p.a.": ipd_mean,
                "IPD days averted p.a. 95% CI lower": ipd_lo,
                "IPD days averted p.a. 95% CI upper": ipd_hi,
                "OPD/IPD offset p.a. USD": visit_mean,
                "OPD/IPD offset p.a. USD 95% CI lower": visit_lo,
                "OPD/IPD offset p.a. USD 95% CI upper": visit_hi,
                "Additional management offset p.a. USD": add_mean,
                "Additional management offset p.a. USD 95% CI lower": add_lo,
                "Additional management offset p.a. USD 95% CI upper": add_hi,
                "Clinical-management offset p.a. USD": clinical_mean,
                "Clinical-management offset p.a. USD 95% CI lower": clinical_lo,
                "Clinical-management offset p.a. USD 95% CI upper": clinical_hi,
                "Staff time saved p.a. hours": staff_hours_mean,
                "Staff time saved p.a. hours 95% CI lower": staff_hours_lo,
                "Staff time saved p.a. hours 95% CI upper": staff_hours_hi,
                "Staff time value p.a. USD": staff_value_mean,
                "Staff time value p.a. USD 95% CI lower": staff_value_lo,
                "Staff time value p.a. USD 95% CI upper": staff_value_hi,
                "Staff time value included in budget p.a. USD": included_mean,
                "Staff time value included in budget p.a. USD 95% CI lower": included_lo,
                "Staff time value included in budget p.a. USD 95% CI upper": included_hi,
                "Budget offset included p.a. USD": budget_mean,
                "Budget offset included p.a. USD 95% CI lower": budget_lo,
                "Budget offset included p.a. USD 95% CI upper": budget_hi,
                f"Discounted {horizon}-year budget offset USD": pv_mean,
                f"Discounted {horizon}-year budget offset USD 95% CI lower": pv_lo,
                f"Discounted {horizon}-year budget offset USD 95% CI upper": pv_hi,
                "Additional management cost per case averted USD": row["extra_cost_per_case_usd"],
                "Staff time value included in net budget impact": bool(include_staff_time_value),
            }
        )
    return pd.DataFrame(out_rows)

def add_daly_averted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with draw-level DALYs-averted columns."""
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()

    out = df.copy()
    for no_mda_col, mda_col, out_col in DALY_AVERTED_PAIRS:
        if no_mda_col in out.columns and mda_col in out.columns:
            out[out_col] = (
                pd.to_numeric(out[no_mda_col], errors="coerce")
                - pd.to_numeric(out[mda_col], errors="coerce")
            )

    if "dalys_averted" in out.columns:
        out["daly_averted"] = out["dalys_averted"]
    return out

def add_cost_effectiveness_columns(
    df: pd.DataFrame,
    annual_prog_cost: float,
    opd_cost: float = 0.0,
    ipd_cost: float = 0.0,
) -> pd.DataFrame:
    """Return a draw-level CE dataframe with effects and incremental costs."""
    out = add_daly_averted_columns(df)
    annual_prog_cost = float(annual_prog_cost)
    out["annual_program_cost_usd"] = annual_prog_cost

    utilization_cols = {"opd_total", "opd_total_mda", "ipd_total", "ipd_total_mda"}
    if utilization_cols.issubset(out.columns):
        opd_no = pd.to_numeric(out["opd_total"], errors="coerce")
        opd_mda = pd.to_numeric(out["opd_total_mda"], errors="coerce")
        ipd_no = pd.to_numeric(out["ipd_total"], errors="coerce")
        ipd_mda = pd.to_numeric(out["ipd_total_mda"], errors="coerce")
        out["health_sector_savings_usd"] = (
            (opd_no - opd_mda) * float(opd_cost)
            + (ipd_no - ipd_mda) * float(ipd_cost)
        )
    else:
        out["health_sector_savings_usd"] = 0.0

    out["program_cost_minus_savings"] = (
        out["annual_program_cost_usd"] - out["health_sector_savings_usd"]
    )
    return out

def build_combined_daly_df(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Build a combined draw-level dataframe for programme ICER calculations."""
    dfs = [df for df in dfs if df is not None and not df.empty]
    base_cols = ["daly_total", "daly_total_mda"]
    optional_cols = ["opd_total", "opd_total_mda", "ipd_total", "ipd_total_mda"]
    if not dfs:
        return pd.DataFrame(columns=base_cols + optional_cols)

    min_len = min(len(df) for df in dfs)
    combined = pd.DataFrame({col: np.zeros(min_len) for col in base_cols})

    for col in optional_cols:
        if any(col in df.columns for df in dfs):
            combined[col] = np.zeros(min_len)

    for df in dfs:
        for col in base_cols + optional_cols:
            if col in combined.columns and col in df.columns:
                combined[col] += pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()[:min_len]

    return add_daly_averted_columns(combined)

@cache_data
def cea_threshold(
    gdp_ppp_country: float,
    uk_cet: float = 26_705,
    gdp_ppp_uk: float = 46_659,
    elasticity: float = 1.478,
) -> float:
    """Woods et al. elasticity-based cost-effectiveness threshold."""
    gdp_ppp_country = max(float(gdp_ppp_country), 1.0)
    return float(uk_cet * (gdp_ppp_country / gdp_ppp_uk) ** elasticity)

def adj_daily_wage(
    annual_ppp: float,
    q1_share: float,
    weekly_hours: float = 40.0,
    hours_per_lost_workday: float = DEFAULT_HOURS_PER_WORKDAY,
) -> float:
    """Inequality-adjusted daily wage for the bottom quintile."""
    annual_bottom_quintile_income = float(annual_ppp) * float(q1_share) / 0.20
    weekly_hours = max(float(weekly_hours), 1.0)
    hourly_wage = annual_bottom_quintile_income / (weekly_hours * 52.0)
    return float(hourly_wage * max(float(hours_per_lost_workday), 0.0))

def daly_summary_table(df: pd.DataFrame, species: str) -> pd.DataFrame:
    """Summarize DALY burden with draw-level 95% PSA intervals."""
    if species == "mansoni":
        cols = {
            "Anemia": ("daly_anemia", "daly_anemia_mda"),
            "Hepatomegaly": ("daly_hepatomeg", "daly_hepatomeg_mda"),
            "Periportal fibrosis": ("daly_fibrosis", "daly_fibrosis_mda"),
            "Portal hypertension": ("daly_portal", "daly_portal_mda"),
            "Esophageal varices": ("daly_varices", "daly_varices_mda"),
            "Total": ("daly_total", "daly_total_mda"),
        }
    else:
        cols = {
            "Hematuria": ("daly_hem", "daly_hem_mda"),
            "Hydronephrosis": ("daly_hyd", "daly_hyd_mda"),
            "Female genital schistosomiasis": ("daly_fgs", "daly_fgs_mda"),
            "Attributable bladder cancer (YLD + YLL)": ("daly_cancer", "daly_cancer_mda"),
            "Total": ("daly_total", "daly_total_mda"),
        }

    rows = []
    for label, (c_no, c_mda) in cols.items():
        if c_no not in df.columns or c_mda not in df.columns:
            continue
        no_vals = pd.to_numeric(df[c_no], errors="coerce")
        mda_vals = pd.to_numeric(df[c_mda], errors="coerce")
        averted_vals = no_vals - mda_vals
        no_mean, no_lo, no_hi = _ci_bounds(no_vals)
        mda_mean, mda_lo, mda_hi = _ci_bounds(mda_vals)
        av_mean, av_lo, av_hi = _ci_bounds(averted_vals)
        rows.append(
            {
                "Outcome": label,
                "No-MDA mean": no_mean,
                "No-MDA 95% CI lower": no_lo,
                "No-MDA 95% CI upper": no_hi,
                "MDA mean": mda_mean,
                "MDA 95% CI lower": mda_lo,
                "MDA 95% CI upper": mda_hi,
                "DALYs averted": av_mean,
                "DALYs averted 95% CI lower": av_lo,
                "DALYs averted 95% CI upper": av_hi,
            }
        )
    return pd.DataFrame(rows)

def bladder_cancer_case_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize S. haematobium-attributable bladder-cancer case impact.

    PSA outputs before v1.3.2 do not contain explicit MDA case-count columns, so
    this function returns an empty table when the recalibrated columns are absent.
    """
    required = {
        "total_ca",
        "total_ca_mda",
        "attributable_ca",
        "attributable_ca_mda",
        "cancer_cases_averted",
        "effective_cancer_reduction",
        "paf",
    }
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    rows = [
        ("All-cause bladder cancer cases", "total_ca", "total_ca_mda"),
        ("S. haematobium-attributable bladder cancer cases", "attributable_ca", "attributable_ca_mda"),
    ]
    out_rows = []
    for label, no_col, mda_col in rows:
        no_vals = pd.to_numeric(df[no_col], errors="coerce")
        mda_vals = pd.to_numeric(df[mda_col], errors="coerce")
        averted = no_vals - mda_vals
        pct_reduction = averted / no_vals.replace(0, np.nan)
        no_mean, no_lo, no_hi = _ci_bounds(no_vals)
        mda_mean, mda_lo, mda_hi = _ci_bounds(mda_vals)
        av_mean, av_lo, av_hi = _ci_bounds(averted)
        red_mean, red_lo, red_hi = _ci_bounds(pct_reduction)
        out_rows.append(
            {
                "Outcome": label,
                "No-MDA mean": no_mean,
                "No-MDA 95% CI lower": no_lo,
                "No-MDA 95% CI upper": no_hi,
                "MDA mean": mda_mean,
                "MDA 95% CI lower": mda_lo,
                "MDA 95% CI upper": mda_hi,
                "Cases averted": av_mean,
                "Cases averted 95% CI lower": av_lo,
                "Cases averted 95% CI upper": av_hi,
                "Mean % reduction": red_mean,
                "Mean % reduction 95% CI lower": red_lo,
                "Mean % reduction 95% CI upper": red_hi,
            }
        )

    paf_mean, paf_lo, paf_hi = _ci_bounds(pd.to_numeric(df["paf"], errors="coerce"))
    eff_mean, eff_lo, eff_hi = _ci_bounds(pd.to_numeric(df["effective_cancer_reduction"], errors="coerce"))
    for label, mean, lo, hi in [
        ("Levin PAF", paf_mean, paf_lo, paf_hi),
        ("Effective attributable-cancer reduction", eff_mean, eff_lo, eff_hi),
    ]:
        out_rows.append(
            {
                "Outcome": label,
                "No-MDA mean": mean,
                "No-MDA 95% CI lower": lo,
                "No-MDA 95% CI upper": hi,
                "MDA mean": np.nan,
                "MDA 95% CI lower": np.nan,
                "MDA 95% CI upper": np.nan,
                "Cases averted": np.nan,
                "Cases averted 95% CI lower": np.nan,
                "Cases averted 95% CI upper": np.nan,
                "Mean % reduction": np.nan,
                "Mean % reduction 95% CI lower": np.nan,
                "Mean % reduction 95% CI upper": np.nan,
            }
        )
    return pd.DataFrame(out_rows)

def productivity_days_gained_draws(df: pd.DataFrame) -> np.ndarray:
    """Draw-level annual workdays gained."""
    if df is None or df.empty or not {"work_days_lost", "work_days_lost_mda"}.issubset(df.columns):
        return np.array([], dtype=float)
    no_draws = pd.to_numeric(df["work_days_lost"], errors="coerce").to_numpy(dtype=float)
    mda_draws = pd.to_numeric(df["work_days_lost_mda"], errors="coerce").to_numpy(dtype=float)
    return no_draws - mda_draws

def productivity_gain_draws(df: pd.DataFrame, daily_wage: float) -> np.ndarray:
    """Draw-level annual productivity gains in USD."""
    return productivity_days_gained_draws(df) * float(daily_wage)

def productivity_summary(
    df: pd.DataFrame,
    daily_wage: float,
    time_horizon: int,
    disc_effects: float,
) -> dict:
    """Summarize workdays and productivity gains with 95% PSA intervals."""
    no_draws = pd.to_numeric(df["work_days_lost"], errors="coerce")
    mda_draws = pd.to_numeric(df["work_days_lost_mda"], errors="coerce")
    gained_draws = no_draws - mda_draws
    econ_gain_draws = gained_draws * float(daily_wage)
    econ_gain_disc_draws = _discount_array(econ_gain_draws, time_horizon, disc_effects)

    mean_no, lo_no, hi_no = _ci_bounds(no_draws)
    mean_mda, lo_mda, hi_mda = _ci_bounds(mda_draws)
    days_gained, days_gained_lo, days_gained_hi = _ci_bounds(gained_draws)
    econ_gain, econ_gain_lo, econ_gain_hi = _ci_bounds(econ_gain_draws)
    econ_gain_disc, econ_gain_disc_lo, econ_gain_disc_hi = _ci_bounds(econ_gain_disc_draws)

    return {
        "mean_no": mean_no,
        "lo_no": lo_no,
        "hi_no": hi_no,
        "mean_mda": mean_mda,
        "lo_mda": lo_mda,
        "hi_mda": hi_mda,
        "days_gained": days_gained,
        "days_gained_lo": days_gained_lo,
        "days_gained_hi": days_gained_hi,
        "econ_gain_pa": econ_gain,
        "econ_gain_pa_lo": econ_gain_lo,
        "econ_gain_pa_hi": econ_gain_hi,
        "econ_gain_disc": econ_gain_disc,
        "econ_gain_disc_lo": econ_gain_disc_lo,
        "econ_gain_disc_hi": econ_gain_disc_hi,
    }

def health_sector_savings_draws(df: pd.DataFrame, opd_c: float, ipd_c: float) -> np.ndarray:
    """Draw-level annual health-sector savings in USD."""
    required = {"opd_total", "opd_total_mda", "ipd_total", "ipd_total_mda"}
    if df is None or df.empty or not required.issubset(df.columns):
        return np.array([], dtype=float)
    opd_no = pd.to_numeric(df["opd_total"], errors="coerce").to_numpy(dtype=float)
    opd_mda = pd.to_numeric(df["opd_total_mda"], errors="coerce").to_numpy(dtype=float)
    ipd_no = pd.to_numeric(df["ipd_total"], errors="coerce").to_numpy(dtype=float)
    ipd_mda = pd.to_numeric(df["ipd_total_mda"], errors="coerce").to_numpy(dtype=float)
    return (opd_no - opd_mda) * float(opd_c) + (ipd_no - ipd_mda) * float(ipd_c)

def health_sector_costs(
    df: pd.DataFrame,
    opd_c: float,
    ipd_c: float,
    time_horizon: int,
    disc_costs: float,
) -> dict:
    """Summarize health-sector costs and savings with 95% PSA intervals."""
    opd_no = pd.to_numeric(df["opd_total"], errors="coerce")
    opd_mda = pd.to_numeric(df["opd_total_mda"], errors="coerce")
    ipd_no = pd.to_numeric(df["ipd_total"], errors="coerce")
    ipd_mda = pd.to_numeric(df["ipd_total_mda"], errors="coerce")

    hs_no_draws = opd_no * float(opd_c) + ipd_no * float(ipd_c)
    hs_mda_draws = opd_mda * float(opd_c) + ipd_mda * float(ipd_c)
    savings_draws = hs_no_draws - hs_mda_draws
    savings_disc_draws = _discount_array(savings_draws, time_horizon, disc_costs)

    hs_cost_no, hs_cost_no_lo, hs_cost_no_hi = _ci_bounds(hs_no_draws)
    hs_cost_mda, hs_cost_mda_lo, hs_cost_mda_hi = _ci_bounds(hs_mda_draws)
    hs_savings_pa, hs_savings_lo, hs_savings_hi = _ci_bounds(savings_draws)
    hs_savings_disc, hs_savings_disc_lo, hs_savings_disc_hi = _ci_bounds(savings_disc_draws)

    return {
        "hs_cost_no": hs_cost_no,
        "hs_cost_no_lo": hs_cost_no_lo,
        "hs_cost_no_hi": hs_cost_no_hi,
        "hs_cost_mda": hs_cost_mda,
        "hs_cost_mda_lo": hs_cost_mda_lo,
        "hs_cost_mda_hi": hs_cost_mda_hi,
        "hs_savings_pa": hs_savings_pa,
        "hs_savings_lo": hs_savings_lo,
        "hs_savings_hi": hs_savings_hi,
        "hs_savings_disc": hs_savings_disc,
        "hs_savings_disc_lo": hs_savings_disc_lo,
        "hs_savings_disc_hi": hs_savings_disc_hi,
    }

def combined_benefit_draws(
    sim_dfs: list[pd.DataFrame],
    daily_wage: float,
    opd_c: float,
    ipd_c: float,
) -> dict:
    """Combine draw-level health-sector savings and productivity gains across species modules."""
    valid_dfs = [df for df in sim_dfs if df is not None and not df.empty]
    hs = _same_length_sum([health_sector_savings_draws(df, opd_c, ipd_c) for df in valid_dfs])
    econ = _same_length_sum([productivity_gain_draws(df, daily_wage) for df in valid_dfs])
    if hs.size and econ.size:
        min_len = min(hs.size, econ.size)
        hs = hs[:min_len]
        econ = econ[:min_len]
    elif hs.size and not econ.size:
        econ = np.zeros_like(hs)
    elif econ.size and not hs.size:
        hs = np.zeros_like(econ)
    total = hs + econ if hs.size or econ.size else np.array([], dtype=float)
    return {
        "hs_savings_pa": hs,
        "econ_gain_pa": econ,
        "total_benefits_pa": total,
    }

def roi_summary_from_draws(
    hs_savings_draws_arr: object,
    econ_gain_draws_arr: object,
    prog_cost: float,
) -> dict:
    """Draw-level ROI summary based on health-sector savings plus productivity gains."""
    hs = pd.to_numeric(pd.Series(hs_savings_draws_arr), errors="coerce").to_numpy(dtype=float)
    econ = pd.to_numeric(pd.Series(econ_gain_draws_arr), errors="coerce").to_numpy(dtype=float)
    if hs.size and econ.size:
        min_len = min(hs.size, econ.size)
        benefits = hs[:min_len] + econ[:min_len]
    elif hs.size:
        benefits = hs
    elif econ.size:
        benefits = econ
    else:
        benefits = np.array([], dtype=float)
    roi_draws = benefits / max(float(prog_cost), 1.0)
    roi_mean, roi_lo, roi_hi = _ci_bounds(roi_draws)
    benefit_mean, benefit_lo, benefit_hi = _ci_bounds(benefits)
    hs_mean, hs_lo, hs_hi = _ci_bounds(hs)
    econ_mean, econ_lo, econ_hi = _ci_bounds(econ)
    return {
        "roi_mean": roi_mean,
        "roi_lo": roi_lo,
        "roi_hi": roi_hi,
        "benefit_mean": benefit_mean,
        "benefit_lo": benefit_lo,
        "benefit_hi": benefit_hi,
        "hs_savings_mean": hs_mean,
        "hs_savings_lo": hs_lo,
        "hs_savings_hi": hs_hi,
        "econ_gain_mean": econ_mean,
        "econ_gain_lo": econ_lo,
        "econ_gain_hi": econ_hi,
        "roi_draws": roi_draws,
        "benefit_draws": benefits,
    }

def productivity_result_table(prod: dict) -> pd.DataFrame:
    """Create a visible productivity-results table with 95% intervals."""
    return pd.DataFrame(
        [
            {"Outcome": "Workdays lost p.a. - no MDA", "Unit": "days", "Mean": prod.get("mean_no", np.nan), "95% CI lower": prod.get("lo_no", np.nan), "95% CI upper": prod.get("hi_no", np.nan)},
            {"Outcome": "Workdays lost p.a. - MDA", "Unit": "days", "Mean": prod.get("mean_mda", np.nan), "95% CI lower": prod.get("lo_mda", np.nan), "95% CI upper": prod.get("hi_mda", np.nan)},
            {"Outcome": "Productivity days gained p.a.", "Unit": "days", "Mean": prod.get("days_gained", np.nan), "95% CI lower": prod.get("days_gained_lo", np.nan), "95% CI upper": prod.get("days_gained_hi", np.nan)},
            {"Outcome": "Annual productivity gain", "Unit": "USD", "Mean": prod.get("econ_gain_pa", np.nan), "95% CI lower": prod.get("econ_gain_pa_lo", np.nan), "95% CI upper": prod.get("econ_gain_pa_hi", np.nan)},
            {"Outcome": "Discounted productivity gain", "Unit": "USD", "Mean": prod.get("econ_gain_disc", np.nan), "95% CI lower": prod.get("econ_gain_disc_lo", np.nan), "95% CI upper": prod.get("econ_gain_disc_hi", np.nan)},
        ]
    )

def health_sector_result_table(hsc: dict) -> pd.DataFrame:
    """Create a visible health-sector-cost table with 95% intervals."""
    return pd.DataFrame(
        [
            {"Outcome": "Annual HS cost - no MDA", "Unit": "USD", "Mean": hsc.get("hs_cost_no", np.nan), "95% CI lower": hsc.get("hs_cost_no_lo", np.nan), "95% CI upper": hsc.get("hs_cost_no_hi", np.nan)},
            {"Outcome": "Annual HS cost - MDA", "Unit": "USD", "Mean": hsc.get("hs_cost_mda", np.nan), "95% CI lower": hsc.get("hs_cost_mda_lo", np.nan), "95% CI upper": hsc.get("hs_cost_mda_hi", np.nan)},
            {"Outcome": "Annual HS savings", "Unit": "USD", "Mean": hsc.get("hs_savings_pa", np.nan), "95% CI lower": hsc.get("hs_savings_lo", np.nan), "95% CI upper": hsc.get("hs_savings_hi", np.nan)},
            {"Outcome": "Discounted HS savings", "Unit": "USD", "Mean": hsc.get("hs_savings_disc", np.nan), "95% CI lower": hsc.get("hs_savings_disc_lo", np.nan), "95% CI upper": hsc.get("hs_savings_disc_hi", np.nan)},
        ]
    )

def cost_effectiveness_summary_table(icer_res: dict) -> pd.DataFrame:
    """Create a CE summary table with 95% intervals."""
    def pct100(value: object) -> float:
        try:
            return float(value) * 100.0
        except (TypeError, ValueError):
            return np.nan

    return pd.DataFrame(
        [
            {"Outcome": "DALYs averted p.a.", "Unit": "DALYs", "Mean": icer_res.get("dalys_averted_point", np.nan), "95% CI lower": icer_res.get("dalys_averted_lo", np.nan), "95% CI upper": icer_res.get("dalys_averted_hi", np.nan)},
            {"Outcome": "Incremental cost", "Unit": "USD", "Mean": icer_res.get("incremental_cost_mean", np.nan), "95% CI lower": icer_res.get("incremental_cost_lo", np.nan), "95% CI upper": icer_res.get("incremental_cost_hi", np.nan)},
            {"Outcome": "ICER", "Unit": "USD/DALY", "Mean": icer_res.get("icer_mean", np.nan), "95% CI lower": icer_res.get("icer_lo", np.nan), "95% CI upper": icer_res.get("icer_hi", np.nan)},
            {"Outcome": "NMB at Woods CET", "Unit": "USD", "Mean": icer_res.get("nmb_at_woods_cet_mean", np.nan), "95% CI lower": icer_res.get("nmb_at_woods_cet_lo", np.nan), "95% CI upper": icer_res.get("nmb_at_woods_cet_hi", np.nan)},
            {"Outcome": "NMB at 1x GDP threshold", "Unit": "USD", "Mean": icer_res.get("nmb_at_gdp_threshold_mean", np.nan), "95% CI lower": icer_res.get("nmb_at_gdp_threshold_lo", np.nan), "95% CI upper": icer_res.get("nmb_at_gdp_threshold_hi", np.nan)},
            {"Outcome": "Probability cost-effective at Woods CET", "Unit": "%", "Mean": pct100(icer_res.get("pct_cost_effective_cet", np.nan)), "95% CI lower": pct100(icer_res.get("pct_cost_effective_cet_lo", np.nan)), "95% CI upper": pct100(icer_res.get("pct_cost_effective_cet_hi", np.nan))},
            {"Outcome": "Probability cost-effective at 1x GDP", "Unit": "%", "Mean": pct100(icer_res.get("pct_cost_effective_gdp", np.nan)), "95% CI lower": pct100(icer_res.get("pct_cost_effective_gdp_lo", np.nan)), "95% CI upper": pct100(icer_res.get("pct_cost_effective_gdp_hi", np.nan))},
        ]
    )

def roi_summary_table(roi_res: dict) -> pd.DataFrame:
    """Create a ROI summary table with 95% intervals."""
    return pd.DataFrame(
        [
            {"Outcome": "Annual health-sector savings", "Unit": "USD", "Mean": roi_res.get("hs_savings_mean", np.nan), "95% CI lower": roi_res.get("hs_savings_lo", np.nan), "95% CI upper": roi_res.get("hs_savings_hi", np.nan)},
            {"Outcome": "Annual productivity gains", "Unit": "USD", "Mean": roi_res.get("econ_gain_mean", np.nan), "95% CI lower": roi_res.get("econ_gain_lo", np.nan), "95% CI upper": roi_res.get("econ_gain_hi", np.nan)},
            {"Outcome": "Total annual benefits", "Unit": "USD", "Mean": roi_res.get("benefit_mean", np.nan), "95% CI lower": roi_res.get("benefit_lo", np.nan), "95% CI upper": roi_res.get("benefit_hi", np.nan)},
            {"Outcome": "Return per $1 invested", "Unit": "ratio", "Mean": roi_res.get("roi_mean", np.nan), "95% CI lower": roi_res.get("roi_lo", np.nan), "95% CI upper": roi_res.get("roi_hi", np.nan)},
        ]
    )

def _align_draw_arrays(
    hs_savings_draws_arr: object,
    econ_gain_draws_arr: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return same-length health-sector and productivity-gain draw arrays."""
    hs = pd.to_numeric(pd.Series(hs_savings_draws_arr), errors="coerce").to_numpy(dtype=float)
    if econ_gain_draws_arr is None:
        econ = np.zeros_like(hs, dtype=float)
    else:
        econ = pd.to_numeric(pd.Series(econ_gain_draws_arr), errors="coerce").to_numpy(dtype=float)

    if hs.size and econ.size:
        min_len = min(hs.size, econ.size)
        return hs[:min_len], econ[:min_len]
    if hs.size and not econ.size:
        return hs, np.zeros_like(hs, dtype=float)
    if econ.size and not hs.size:
        return np.zeros_like(econ, dtype=float), econ
    return np.array([], dtype=float), np.array([], dtype=float)


def add_economic_impact_intervals(
    economic_df: pd.DataFrame,
    hs_savings_draws_arr: object,
    econ_gain_draws_arr: object,
    disc_rate: float,
) -> pd.DataFrame:
    """Add draw-based 95% interval columns to an economic-impact table.

    Economic impact takes a societal/economic perspective and therefore includes
    health-sector savings plus productivity gains. It should not be labelled as
    a budget-impact analysis.
    """
    out = economic_df.copy()
    hs, econ = _align_draw_arrays(hs_savings_draws_arr, econ_gain_draws_arr)
    if not hs.size and not econ.size:
        return out

    cumulative_net_draws = np.zeros_like(hs, dtype=float)
    for i, row in out.iterrows():
        year_idx = int(i) + 1
        disc = (1.0 + float(disc_rate)) ** year_idx
        hs_disc = hs / disc
        econ_disc = econ / disc
        total_benefit_draws = hs_disc + econ_disc
        net_draws = total_benefit_draws - float(row["Total_prog_cost_USD"])
        cumulative_net_draws = cumulative_net_draws + net_draws

        hs_mean, hs_lo, hs_hi = _ci_bounds(hs_disc)
        econ_mean, econ_lo, econ_hi = _ci_bounds(econ_disc)
        total_mean, total_lo, total_hi = _ci_bounds(total_benefit_draws)
        net_mean, net_lo, net_hi = _ci_bounds(net_draws)
        cum_mean, cum_lo, cum_hi = _ci_bounds(cumulative_net_draws)

        out.loc[i, "Health_sector_savings_USD"] = hs_mean
        out.loc[i, "Health_sector_savings_USD_95CI_lower"] = hs_lo
        out.loc[i, "Health_sector_savings_USD_95CI_upper"] = hs_hi
        out.loc[i, "Economic_gains_USD"] = econ_mean
        out.loc[i, "Economic_gains_USD_95CI_lower"] = econ_lo
        out.loc[i, "Economic_gains_USD_95CI_upper"] = econ_hi
        out.loc[i, "Total_economic_benefits_USD"] = total_mean
        out.loc[i, "Total_economic_benefits_USD_95CI_lower"] = total_lo
        out.loc[i, "Total_economic_benefits_USD_95CI_upper"] = total_hi
        out.loc[i, "Net_economic_benefit_USD"] = net_mean
        out.loc[i, "Net_economic_benefit_USD_95CI_lower"] = net_lo
        out.loc[i, "Net_economic_benefit_USD_95CI_upper"] = net_hi
        out.loc[i, "Cumulative_net_economic_benefit_USD"] = cum_mean
        out.loc[i, "Cumulative_net_economic_benefit_USD_95CI_lower"] = cum_lo
        out.loc[i, "Cumulative_net_economic_benefit_USD_95CI_upper"] = cum_hi
    return out


def add_budget_impact_intervals(
    budget_df: pd.DataFrame,
    clinical_offset_draws_arr: object,
    disc_rate: float,
    staff_time_value_draws_arr: object | None = None,
    staff_hours_draws_arr: object | None = None,
    include_staff_time_value: bool = False,
) -> pd.DataFrame:
    """Add draw-based 95% interval columns to a budget-impact table.

    Budget impact excludes productivity and other societal benefits. It reports
    gross programme costs, morbidity-state clinical-management offsets, staff
    time savings, and net budget impact. Staff time value is only netted against
    the budget when include_staff_time_value is True.
    """
    out = budget_df.copy()
    clinical, staff_value = _align_draw_arrays(clinical_offset_draws_arr, staff_time_value_draws_arr)
    if not clinical.size and not staff_value.size:
        return out
    if not clinical.size:
        clinical = np.zeros_like(staff_value, dtype=float)
    if not staff_value.size:
        staff_value = np.zeros_like(clinical, dtype=float)

    if staff_hours_draws_arr is None:
        staff_hours = np.zeros_like(clinical, dtype=float)
    else:
        staff_hours = pd.to_numeric(pd.Series(staff_hours_draws_arr), errors="coerce").to_numpy(dtype=float)
        if staff_hours.size and clinical.size:
            min_len = min(staff_hours.size, clinical.size)
            staff_hours = np.nan_to_num(staff_hours[:min_len], nan=0.0, posinf=0.0, neginf=0.0)
            clinical = clinical[:min_len]
            staff_value = staff_value[:min_len]
        elif not staff_hours.size:
            staff_hours = np.zeros_like(clinical, dtype=float)

    cumulative_net_budget_draws = np.zeros_like(clinical, dtype=float)
    cumulative_staff_hours_draws = np.zeros_like(clinical, dtype=float)
    for i, row in out.iterrows():
        year_idx = int(i) + 1
        disc = (1.0 + float(disc_rate)) ** year_idx
        clinical_disc = clinical / disc
        staff_value_disc = staff_value / disc
        staff_value_included_disc = staff_value_disc if include_staff_time_value else np.zeros_like(staff_value_disc)
        total_offset_draws = clinical_disc + staff_value_included_disc
        net_budget_draws = float(row["Total_prog_cost_USD"]) - total_offset_draws
        cumulative_net_budget_draws = cumulative_net_budget_draws + net_budget_draws
        cumulative_staff_hours_draws = cumulative_staff_hours_draws + staff_hours

        clinical_mean, clinical_lo, clinical_hi = _ci_bounds(clinical_disc)
        staff_val_mean, staff_val_lo, staff_val_hi = _ci_bounds(staff_value_disc)
        staff_inc_mean, staff_inc_lo, staff_inc_hi = _ci_bounds(staff_value_included_disc)
        staff_hours_mean, staff_hours_lo, staff_hours_hi = _ci_bounds(staff_hours)
        cum_staff_hours_mean, cum_staff_hours_lo, cum_staff_hours_hi = _ci_bounds(cumulative_staff_hours_draws)
        offset_mean, offset_lo, offset_hi = _ci_bounds(total_offset_draws)
        net_mean, net_lo, net_hi = _ci_bounds(net_budget_draws)
        cum_mean, cum_lo, cum_hi = _ci_bounds(cumulative_net_budget_draws)

        out.loc[i, "Clinical_management_budget_offset_USD"] = clinical_mean
        out.loc[i, "Clinical_management_budget_offset_USD_95CI_lower"] = clinical_lo
        out.loc[i, "Clinical_management_budget_offset_USD_95CI_upper"] = clinical_hi
        out.loc[i, "Staff_time_hours_saved"] = staff_hours_mean
        out.loc[i, "Staff_time_hours_saved_95CI_lower"] = staff_hours_lo
        out.loc[i, "Staff_time_hours_saved_95CI_upper"] = staff_hours_hi
        out.loc[i, "Cumulative_staff_time_hours_saved"] = cum_staff_hours_mean
        out.loc[i, "Cumulative_staff_time_hours_saved_95CI_lower"] = cum_staff_hours_lo
        out.loc[i, "Cumulative_staff_time_hours_saved_95CI_upper"] = cum_staff_hours_hi
        out.loc[i, "Staff_time_value_USD"] = staff_val_mean
        out.loc[i, "Staff_time_value_USD_95CI_lower"] = staff_val_lo
        out.loc[i, "Staff_time_value_USD_95CI_upper"] = staff_val_hi
        out.loc[i, "Staff_time_value_included_USD"] = staff_inc_mean
        out.loc[i, "Staff_time_value_included_USD_95CI_lower"] = staff_inc_lo
        out.loc[i, "Staff_time_value_included_USD_95CI_upper"] = staff_inc_hi
        out.loc[i, "Health_sector_budget_offset_USD"] = offset_mean
        out.loc[i, "Health_sector_budget_offset_USD_95CI_lower"] = offset_lo
        out.loc[i, "Health_sector_budget_offset_USD_95CI_upper"] = offset_hi
        out.loc[i, "Net_budget_impact_USD"] = net_mean
        out.loc[i, "Net_budget_impact_USD_95CI_lower"] = net_lo
        out.loc[i, "Net_budget_impact_USD_95CI_upper"] = net_hi
        out.loc[i, "Cumulative_net_budget_impact_USD"] = cum_mean
        out.loc[i, "Cumulative_net_budget_impact_USD_95CI_lower"] = cum_lo
        out.loc[i, "Cumulative_net_budget_impact_USD_95CI_upper"] = cum_hi
    return out

def economic_benefit_cost_ratio_summary(
    economic_df: pd.DataFrame,
    hs_savings_draws_arr: object,
    econ_gain_draws_arr: object,
    disc_rate: float,
) -> dict:
    """Draw-level benefit-cost ratio across the economic-impact horizon."""
    total_cost = float(pd.to_numeric(economic_df["Total_prog_cost_USD"], errors="coerce").fillna(0.0).sum())
    hs, econ = _align_draw_arrays(hs_savings_draws_arr, econ_gain_draws_arr)
    if not hs.size and not econ.size:
        return {"bcr_mean": np.nan, "bcr_lo": np.nan, "bcr_hi": np.nan, "total_cost": total_cost}

    total_benefit_draws = np.zeros_like(hs, dtype=float)
    for year_idx in range(1, len(economic_df) + 1):
        disc = (1.0 + float(disc_rate)) ** year_idx
        total_benefit_draws += (hs + econ) / disc
    bcr_draws = total_benefit_draws / max(total_cost, 1.0)
    bcr_mean, bcr_lo, bcr_hi = _ci_bounds(bcr_draws)
    return {"bcr_mean": bcr_mean, "bcr_lo": bcr_lo, "bcr_hi": bcr_hi, "total_cost": total_cost}


# Backward-compatible name for code that already imports this function.
benefit_cost_ratio_summary = economic_benefit_cost_ratio_summary

def compute_icer(
    df: pd.DataFrame,
    annual_prog_cost: float,
    annual_ppp: float,
    cet: float,
    incremental_cost_draws: Optional[object] = None,
) -> dict:
    """Compute ICER, CEAC, and draw-level CE-plane outputs with 95% intervals.

    Each row in df is one PSA draw. DALYs averted are calculated as no-MDA
    DALYs minus MDA DALYs for that draw. By default, incremental cost is the
    annual programme cost for every draw. Pass incremental_cost_draws to use a
    draw-level net cost, such as programme cost minus health-sector savings.
    """
    required_cols = {"daly_total", "daly_total_mda"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(
            "compute_icer requires columns: " + ", ".join(sorted(required_cols))
            + f". Missing: {', '.join(sorted(missing_cols))}"
        )

    daly_total = pd.to_numeric(df["daly_total"], errors="coerce").to_numpy(dtype=float)
    daly_total_mda = pd.to_numeric(df["daly_total_mda"], errors="coerce").to_numpy(dtype=float)
    finite_daly_draws = np.isfinite(daly_total) & np.isfinite(daly_total_mda)

    dalys_averted_draws = np.full(daly_total.shape, np.nan, dtype=float)
    dalys_averted_draws[finite_daly_draws] = daly_total[finite_daly_draws] - daly_total_mda[finite_daly_draws]

    daly_total_mean, daly_total_lo, daly_total_hi = _ci_bounds(daly_total[finite_daly_draws])
    daly_total_mda_mean, daly_total_mda_lo, daly_total_mda_hi = _ci_bounds(daly_total_mda[finite_daly_draws])
    dalys_averted_point, dalys_averted_lo, dalys_averted_hi = _ci_bounds(dalys_averted_draws[finite_daly_draws])

    annual_prog_cost = float(annual_prog_cost)
    annual_ppp = float(annual_ppp)
    cet = float(cet)

    if incremental_cost_draws is None:
        incremental_cost = np.full(dalys_averted_draws.shape, annual_prog_cost, dtype=float)
    else:
        incremental_cost = pd.to_numeric(pd.Series(incremental_cost_draws), errors="coerce").to_numpy(dtype=float)
        if incremental_cost.size != dalys_averted_draws.size:
            raise ValueError("incremental_cost_draws must have the same length as the PSA dataframe.")

    finite_cost_draws = np.isfinite(incremental_cost)
    positive_daly_draws = finite_daly_draws & (dalys_averted_draws > 0)
    nonpositive_daly_draws = finite_daly_draws & (dalys_averted_draws <= 0)

    icer_vec = np.full(dalys_averted_draws.shape, np.nan, dtype=float)
    np.divide(incremental_cost, dalys_averted_draws, out=icer_vec, where=positive_daly_draws & finite_cost_draws)

    def _prob_cost_effective_info(wtp: float) -> tuple[float, float, float, int, int]:
        nmb = dalys_averted_draws * float(wtp) - incremental_cost
        valid_nmb = np.isfinite(nmb)
        total = int(valid_nmb.sum())
        if total <= 0:
            return (np.nan, np.nan, np.nan, 0, 0)
        successes = int((nmb[valid_nmb] > 0).sum())
        prob = successes / total
        lo, hi = _wilson_ci(successes, total)
        return (float(prob), lo, hi, successes, total)

    def _prob_cost_effective(wtp: float) -> float:
        return _prob_cost_effective_info(wtp)[0]

    wtp_max = max(annual_ppp * 5.0, cet * 1.5, 1.0)
    wtp_range = np.linspace(0.0, wtp_max, 500)
    ceac_probs = np.array([_prob_cost_effective(wtp) for wtp in wtp_range])

    valid_icers = np.isfinite(icer_vec)
    icer_mean, icer_lo, icer_hi = _ci_bounds(icer_vec[valid_icers])

    pct_cet, pct_cet_lo, pct_cet_hi, pct_cet_successes, pct_cet_total = (
        _prob_cost_effective_info(cet) if cet > 0 and dalys_averted_draws.size else (np.nan, np.nan, np.nan, 0, 0)
    )
    pct_gdp, pct_gdp_lo, pct_gdp_hi, pct_gdp_successes, pct_gdp_total = (
        _prob_cost_effective_info(annual_ppp) if annual_ppp > 0 and dalys_averted_draws.size else (np.nan, np.nan, np.nan, 0, 0)
    )

    annual_program_cost = (
        pd.to_numeric(df["annual_program_cost_usd"], errors="coerce").to_numpy(dtype=float)
        if "annual_program_cost_usd" in df.columns
        else np.full(dalys_averted_draws.shape, annual_prog_cost, dtype=float)
    )
    health_sector_savings = (
        pd.to_numeric(df["health_sector_savings_usd"], errors="coerce").to_numpy(dtype=float)
        if "health_sector_savings_usd" in df.columns
        else np.zeros(dalys_averted_draws.shape, dtype=float)
    )
    program_cost_minus_savings = (
        pd.to_numeric(df["program_cost_minus_savings"], errors="coerce").to_numpy(dtype=float)
        if "program_cost_minus_savings" in df.columns
        else annual_program_cost - health_sector_savings
    )
    incremental_cost_mean, incremental_cost_lo, incremental_cost_hi = _ci_bounds(incremental_cost)
    health_savings_mean, health_savings_lo, health_savings_hi = _ci_bounds(health_sector_savings)
    nmb_cet = dalys_averted_draws * cet - incremental_cost
    nmb_gdp = dalys_averted_draws * annual_ppp - incremental_cost
    nmb_cet_mean, nmb_cet_lo, nmb_cet_hi = _ci_bounds(nmb_cet)
    nmb_gdp_mean, nmb_gdp_lo, nmb_gdp_hi = _ci_bounds(nmb_gdp)

    draw_level_outputs = pd.DataFrame(
        {
            "draw": np.arange(1, dalys_averted_draws.size + 1, dtype=int),
            "daly_total_no_mda": daly_total,
            "daly_total_mda": daly_total_mda,
            "dalys_averted": dalys_averted_draws,
            "annual_program_cost_usd": annual_program_cost,
            "health_sector_savings_usd": health_sector_savings,
            "program_cost_minus_savings": program_cost_minus_savings,
            "incremental_cost_usd": incremental_cost,
            "icer_usd_per_daly": icer_vec,
            "nmb_at_woods_cet": nmb_cet,
            "nmb_at_gdp_threshold": nmb_gdp,
        }
    )

    return {
        "daly_total_mean": daly_total_mean,
        "daly_total_lo": daly_total_lo,
        "daly_total_hi": daly_total_hi,
        "daly_total_mda_mean": daly_total_mda_mean,
        "daly_total_mda_lo": daly_total_mda_lo,
        "daly_total_mda_hi": daly_total_mda_hi,
        "dalys_averted_point": dalys_averted_point,
        "dalys_averted_mean": dalys_averted_point,
        "dalys_averted_draws": dalys_averted_draws,
        "dalys_averted_lo": dalys_averted_lo,
        "dalys_averted_hi": dalys_averted_hi,
        "draw_level_outputs": draw_level_outputs,
        "incremental_cost_draws": incremental_cost,
        "incremental_cost_mean": incremental_cost_mean,
        "incremental_cost_lo": incremental_cost_lo,
        "incremental_cost_hi": incremental_cost_hi,
        "health_sector_savings_mean": health_savings_mean,
        "health_sector_savings_lo": health_savings_lo,
        "health_sector_savings_hi": health_savings_hi,
        "icer_draws": icer_vec,
        "icer_mean": icer_mean,
        "icer_lo": icer_lo,
        "icer_hi": icer_hi,
        "nmb_at_woods_cet_mean": nmb_cet_mean,
        "nmb_at_woods_cet_lo": nmb_cet_lo,
        "nmb_at_woods_cet_hi": nmb_cet_hi,
        "nmb_at_gdp_threshold_mean": nmb_gdp_mean,
        "nmb_at_gdp_threshold_lo": nmb_gdp_lo,
        "nmb_at_gdp_threshold_hi": nmb_gdp_hi,
        "ceac_wtp": wtp_range,
        "ceac_prob": ceac_probs,
        "pct_cost_effective_cet": pct_cet,
        "pct_cost_effective_cet_lo": pct_cet_lo,
        "pct_cost_effective_cet_hi": pct_cet_hi,
        "pct_cost_effective_cet_successes": pct_cet_successes,
        "pct_cost_effective_cet_total": pct_cet_total,
        "pct_cost_effective_gdp": pct_gdp,
        "pct_cost_effective_gdp_lo": pct_gdp_lo,
        "pct_cost_effective_gdp_hi": pct_gdp_hi,
        "pct_cost_effective_gdp_successes": pct_gdp_successes,
        "pct_cost_effective_gdp_total": pct_gdp_total,
        "positive_daly_draws": int(positive_daly_draws.sum()),
        "nonpositive_daly_draws": int(nonpositive_daly_draws.sum()),
        "valid_icer_draws": int(valid_icers.sum()),
        "finite_daly_draws": int(finite_daly_draws.sum()),
        "invalid_daly_draws": int((~finite_daly_draws).sum()),
        "total_draws": int(dalys_averted_draws.size),
        "positive_daly_share": float(positive_daly_draws.sum() / finite_daly_draws.sum()) if finite_daly_draws.any() else np.nan,
    }

def compute_roi(hs_savings: float, econ_gain: float, prog_cost: float) -> float:
    return (float(hs_savings) + float(econ_gain)) / max(float(prog_cost), 1.0)


def _programme_cost_schedule(
    off_year_fixed_cost: float,
    horizon: int,
    disc_rate: float,
    pzq_cost: float,
    pop_treat: float,
    pzq_per_person: float,
    delivery_c: float,
    fixed_costs_mda_year: float,
    freq: str = "Annual",
    base_year: int = 2026,
) -> pd.DataFrame:
    """Project discounted programme costs year by year."""
    rows = []
    for yr in range(1, int(horizon) + 1):
        disc = (1.0 + float(disc_rate)) ** yr
        mda_this_year = (yr % 2 == 1) if freq == "Biennial" else True

        if mda_this_year:
            drug_cost = float(pzq_cost) * float(pzq_per_person) * float(pop_treat) / disc
            deliv_cost = float(delivery_c) * float(pop_treat) / disc
            other_cost = float(fixed_costs_mda_year) / disc
        else:
            drug_cost = 0.0
            deliv_cost = 0.0
            other_cost = float(off_year_fixed_cost) / disc
        prog_cost_yr = drug_cost + deliv_cost + other_cost

        rows.append(
            {
                "Year": int(base_year) + yr - 1,
                "MDA_delivered": bool(mda_this_year),
                "Drug_costs_USD": drug_cost,
                "Delivery_costs_USD": deliv_cost,
                "Other_prog_USD": other_cost,
                "Total_prog_cost_USD": prog_cost_yr,
            }
        )

    df = pd.DataFrame(rows)
    df["Cumulative_prog_cost_USD"] = df["Total_prog_cost_USD"].cumsum()
    return df


def economic_impact_analysis(
    programme_cost_mda_year: float,
    off_year_fixed_cost: float,
    hs_savings_pa: float,
    econ_gain_pa: float,
    horizon: int,
    disc_rate: float,
    pzq_cost: float,
    pop_treat: float,
    pzq_per_person: float,
    delivery_c: float,
    fixed_costs_mda_year: float,
    freq: str = "Annual",
    base_year: int = 2026,
) -> pd.DataFrame:
    """Project discounted costs and societal/economic benefits year by year.

    This is not a budget-impact analysis because it includes productivity gains.
    """
    _ = programme_cost_mda_year
    df = _programme_cost_schedule(
        off_year_fixed_cost=off_year_fixed_cost,
        horizon=horizon,
        disc_rate=disc_rate,
        pzq_cost=pzq_cost,
        pop_treat=pop_treat,
        pzq_per_person=pzq_per_person,
        delivery_c=delivery_c,
        fixed_costs_mda_year=fixed_costs_mda_year,
        freq=freq,
        base_year=base_year,
    )

    hs_vals = []
    econ_vals = []
    total_benefit_vals = []
    net_vals = []
    for idx, row in df.iterrows():
        year_idx = int(idx) + 1
        disc = (1.0 + float(disc_rate)) ** year_idx
        hs_sav_disc = float(hs_savings_pa) / disc
        econ_gain_disc = float(econ_gain_pa) / disc
        total_benefit = hs_sav_disc + econ_gain_disc
        net_benefit = total_benefit - float(row["Total_prog_cost_USD"])
        hs_vals.append(hs_sav_disc)
        econ_vals.append(econ_gain_disc)
        total_benefit_vals.append(total_benefit)
        net_vals.append(net_benefit)

    df["Health_sector_savings_USD"] = hs_vals
    df["Economic_gains_USD"] = econ_vals
    df["Total_economic_benefits_USD"] = total_benefit_vals
    df["Net_economic_benefit_USD"] = net_vals
    df["Cumulative_net_economic_benefit_USD"] = pd.Series(net_vals, dtype=float).cumsum()
    return df


def budget_impact_analysis(
    off_year_fixed_cost: float,
    hs_savings_pa: float,
    horizon: int,
    disc_rate: float,
    pzq_cost: float,
    pop_treat: float,
    pzq_per_person: float,
    delivery_c: float,
    fixed_costs_mda_year: float,
    freq: str = "Annual",
    base_year: int = 2026,
    staff_time_value_pa: float = 0.0,
    staff_hours_saved_pa: float = 0.0,
    include_staff_time_value: bool = False,
) -> pd.DataFrame:
    """Project health-system/payer budget impact year by year.

    hs_savings_pa is interpreted as the annual clinical-management budget
    offset from health states averted. Productivity gains are excluded. Staff
    time saved is reported in hours; monetized staff value is only included in
    net budget impact when include_staff_time_value is True.
    """
    df = _programme_cost_schedule(
        off_year_fixed_cost=off_year_fixed_cost,
        horizon=horizon,
        disc_rate=disc_rate,
        pzq_cost=pzq_cost,
        pop_treat=pop_treat,
        pzq_per_person=pzq_per_person,
        delivery_c=delivery_c,
        fixed_costs_mda_year=fixed_costs_mda_year,
        freq=freq,
        base_year=base_year,
    )

    clinical_offsets = []
    staff_value = []
    staff_value_included = []
    total_offsets = []
    staff_hours = []
    cum_staff_hours = []
    net_budget = []
    running_staff_hours = 0.0
    for idx, row in df.iterrows():
        year_idx = int(idx) + 1
        disc = (1.0 + float(disc_rate)) ** year_idx
        clinical = float(hs_savings_pa) / disc
        staff_val = float(staff_time_value_pa) / disc
        staff_included = staff_val if include_staff_time_value else 0.0
        offset = clinical + staff_included
        net = float(row["Total_prog_cost_USD"]) - offset
        hours = float(staff_hours_saved_pa)
        running_staff_hours += hours
        clinical_offsets.append(clinical)
        staff_value.append(staff_val)
        staff_value_included.append(staff_included)
        total_offsets.append(offset)
        staff_hours.append(hours)
        cum_staff_hours.append(running_staff_hours)
        net_budget.append(net)

    df["Clinical_management_budget_offset_USD"] = clinical_offsets
    df["Staff_time_hours_saved"] = staff_hours
    df["Cumulative_staff_time_hours_saved"] = cum_staff_hours
    df["Staff_time_value_USD"] = staff_value
    df["Staff_time_value_included_USD"] = staff_value_included
    df["Health_sector_budget_offset_USD"] = total_offsets
    df["Net_budget_impact_USD"] = net_budget
    df["Cumulative_net_budget_impact_USD"] = pd.Series(net_budget, dtype=float).cumsum()
    return df
