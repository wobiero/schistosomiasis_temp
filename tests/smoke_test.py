from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schisto_tool.caseloads import estimate_caseloads_haematobium, estimate_caseloads_mansoni
from schisto_tool.economics import (
    add_budget_impact_intervals,
    add_daly_averted_columns,
    add_economic_impact_intervals,
    budget_impact_analysis,
    combined_morbidity_budget_draws,
    compute_icer,
    economic_benefit_cost_ratio_summary,
    economic_impact_analysis,
    morbidity_budget_summary_table,
)
from schisto_tool.parameters import HaematobiumInputs, MansoniInputs
from schisto_tool.prevalence import build_prevalence_projection_df
from schisto_tool.simulation import run_monte_carlo_haematobium, run_monte_carlo_mansoni
from schistosomiasis_data_uploader import load_espen_with_species, prepare_espen_dataframe, validate_prepared_espen


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        espen_path = Path(tmpdir) / "synthetic_espen.csv"
        pd.DataFrame(
            [
                {
                    "Country": "Kenya",
                    "Region": "Nyanza",
                    "District": "Kisumu",
                    "Implementing Unit": "Kisumu East",
                    "Population requiring MDA": 10_000,
                    "Population treated": 8_500,
                    "SAC prevalence": "12.5%",
                    "Adult prevalence": "8%",
                    "MDA rounds": 3,
                    "Species": "S. mansoni",
                },
                {
                    "Country": "Kenya",
                    "Region": "Coast",
                    "District": "Kilifi",
                    "Implementing Unit": "Kilifi North",
                    "Population requiring MDA": 5_000,
                    "Population treated": 4_000,
                    "SAC prevalence": "10%",
                    "Adult prevalence": "5%",
                    "MDA rounds": 2,
                    "Species": "S. haematobium",
                },
                {
                    "Country": "Uganda",
                    "Region": "West Nile",
                    "District": "Arua",
                    "Implementing Unit": "Arua",
                    "Population requiring MDA": 8_000,
                    "Population treated": 6_000,
                    "SAC prevalence": 20,
                    "Adult prevalence": 10,
                    "MDA rounds": 1,
                    "Species": "Both",
                },
                {
                    "Country": "Tanzania",
                    "Region": "Zero",
                    "District": "NoEndemic",
                    "Implementing Unit": "None",
                    "Population requiring MDA": 0,
                    "Population treated": 0,
                    "SAC prevalence": 0,
                    "Adult prevalence": 0,
                    "MDA rounds": 0,
                    "Species": "",
                },
            ]
        ).to_csv(espen_path, index=False)
        espen = load_espen_with_species(espen_path, exclude_non_endemic=True)
        assert len(espen) == 3
        required_cols = {
            "ADMIN0", "ADMIN1", "ADMIN2", "IUs_NAME", "PopReq", "PopTreat",
            "Prev_SAC", "Prev_Adults", "Sch_MDA_Rounds", "sm_prev_pct",
            "sh_prev_pct", "sm_share", "sh_share", "sm_share_pct", "sh_share_pct",
        }
        assert required_cols.issubset(espen.columns)
        assert validate_prepared_espen(espen)["missing_required_columns"] == []
        assert espen.loc[espen["Species"] == "S. mansoni", "sm_prev_pct"].iloc[0] == 12.5
        assert espen.loc[espen["Species"] == "S. haematobium", "sh_prev_pct"].iloc[0] == 10.0
        assert espen["sm_share_pct"].between(0, 100).all()
        assert espen["sh_share_pct"].between(0, 100).all()

    raw_espen = pd.DataFrame({
        "Country": ["Kenya", "Kenya", "Kenya", "Kenya"],
        "Region": ["R1", "R1", "R2", "R2"],
        "District": ["D1", "D2", "D3", "D4"],
        "Population requiring MDA": ["1,000", "2,000", "3,000", "4,000"],
        "Population treated": ["800", "1,200", "1,500", "0"],
        "SAC prevalence": ["20%", "15%", "5", "0"],
        "Adult prevalence": ["10", "5%", "2", "0"],
        "Species": ["S. mansoni", "S. haematobium", "Both species", "Unknown"],
        "MDA rounds": [2, 1, 0, 0],
    })
    prepared_espen = prepare_espen_dataframe(raw_espen, exclude_non_endemic=True)
    assert len(prepared_espen) == 3
    assert {"ADMIN0", "ADMIN1", "ADMIN2", "IUs_NAME", "PopReq", "PopTreat", "Prev_SAC", "Prev_Adults", "sm_prev_pct", "sh_prev_pct", "sm_share", "sh_share", "sm_share_pct", "species", "species_source"}.issubset(prepared_espen.columns)
    assert prepared_espen["sm_prev_pct"].max() > 0
    assert prepared_espen["sh_prev_pct"].max() > 0
    assert validate_prepared_espen(prepared_espen)["missing_required_columns"] == []
    tmp_espen = PROJECT_ROOT / "_tmp_espen_smoke.csv"
    raw_espen.to_csv(tmp_espen, index=False)
    try:
        loaded_espen = load_espen_with_species(tmp_espen, exclude_non_endemic=True)
        assert len(loaded_espen) == 3
    finally:
        tmp_espen.unlink(missing_ok=True)

    m_params = MansoniInputs(at_risk_pop=10_000)
    h_params = HaematobiumInputs(at_risk_pop=10_000)

    m_cases = estimate_caseloads_mansoni(10_000, 20, m_params)
    h_cases = estimate_caseloads_haematobium(10_000, 15, h_params, 0.5)
    assert m_cases["infected"] == 2_000
    assert h_cases["infected"] == 1_500

    prev = build_prevalence_projection_df(np.arange(0, 3), 20, 10)
    assert list(prev.columns) == ["Year", "Prev_SAC", "Prev_Adult"]

    sim_m = add_daly_averted_columns(run_monte_carlo_mansoni(10, 10_000, 20, m_params, 0.75, 42))
    sim_h = add_daly_averted_columns(run_monte_carlo_haematobium(10, 10_000, 15, 0.5, 60, h_params, 0.75, 43))
    assert not sim_m.empty
    assert not sim_h.empty

    icer = compute_icer(sim_m, annual_prog_cost=1_000, annual_ppp=2_000, cet=500)
    assert "icer_mean" in icer

    economic_df = economic_impact_analysis(
        programme_cost_mda_year=1_000,
        off_year_fixed_cost=100,
        hs_savings_pa=200,
        econ_gain_pa=300,
        horizon=3,
        disc_rate=0.03,
        pzq_cost=0.08,
        pop_treat=1_000,
        pzq_per_person=6,
        delivery_c=0.50,
        fixed_costs_mda_year=100,
        freq="Annual",
        base_year=2026,
    )
    economic_df = add_economic_impact_intervals(
        economic_df,
        np.array([200, 210], dtype=float),
        np.array([300, 310], dtype=float),
        0.03,
    )
    assert "Economic_gains_USD" in economic_df.columns
    assert "Total_economic_benefits_USD" in economic_df.columns
    assert "Cumulative_net_economic_benefit_USD" in economic_df.columns
    assert "Cumulative_net_benefit_USD" not in economic_df.columns

    morbidity_budget = combined_morbidity_budget_draws(
        [sim_m, sim_h],
        opd_cost=5.0,
        ipd_cost=25.0,
        health_state_costs={"haematobium_bladder_cancer": 100.0, "mansoni_fibrosis": 10.0},
        opd_staff_minutes=15.0,
        ipd_staff_minutes=120.0,
        staff_hourly_cost=8.0,
        include_staff_time_value=True,
    )
    assert morbidity_budget["clinical_budget_offset_pa"].size > 0
    state_table = morbidity_budget_summary_table(morbidity_budget["state_rows"], 3, 0.03, True)
    assert "Health state" in state_table.columns
    assert "Staff time saved p.a. hours" in state_table.columns
    assert state_table["Health state"].isin(["Anemia", "Periportal fibrosis", "Attributable bladder cancer"]).any()

    budget_df = budget_impact_analysis(
        off_year_fixed_cost=100,
        hs_savings_pa=float(np.nanmean(morbidity_budget["clinical_budget_offset_pa"])),
        horizon=3,
        disc_rate=0.03,
        pzq_cost=0.08,
        pop_treat=1_000,
        pzq_per_person=6,
        delivery_c=0.50,
        fixed_costs_mda_year=100,
        freq="Annual",
        base_year=2026,
        staff_time_value_pa=float(np.nanmean(morbidity_budget["staff_time_value_pa"])),
        staff_hours_saved_pa=float(np.nanmean(morbidity_budget["staff_hours_saved_pa"])),
        include_staff_time_value=True,
    )
    budget_df = add_budget_impact_intervals(
        budget_df,
        morbidity_budget["clinical_budget_offset_pa"],
        0.03,
        staff_time_value_draws_arr=morbidity_budget["staff_time_value_pa"],
        staff_hours_draws_arr=morbidity_budget["staff_hours_saved_pa"],
        include_staff_time_value=True,
    )
    assert "Economic_gains_USD" not in budget_df.columns
    assert "Total_economic_benefits_USD" not in budget_df.columns
    assert "Clinical_management_budget_offset_USD" in budget_df.columns
    assert "Staff_time_hours_saved" in budget_df.columns
    assert "Staff_time_value_included_USD" in budget_df.columns
    assert "Health_sector_budget_offset_USD" in budget_df.columns
    assert "Net_budget_impact_USD" in budget_df.columns
    assert "Cumulative_net_budget_impact_USD" in budget_df.columns

    bcr = economic_benefit_cost_ratio_summary(
        economic_df,
        np.array([200, 210], dtype=float),
        np.array([300, 310], dtype=float),
        0.03,
    )
    assert "bcr_mean" in bcr


if __name__ == "__main__":
    main()
