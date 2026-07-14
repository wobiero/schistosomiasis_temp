import numpy as np

from schisto_tool.prevalence import (
    build_prevalence_scenario_df,
    derive_projection_prevalence_inputs,
    project_mda_prevalence,
)


def _final_combined(df, scenario):
    sub = df[df["Scenario"] == scenario].sort_values("Year")
    return float(sub.iloc[-1]["Prev_Combined"])


def test_trajectory_starts_at_baseline_and_does_not_rise_to_floor():
    years = np.arange(0, 11)
    prev = project_mda_prevalence(
        years,
        initial_prev_pct=0.2,
        coverage_pct=75,
        frequency="Annual",
        annual_decay_at_reference=0.305,
        floor_pct=0.5,
    )
    assert prev[0] == 0.2
    assert np.all(prev <= 0.2 + 1e-12)


def test_higher_coverage_and_annual_frequency_reduce_more():
    years = np.arange(0, 11)
    low = build_prevalence_scenario_df(years, 30, 15, 50, "Biennial", target_multiplier=1.5)
    high = build_prevalence_scenario_df(years, 30, 15, 85, "Annual", target_multiplier=1.5)
    assert _final_combined(high, "MDA scenario") < _final_combined(low, "MDA scenario")
    assert _final_combined(low, "No-MDA counterfactual") == _final_combined(high, "No-MDA counterfactual")


def test_sac_only_does_not_apply_mda_decay_to_adults():
    years = np.arange(0, 6)
    df = build_prevalence_scenario_df(
        years,
        initial_prev_sac=20,
        initial_prev_adult=40,
        coverage_pct=85,
        frequency="Annual",
        target_multiplier=1.5,
        treat_adults=False,
    )
    mda = df[df["Scenario"] == "MDA scenario"].sort_values("Year")
    no = df[df["Scenario"] == "No-MDA counterfactual"].sort_values("Year")
    assert np.allclose(mda["Prev_Adult"].to_numpy(), no["Prev_Adult"].to_numpy())
    assert float(mda.iloc[-1]["Prev_SAC"]) < float(no.iloc[-1]["Prev_SAC"])


def test_effective_prevalence_split_preserves_weighted_prevalence():
    sac, adult = derive_projection_prevalence_inputs(
        effective_prev=24.0,
        default_prev_sac=30.0,
        default_prev_adult=15.0,
        target_multiplier=1.5,
    )
    combined = (sac + adult * 0.5) / 1.5
    assert abs(combined - 24.0) < 1e-9


def test_biennial_frequency_effect_factor_changes_trajectory():
    years = np.arange(0, 11)
    low_effect = build_prevalence_scenario_df(
        years,
        initial_prev_sac=25,
        initial_prev_adult=12,
        coverage_pct=75,
        frequency="Biennial",
        target_multiplier=1.5,
        frequency_effect_factor=0.50,
    )
    high_effect = build_prevalence_scenario_df(
        years,
        initial_prev_sac=25,
        initial_prev_adult=12,
        coverage_pct=75,
        frequency="Biennial",
        target_multiplier=1.5,
        frequency_effect_factor=0.90,
    )
    assert _final_combined(high_effect, "MDA scenario") < _final_combined(low_effect, "MDA scenario")


def test_species_specific_decay_makes_haematobium_decline_faster():
    years = np.arange(0, 11)
    mansoni = build_prevalence_scenario_df(
        years,
        initial_prev_sac=20,
        initial_prev_adult=10,
        coverage_pct=75,
        frequency="Annual",
        target_multiplier=1.5,
        species="mansoni",
        frequency_effect_factor=1.0,
    )
    haematobium = build_prevalence_scenario_df(
        years,
        initial_prev_sac=20,
        initial_prev_adult=10,
        coverage_pct=75,
        frequency="Annual",
        target_multiplier=1.5,
        species="haematobium",
        frequency_effect_factor=1.0,
    )
    assert _final_combined(haematobium, "MDA scenario") < _final_combined(mansoni, "MDA scenario")


def test_no_mda_change_uses_discrete_annual_compounding():
    years = np.array([0, 10])
    df = build_prevalence_scenario_df(
        years,
        initial_prev_sac=20,
        initial_prev_adult=10,
        coverage_pct=75,
        frequency="Annual",
        target_multiplier=1.5,
        no_mda_annual_change_pct=-5.0,
    )
    no = df[df["Scenario"] == "No-MDA counterfactual"].sort_values("Year")
    assert abs(float(no.iloc[-1]["Prev_SAC"]) - 20 * (0.95 ** 10)) < 1e-9


def test_biennial_frequency_effect_factor_changes_trajectory():
    years = np.arange(0, 11)
    low = build_prevalence_scenario_df(
        years, 30, 15, 75, "Biennial", target_multiplier=1.5, frequency_effect_factor=0.50
    )
    high = build_prevalence_scenario_df(
        years, 30, 15, 75, "Biennial", target_multiplier=1.5, frequency_effect_factor=1.00
    )
    assert _final_combined(high, "MDA scenario") < _final_combined(low, "MDA scenario")


def test_species_specific_default_decay_is_faster_for_haematobium():
    years = np.arange(0, 11)
    m = build_prevalence_scenario_df(years, 30, 15, 75, "Annual", target_multiplier=1.5, species="mansoni")
    h = build_prevalence_scenario_df(years, 30, 15, 75, "Annual", target_multiplier=1.5, species="haematobium")
    assert _final_combined(h, "MDA scenario") < _final_combined(m, "MDA scenario")


def test_no_mda_change_uses_discrete_annual_compounding():
    from schisto_tool.prevalence import project_no_mda_prevalence
    years = np.array([0, 10])
    prev = project_no_mda_prevalence(years, 20.0, annual_change_pct=-5.0)
    assert prev[0] == 20.0
    assert abs(prev[1] - 20.0 * (0.95 ** 10)) < 1e-12
