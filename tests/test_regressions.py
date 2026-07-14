import pandas as pd
import numpy as np

from schisto_tool.caseloads import partitioned_species_defaults, estimate_caseloads_mansoni
from schisto_tool.parameters import MansoniInputs
from schisto_tool.sensitivity import run_sensitivity_analysis


def test_both_species_split_controls_population_allocation():
    df = pd.DataFrame({
        'sh_prev_pct':[10.0],
        'sm_prev_pct':[10.0],
        'PopReq':[1000.0],
        'Prev_SAC':[10.0],
        'Prev_Adults':[5.0],
    })
    for share in [0.1, 0.5, 0.9]:
        m = partitioned_species_defaults(df, 'mansoni', both_species_mansoni_share=share)
        h = partitioned_species_defaults(df, 'haematobium', both_species_mansoni_share=share)
        assert np.isclose(m['pop_req'], 1000.0 * share)
        assert np.isclose(h['pop_req'], 1000.0 * (1.0 - share))


def test_sensitivity_net_benefit_uses_savings_not_residual_cost():
    psa = pd.DataFrame({'infected':[100.0], 'opd_total':[250.0], 'ipd_total':[10.0]})
    df = run_sensitivity_analysis(
        [50, 85], {'Annual':1.0}, 10, 10000, 20,
        estimate_caseloads_mansoni, MansoniInputs(),
        annual_prog_cost=1000, psa_df=psa, opd_cost=5, ipd_cost=30, discount_rate=0.03,
    )
    yr10 = df[df['Year'] == 10].set_index('Coverage (%)')
    assert 'HS_savings_discounted_USD' in yr10.columns
    assert yr10.loc[85, 'Cases_averted'] > yr10.loc[50, 'Cases_averted']
    assert yr10.loc[85, 'HS_savings_discounted_USD'] > yr10.loc[50, 'HS_savings_discounted_USD']
    assert yr10.loc[85, 'Net_benefit_USD'] > yr10.loc[50, 'Net_benefit_USD']


def test_sensitivity_biennial_effect_slider_changes_results():
    psa = pd.DataFrame({'infected':[100.0], 'opd_total':[250.0], 'ipd_total':[10.0]})
    low = run_sensitivity_analysis(
        [75], {'Biennial':0.50}, 10, 10000, 20,
        estimate_caseloads_mansoni, MansoniInputs(),
        annual_prog_cost=1000, psa_df=psa, opd_cost=5, ipd_cost=30, discount_rate=0.03,
    )
    high = run_sensitivity_analysis(
        [75], {'Biennial':1.00}, 10, 10000, 20,
        estimate_caseloads_mansoni, MansoniInputs(),
        annual_prog_cost=1000, psa_df=psa, opd_cost=5, ipd_cost=30, discount_rate=0.03,
    )
    low10 = low[low['Year'] == 10].iloc[0]
    high10 = high[high['Year'] == 10].iloc[0]
    assert low10['Frequency_effect_factor'] == 0.50
    assert high10['Frequency_effect_factor'] == 1.00
    assert high10['Prevalence (%)'] < low10['Prevalence (%)']
    assert high10['Cases_averted'] > low10['Cases_averted']


def test_sensitivity_uses_frequency_specific_annualized_programme_costs():
    psa = pd.DataFrame({'infected':[100.0], 'opd_total':[250.0], 'ipd_total':[10.0]})
    df = run_sensitivity_analysis(
        [75], {'Annual':1.0, 'Biennial':0.70}, 10, 10000, 20,
        estimate_caseloads_mansoni, MansoniInputs(),
        annual_prog_cost=999, psa_df=psa, opd_cost=5, ipd_cost=30, discount_rate=0.03,
        mda_year_prog_cost=1200, off_year_fixed_cost=200,
    )
    year5 = df[df['Year'] == 5].set_index('Frequency')
    assert year5.loc['Annual', 'Annualized_programme_cost_USD'] == 1200
    assert year5.loc['Biennial', 'Annualized_programme_cost_USD'] == 700
