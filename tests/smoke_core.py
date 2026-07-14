import numpy as np
import pandas as pd

from schistosomiasis_data_uploader import prepare_espen_dataframe, validate_prepared_espen
from schisto_tool.parameters import MansoniInputs, HaematobiumInputs
from schisto_tool.caseloads import (
    estimate_caseloads_mansoni,
    estimate_caseloads_haematobium,
    partitioned_species_defaults,
    effective_prevalence,
)
from schisto_tool.simulation import run_monte_carlo_mansoni, run_monte_carlo_haematobium
from schisto_tool.economics import (
    add_daly_averted_columns,
    build_combined_daly_df,
    add_cost_effectiveness_columns,
    compute_icer,
    combined_benefit_draws,
    roi_summary_from_draws,
    budget_impact_analysis,
    add_budget_impact_intervals,
    combined_morbidity_budget_draws,
)
from schisto_tool.prevalence import project_caseloads_over_time
from schisto_tool.health_sector import project_health_sector_costs
from schisto_tool.elimination import ephp_target, project_elimination, evaluate_target, probability_of_target

raw = pd.DataFrame({
    'country':['Testland','Testland','Testland'],
    'region':['North','North','South'],
    'district':['D1','D2','D3'],
    'iu':['IU1','IU2','IU3'],
    'population requiring mda':[10000,8000,6000],
    'population treated':[7500,5000,4500],
    'sac_prevalence':['20%','0.15','10'],
    'adult_prevalence':[10,0.05,8],
    'species':['mansoni','haematobium','both'],
})
prepared = prepare_espen_dataframe(raw, prevalence_scale='auto')
print('prepared columns ok', validate_prepared_espen(prepared)['missing_required_columns'])
assert len(prepared) == 3
assert prepared['sm_prev_pct'].sum() > 0 and prepared['sh_prev_pct'].sum() > 0

defaults_m = partitioned_species_defaults(prepared, 'mansoni', 0.5)
defaults_h = partitioned_species_defaults(prepared, 'haematobium', 0.5)
print('defaults', defaults_m, defaults_h)
assert defaults_m['pop_req'] > 0 and defaults_h['pop_req'] > 0

m_params = MansoniInputs(at_risk_pop=12000)
h_params = HaematobiumInputs(at_risk_pop=9000, female_fraction=0.5)
cl_m = estimate_caseloads_mansoni(12000, 15, m_params)
cl_h = estimate_caseloads_haematobium(9000, 12, h_params, 0.5)
assert cl_m['infected'] == 1800
assert cl_h['infected'] == 1080

sim_m = add_daly_averted_columns(run_monte_carlo_mansoni(120, 12000, 15, m_params, 0.75, 123))
sim_h = add_daly_averted_columns(run_monte_carlo_haematobium(120, 9000, 12, 0.5, 60, h_params, 0.75, 456))
print('sim shapes', sim_m.shape, sim_h.shape)
assert {'daly_total','daly_total_mda','dalys_averted','opd_total','ipd_total'}.issubset(sim_m.columns)
assert {'daly_total','daly_total_mda','dalys_averted','total_ca','total_ca_mda'}.issubset(sim_h.columns)
assert (sim_m['dalys_averted'] >= 0).all()
assert (sim_h['dalys_averted'] >= 0).all()

combined = build_combined_daly_df(sim_m, sim_h)
ce = add_cost_effectiveness_columns(combined, 25000, 5.0, 30.0)
icer = compute_icer(ce, 25000, 2000, 500, ce['program_cost_minus_savings'])
print('icer keys', icer['total_draws'], icer['positive_daly_draws'], icer['icer_mean'])
assert icer['total_draws'] == 120
assert icer['positive_daly_draws'] > 0

benefits = combined_benefit_draws([sim_m, sim_h], 2.5, 5.0, 30.0)
roi = roi_summary_from_draws(benefits['hs_savings_pa'], benefits['econ_gain_pa'], 25000)
assert np.isfinite(roi['roi_mean'])

bdraws = combined_morbidity_budget_draws([sim_m, sim_h], 5.0, 30.0)
budget = budget_impact_analysis(1000, 500, 5, 0.03, 0.08, 15000, 6, 0.5, 10000)
budget_i = add_budget_impact_intervals(budget, bdraws['clinical_budget_offset_pa'], 0.03, bdraws['staff_time_value_pa'], bdraws['staff_hours_saved_pa'])
assert 'Cumulative_net_budget_impact_USD_95CI_upper' in budget_i.columns

proj = project_caseloads_over_time(np.arange(0, 6), 12000, 15, 8, estimate_caseloads_mansoni, m_params, target_multiplier=1.5)
hs_proj = project_health_sector_costs(proj, sim_m, 5.0, 30.0)
assert len(hs_proj) == 6

elim = project_elimination(species='mansoni', prev_sac_pct=15, prev_adult_pct=8, sac_population=8000, adult_population=4000, heavy_share=0.4, coverage_pct=75, years=6, base_year=2026)
ev = evaluate_target(elim, ephp_target(2030))
pr = probability_of_target('mansoni', 15, 0.4, 75, 'Annual', ephp_target(2030), n_iter=100, seed=99)
print('elim', ev['year_reached'], pr['prob_reached_by_target'])
assert 'prob_reached_by_target' in pr
print('SMOKE_OK')
