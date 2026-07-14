# QA Report - v1.3.0 Prevalence Consistency Release

Date: 2026-07-13

## Purpose

This release addresses reviewer-facing prevalence trajectory concerns identified after the v1.2 deterministic trajectory hardening. The focus is consistency across the Results-tab trajectory, PSA/ICER effect multiplier, sensitivity analysis, and elimination scenario views.

## Changes implemented

1. **Biennial effect factor now drives sensitivity analysis.**
   - The sensitivity panel's "Biennial annual-equivalent effect vs annual" slider is no longer dead.
   - `run_sensitivity_analysis()` and `plot_prevalence_sensitivity()` now pass the frequency effect factor into the prevalence projection.
   - Sensitivity outputs now include `Frequency_effect_factor`.

2. **Biennial representation reconciled across PSA, trajectory, sensitivity, and elimination.**
   - The deterministic prevalence trajectory now uses `annual_equivalent_frequency_factor x selected_coverage / 75%`.
   - Annual MDA uses factor 1.0.
   - Biennial MDA uses the analyst-selected effect-vs-annual factor.
   - Setting biennial to 0.50 reproduces a literal every-other-year rounds assumption.
   - The elimination projection functions now accept and use the same frequency effect factor when the app passes it.

3. **Species-specific trajectory decay added.**
   - `S. mansoni` default annual-equivalent decay at 75% MDA remains 0.305.
   - `S. haematobium` default annual-equivalent decay at 75% MDA is 0.470.
   - Sidebar trajectory controls are now species-specific and remain editable scenario assumptions.

4. **No-MDA annual change uses ordinary discrete compounding.**
   - `P_noMDA(t) = P0 x (1 + g)^t`.
   - This matches what analysts generally expect when they enter an annual percent change.

5. **Sensitivity programme costs are frequency-specific.**
   - Annual sensitivity scenarios use the MDA-year programme cost.
   - Biennial sensitivity scenarios use `(MDA-year programme cost + off-year fixed cost) / 2` when these costs are supplied.

6. **Reviewer caveat added in-app and in the manual.**
   - The Results-tab trajectory is explicitly framed as a time-path preview for prevalence, caseloads, and health-sector costs.
   - PSA DALYs, ICERs, and CEACs are documented as annual steady-state calculations and should not be recalculated from the deterministic trajectory.

## Validation checks run

```bash
python -m py_compile app.py schistosomiasis_data_uploader.py schisto_tool/*.py
PYTHONPATH=. python tests/smoke_core.py
PYTHONPATH=. python -m pytest -q tests
```

## Validation results

- Python compilation: passed.
- Core smoke test: passed.
- Regression/unit tests: 13 passed.

## Added regression coverage

- Biennial frequency effect factor changes deterministic trajectory outputs.
- Biennial sensitivity slider changes prevalence and cases averted.
- Species-specific defaults make haematobium decline faster than mansoni with equal starting prevalence and coverage.
- No-MDA annual change uses discrete annual compounding.
- Sensitivity analysis uses frequency-specific annualized programme costs.

## Manual QA

The DOCX technical manual was updated and rendered to PNG pages plus PDF using the DOCX render workflow. The rendered pages were visually reviewed as a contact sheet and the prevalence assumptions page was inspected at full size.

## Remaining note before deployment

A browser-based Streamlit smoke test is still recommended in the deployment environment, focusing on:

- the species-specific prevalence trajectory controls;
- the main biennial annual-equivalent effect slider;
- the sensitivity biennial effect slider;
- the Results-tab prevalence trajectory table columns; and
- the Elimination Projections tab frequency-effect caption.
