# Modular Schistosomiasis Endgame Costing Tool

This refactor splits the original single-file Streamlit app into a small Python package plus a thin `app.py` Streamlit entry point.

## Run

Copy your existing `datasets/` directory next to `app.py`; the data uploader/normalizer is included in this package. Then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Expected directory layout:

```text
schisto_modularized/
  app.py
  schistosomiasis_data_uploader.py
  datasets/
    consolidated_schisto.csv
    df_gdp.csv
  schisto_tool/
    config.py
    utils.py
    data.py
    parameters.py
    caseloads.py
    simulation.py
    economics.py
    prevalence.py
    health_sector.py
    sensitivity.py
    visualization.py
    ui_helpers.py
```

## ESPEN data uploader / normalizer

`schistosomiasis_data_uploader.py` provides the `load_espen_with_species()` function imported by `schisto_tool.data`. It is import-safe, has no Streamlit dependency, and standardizes common ESPEN or consolidated country-programme column variants into the columns used by the app:

```text
ADMIN0, ADMIN1, ADMIN2, IUs_NAME, PopReq, PopTreat, Prev_SAC, Prev_Adults,
Sch_MDA_Rounds, sm_prev_pct, sh_prev_pct, sm_share, sh_share,
sm_share_pct, sh_share_pct, species, species_status, species_source
```

It can derive species-specific prevalence and allocation shares when the source has generic schistosomiasis prevalence plus a species label, species flag, or species-specific prevalence columns. With `exclude_non_endemic=True`, rows are retained only when there is a positive schistosomiasis prevalence signal. When species-specific information is unavailable but generic schistosomiasis prevalence is present, the default strategy keeps the row usable by assigning the generic prevalence to both species and using a mixed-species allocation. Override that behavior with `--unknown-species-strategy mansoni`, `--unknown-species-strategy haematobium`, or `--unknown-species-strategy none` if your source data require a different convention.

To sanity-check or normalize a file outside Streamlit:

```bash
python schistosomiasis_data_uploader.py datasets/consolidated_schisto.csv datasets/consolidated_schisto_prepared.csv
```

For prevalence values stored as fractions rather than percentages, add:

```bash
python schistosomiasis_data_uploader.py datasets/consolidated_schisto.csv datasets/consolidated_schisto_prepared.csv --prevalence-scale fraction
```

## Module map

- `config.py`: version, constants, project/data paths.
- `utils.py`: numerical helpers, interval formatting, widget key helpers, weighted means.
- `data.py`: cached country and ESPEN data loading.
- `parameters.py`: `MansoniInputs` and `HaematobiumInputs` dataclasses.
- `caseloads.py`: baseline caseload and species-allocation logic.
- `simulation.py`: Monte Carlo draw engines.
- `economics.py`: DALYs, CEA, ROI, productivity, health-sector costs, economic-impact calculations, and budget-impact calculations with morbidity-state clinical-management offsets and staff time savings but no societal productivity gains.
- `prevalence.py`: SCHISTOX-style prevalence and caseload projections.
- `health_sector.py`: utilization-based health-sector cost projections.
- `sensitivity.py`: coverage/frequency sensitivity calculations.
- `visualization.py`: Altair chart builders.
- `ui_helpers.py`: Streamlit-only download/table helpers.
- `app.py`: UI composition only.

## Economic impact vs budget impact

The app separates these perspectives:

- **Economic impact** includes programme costs, direct health-sector savings from the morbidity-state resource model, productivity gains, net benefit, and benefit-cost ratio.
- **Budget impact** is a programme/payer budget view. It excludes productivity gains, ROI, benefit-cost ratios, and other societal benefits. It estimates direct budget offsets separately by morbidity state, including anemia, periportal fibrosis, varices, hematuria, hydronephrosis, FGS, and attributable bladder cancer, and reports staff time saved from avoided OPD visits and IPD bed-days.

## Refactor notes

The computational modules avoid importing Streamlit directly except `ui_helpers.py` and `app.py`. Functions that were previously decorated with `st.cache_data` now use `schisto_tool.cache.cache_data`, which delegates to Streamlit when available and becomes a no-op in non-Streamlit test contexts.
