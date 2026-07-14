from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import APP_VERSION, DOCS_DIR

MANUAL_PDF = "Schistosomiasis_Endgame_Costing_Tool_Technical_Manual.pdf"
MANUAL_DOCX = "Schistosomiasis_Endgame_Costing_Tool_Technical_Manual.docx"

HELP_TEXT: dict[str, str] = {
    "country": "Country-level economic inputs and ESPEN schistosomiasis rows are filtered to this country.",
    "disease_module": "Choose the species module to run. Select both species when the selected geography has intestinal and urogenital schistosomiasis.",
    "both_species_split": "For rows marked as both species, this assigns the population denominator between the mansoni and haematobium modules. Species-specific prevalence columns are still used for prevalence defaults.",
    "admin1": "Select national level or a first administrative unit. Defaults and denominators update to the selected geography.",
    "admin2": "Select all of the ADMIN1 unit or a specific second administrative unit.",
    "iu": "Select all of the ADMIN2 unit or one implementing unit used for programme delivery.",
    "mda_coverage": "Percentage of the target population expected to receive praziquantel in each delivered treatment round. This changes both programme cost and MDA effect size.",
    "mda_frequency": "Annual assumes one MDA round every year. Biennial assumes treatment every other year and applies the selected relative effectiveness factor.",
    "mda_target": "SAC only uses the SAC proxy population. SAC + at-risk adults multiplies the SAC denominator to approximate the broader treatment population.",
    "target_multiplier": "Multiplier applied to the SAC proxy denominator when adults are included. For example, 1.5 means total target population equals 1.5 times the SAC denominator.",
    "disc_costs": "Annual discount rate applied to programme costs, health-sector costs, and budget/economic impact projections.",
    "disc_effects": "Annual discount rate applied to health effects and productivity gains over the analysis horizon.",
    "time_horizon": "Number of years over which discounted cost-effectiveness and ROI outcomes are summarized.",
    "bia_horizon": "Number of years in the economic-impact and budget-impact projection tables.",
    "base_year": "Calendar year assigned to year 1 of the impact-analysis projection.",
    "biennial_effect": "Annual-equivalent effect of biennial MDA compared with annual MDA. The same factor is used in the PSA effect multiplier, deterministic trajectory, and sensitivity analysis. A value of 70% means biennial delivery is assumed to achieve 70% of the annual effect; use 50% for a literal every-other-year rounds assumption.",
    "psa_iterations": "Number of Monte Carlo draws. Higher values give smoother uncertainty intervals but take longer to run.",
    "seed": "Random seed used to make the probabilistic sensitivity analysis reproducible.",
    "auto_run_psa": "When enabled, the PSA refreshes whenever relevant inputs change. Turn off for slower computers or large iteration counts.",
    "trajectory_horizon": "Number of years shown in the deterministic prevalence trajectory and health-sector cost preview.",
    "trajectory_decay_sac": "Species-specific annual-equivalent exponential prevalence decay parameter for SAC at 75% annual MDA coverage. This is a transparent scenario assumption, not a fitted transmission estimate.",
    "trajectory_decay_adult": "Species-specific annual-equivalent exponential prevalence decay parameter for adults at 75% annual MDA coverage. Adult decay is applied only when adult treatment is selected.",
    "trajectory_floor": "Minimum residual prevalence allowed in the MDA trajectory. The model never increases prevalence up to this floor when baseline prevalence is already lower.",
    "trajectory_no_mda_change": "Annual percentage change in prevalence under the no-MDA comparator. The default is 0%, meaning constant prevalence. Non-zero values use ordinary discrete annual compounding.",
    "trajectory_oscillations": "Optional illustrative short-term oscillations around the MDA trajectory. Keep off for the clearest reviewer-facing base case.",
    "coverage_min": "Lowest MDA coverage scenario included in the sensitivity analysis grid.",
    "coverage_max": "Highest MDA coverage scenario included in the sensitivity analysis grid.",
    "coverage_step": "Increment, in percentage points, between sensitivity-analysis coverage scenarios.",
    "annual_sens": "Include annual MDA scenarios in the sensitivity-analysis comparison.",
    "biennial_sens": "Include biennial MDA scenarios in the sensitivity-analysis comparison. These use the annual-equivalent biennial effect slider, not a hard-coded 0.5 factor.",
    "sens_biennial_effect": "Annual-equivalent effect of biennial MDA in the sensitivity-analysis grid. Changing this slider changes biennial prevalence trajectories, cases averted, net benefit, and charts.",
    "cost_pzq": "Unit procurement cost for one praziquantel tablet in USD.",
    "cost_tablets": "Average number of praziquantel tablets per completed treatment course.",
    "cost_delivery": "Non-drug delivery cost per person treated, such as campaign delivery, community distribution, and logistics.",
    "cost_mapping": "Annual mapping, monitoring, evaluation, or surveillance cost allocated to the selected geography.",
    "cost_training": "Training cost incurred in years when MDA is delivered.",
    "cost_supervision": "Supervision cost incurred in years when MDA is delivered.",
    "cost_other": "Other programme cost not captured by drug, delivery, mapping, training, or supervision categories.",
    "annual_ppp": "Per-capita GDP in purchasing-power-parity terms. Used for income, threshold, and economic interpretation calculations.",
    "q1_share": "Share of national income received by the lowest income quintile. Used to estimate the inequality-adjusted daily wage.",
    "weekly_hours": "Average weekly work hours used to convert annual income to an hourly and daily wage.",
    "life_exp": "Life expectancy used in DALY calculations, especially years of life lost for cancer outcomes.",
    "opd_cost": "Unit cost of one outpatient visit, escalated to current USD inside the tool.",
    "ipd_cost": "Unit cost of one inpatient bed-day, escalated to current USD inside the tool.",
    "staff_minutes_opd": "Staff minutes saved for each outpatient visit averted in the budget-impact module.",
    "staff_minutes_ipd": "Staff minutes saved for each inpatient bed-day averted in the budget-impact module.",
    "staff_hourly_cost": "Optional staff cost per hour used only if monetized staff time is included in net budget impact.",
    "include_staff_value": "Include monetized staff time in the net budget impact. Leave off if OPD/IPD unit costs already include staff costs.",
    "m_prev": "Effective S. mansoni prevalence used for baseline caseload estimation after combining SAC and adult prevalence according to the target population choice.",
    "m_pop": "At-risk population denominator assigned to the S. mansoni module for the selected geography.",
    "m_heavy": "Share of infected people assumed to have heavy-intensity S. mansoni infection.",
    "m_hepatomeg": "Probability that heavy-intensity S. mansoni infection progresses to hepatomegaly.",
    "m_morbidity_red": "Reduction in hepatic morbidity attributable to MDA among affected individuals.",
    "m_cure": "Praziquantel cure rate for S. mansoni infection.",
    "h_prev": "Effective S. haematobium prevalence used for baseline caseload estimation after combining SAC and adult prevalence according to the target population choice.",
    "h_pop": "At-risk population denominator assigned to the S. haematobium module for the selected geography.",
    "h_female": "Female share of the at-risk population, used for female genital schistosomiasis estimates.",
    "h_bg_cancer": "Observed all-cause bladder-cancer incidence rate per 100,000 population. The attributable fraction is estimated internally.",
    "h_morbidity_red": "Reduction in urinary morbidity attributable to MDA among affected individuals.",
    "h_cure": "Praziquantel cure rate for S. haematobium infection.",
    "h_cancer_red": "Reduction in the S. haematobium-attributable component of bladder-cancer risk under MDA.",
    "target_year": "Calendar year against which EPHP or interruption progress is assessed.",
    "reservoir_floor": "Minimum prevalence floor representing residual transmission that may remain despite MDA.",
    "heavy_accel": "Acceleration factor for clearance of heavy-intensity infections relative to overall prevalence.",
    "target_kind": "EPHP uses heavy-intensity SAC prevalence below 1%; interruption uses a near-zero overall prevalence proxy.",
}

GUIDED_TAB_TIPS: dict[str, str] = {
    "country": "Step 1: Review the selected geography, denominator, baseline ESPEN values, and country economic inputs before editing disease assumptions.",
    "disease": "Step 2: Check disease prevalence, at-risk population, morbidity probabilities, and cure-rate assumptions. These are the main epidemiological drivers of burden and benefit.",
    "results": "Step 3: Review DALYs, health-sector costs, productivity gains, ICERs, ROI, and PSA uncertainty. Download the draw-level CSVs when results will be audited.",
    "economic": "Step 4: Use economic impact for a societal view. It includes programme costs, health-sector savings, and productivity gains.",
    "budget": "Step 5: Use budget impact for the payer or health-system affordability view. It excludes productivity gains and reports gross costs, offsets, and net budget impact.",
    "elimination": "Step 6: Use elimination projections to assess whether the selected coverage and frequency are plausibly on track for the selected WHO target year.",
}


def _st():
    import streamlit as st
    return st


def manual_paths() -> dict[str, Path]:
    return {
        "pdf": DOCS_DIR / MANUAL_PDF,
        "docx": DOCS_DIR / MANUAL_DOCX,
    }


def manual_availability() -> dict[str, bool]:
    return {kind: path.exists() for kind, path in manual_paths().items()}


def _download_button(label: str, path: Path, mime: str, key: str) -> None:
    st = _st()
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=key,
        )
    else:
        st.caption(f"{path.name} is not packaged with this deployment.")


def render_manual_downloads(prefix: str = "manual") -> None:
    st = _st()
    paths = manual_paths()
    cols = st.columns(2)
    with cols[0]:
        _download_button(
            "Download full technical manual (PDF)",
            paths["pdf"],
            "application/pdf",
            f"{prefix}_pdf",
        )
    with cols[1]:
        _download_button(
            "Download editable manual (DOCX)",
            paths["docx"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            f"{prefix}_docx",
        )


def render_setup_help() -> None:
    st = _st()
    with st.expander("Setup checklist", expanded=True):
        st.markdown(
            """
            The application needs two CSV files in the `datasets/` directory before the analytical tabs can run:

            - `consolidated_schisto.csv`: ESPEN-style schistosomiasis programme data.
            - `df_gdp.csv`: country economic inputs used for wages, unit costs, and cost-effectiveness thresholds.

            The ESPEN loader standardizes common column aliases into the canonical fields used by the app, including
            `ADMIN0`, `ADMIN1`, `ADMIN2`, `IUs_NAME`, `PopReq`, `PopTreat`, `Prev_SAC`, `Prev_Adults`,
            `Sch_MDA_Rounds`, `sh_prev_pct`, and `sm_prev_pct`.
            """
        )
    render_manual_downloads(prefix="setup_manual")


def render_guided_sidebar(
    *,
    enabled: bool,
    country: str,
    unit_label: str,
    disease_choice: str,
    pop_req_mda: float,
    planned_pop_treat: float,
    mda_coverage: float,
    time_horizon: int,
    bia_horizon: int,
    n_iterations: int,
) -> None:
    if not enabled:
        return
    st = _st()
    with st.sidebar.expander("Guided workflow", expanded=True):
        st.markdown(
            f"""
            **1. Geography**  
            Country: **{country}**  
            Unit: **{unit_label}**

            **2. Disease module**  
            Selected: **{disease_choice}**

            **3. Programme scenario**  
            MDA coverage: **{mda_coverage:.0f}%**  
            Planned population treated: **{planned_pop_treat:,.0f}**

            **4. Analysis settings**  
            Cost-effectiveness horizon: **{int(time_horizon)} years**  
            Impact-analysis horizon: **{int(bia_horizon)} years**  
            PSA draws: **{int(n_iterations):,}**

            **Next:** review the **Disease inputs** tab, then open **Results**.
            """
        )
        st.caption(f"Population requiring MDA in selected unit: {pop_req_mda:,.0f}.")


def render_tab_tip(key: str) -> None:
    st = _st()
    text = GUIDED_TAB_TIPS.get(key)
    if text:
        st.info(text)


def _workflow_table() -> None:
    st = _st()
    st.table(
        [
            {"Step": 1, "Where": "Sidebar", "Action": "Select country, species module, and administrative geography."},
            {"Step": 2, "Where": "Sidebar", "Action": "Set MDA coverage, frequency, target population, horizons, and PSA settings."},
            {"Step": 3, "Where": "Disease inputs", "Action": "Review prevalence, at-risk population, morbidity, cure-rate, and cancer assumptions."},
            {"Step": 4, "Where": "Results", "Action": "Review burden, DALYs, ICERs, ROI, sensitivity analysis, and raw PSA downloads."},
            {"Step": 5, "Where": "Economic impact", "Action": "Review societal economic impact including productivity gains."},
            {"Step": 6, "Where": "Budget impact", "Action": "Review health-system/payer affordability and budget offsets."},
            {"Step": 7, "Where": "Elimination Projections", "Action": "Assess progress toward EPHP or interruption targets."},
        ]
    )


def render_user_guide_tab() -> None:
    st = _st()
    st.header("User Guide")
    st.caption(f"In-app guide for Schistosomiasis Endgame Costing Tool v{APP_VERSION}.")

    st.markdown(
        """
        This guide is intended for programme managers, health economists, analysts, and reviewers who need to
        run or interpret schistosomiasis MDA costing, cost-effectiveness, economic-impact, budget-impact, and
        endgame projections. It complements the full technical manual and keeps the most important instructions
        inside the application.
        """
    )
    render_manual_downloads(prefix="user_guide_manual")

    with st.expander("Quick start", expanded=True):
        _workflow_table()
        st.success(
            "For a first run, keep the default PSA iterations and country economic inputs, review disease defaults, "
            "then use the Results tab to validate whether outputs are plausible before exporting CSVs."
        )

    with st.expander("Required datasets", expanded=False):
        st.markdown(
            """
            The deployed app expects a `datasets/` folder containing:

            - `consolidated_schisto.csv` for ESPEN-style schistosomiasis programme data.
            - `df_gdp.csv` for country-level GDP, inequality, life expectancy, work-hour, inflation, OPD, and IPD inputs.

            The ESPEN preparation helper maps common raw column names to the canonical fields used by the app:
            `ADMIN0`, `ADMIN1`, `ADMIN2`, `IUs_NAME`, `PopReq`, `PopTreat`, `Prev_SAC`, `Prev_Adults`,
            `Sch_MDA_Rounds`, `sh_prev_pct`, `sm_prev_pct`, `species`, and species-share fields.
            """
        )

    with st.expander("Understanding the main outputs", expanded=False):
        st.markdown(
            """
            **DALYs averted** compare annual DALYs under no MDA with DALYs under the selected MDA scenario.

            **ICER** is the incremental cost per DALY averted. In this tool, programme-level ICERs use programme
            cost minus draw-level health-sector savings as the incremental cost.

            **CEAC** shows the probability that the intervention is cost-effective across willingness-to-pay thresholds.

            **ROI** reports annual health-sector savings plus productivity gains per dollar invested.

            **Economic impact** uses a broader societal perspective: programme costs, health-sector savings, and
            productivity gains.

            **Budget impact** uses a payer or health-system affordability perspective: gross programme costs,
            clinical-management offsets, optional staff-time value, and net budget impact. Productivity gains are excluded.

            **Elimination probability** is the proportion of PSA draws that reach the selected EPHP or interruption target
            by the selected year.
            """
        )

    with st.expander("How to interpret common metrics", expanded=False):
        st.markdown(
            """
            - If the ICER is below the Woods opportunity-cost threshold, the scenario is interpreted as cost-effective
              under that threshold.
            - If net monetary benefit is positive at the selected threshold, benefits valued at that threshold exceed net costs.
            - If ROI is greater than 1, estimated health-sector savings plus productivity gains exceed annual programme cost.
            - If net budget impact is positive, the programme requires additional payer budget after clinical offsets.
            - If probability of elimination is low, higher coverage, annual delivery, lower reservoir floor, or longer time horizon
              may be needed, depending on local feasibility.
            """
        )

    with st.expander("Common mistakes and QA checks", expanded=False):
        st.markdown(
            """
            Before using outputs externally, check the following:

            - The selected administrative unit matches the intended decision level.
            - `PopReq` and `PopTreat` are plausible for the geography.
            - Adult treatment multiplier is appropriate when adults are included.
            - Species split for mixed rows is reasonable for co-endemic geographies.
            - Prevalence is in percentage points, not fractions, after data preparation.
            - OPD/IPD unit costs are credible and not double counting staff costs if staff time value is included.
            - PSA intervals are stable enough for reporting; increase iterations if results are noisy.
            """
        )

    with st.expander("Glossary", expanded=False):
        glossary = {
            "BIA": "Budget impact analysis; affordability analysis from payer or health-system perspective.",
            "CEA": "Cost-effectiveness analysis.",
            "CEAC": "Cost-effectiveness acceptability curve.",
            "DALY": "Disability-adjusted life year; measure of disease burden.",
            "EPHP": "Elimination as a public health problem; here, heavy-intensity SAC prevalence below 1%.",
            "ICER": "Incremental cost-effectiveness ratio; incremental cost divided by DALYs averted.",
            "MDA": "Mass drug administration.",
            "NMB": "Net monetary benefit; DALYs averted multiplied by threshold minus incremental cost.",
            "PSA": "Probabilistic sensitivity analysis using repeated Monte Carlo draws.",
            "ROI": "Return on investment; benefits divided by programme cost.",
            "SAC": "School-age children.",
        }
        rows = [{"Term": key, "Meaning": value} for key, value in glossary.items()]
        st.table(rows)

    with st.expander("Troubleshooting", expanded=False):
        st.markdown(
            """
            **No countries appear.** Confirm that `df_gdp.csv` is in the `datasets/` folder and has a `Country` column.

            **A selected country has no rows.** Confirm that country names match between the economic input file and ESPEN file.

            **Prevalence looks 100 times too small or too large.** Confirm whether source prevalence columns are fractions or percentages.
            The loader can auto-detect many cases, but mixed encodings should be reviewed.

            **ICER is NA.** This usually means no PSA draws produced positive DALYs averted.

            **Budget impact differs from economic impact.** That is expected: budget impact excludes productivity gains, while economic impact includes them.
            """
        )
