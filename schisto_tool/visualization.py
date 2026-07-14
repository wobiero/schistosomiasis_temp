
from __future__ import annotations

from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from .config import DEFAULT_SEED
from .prevalence import build_prevalence_projection_df, project_prevalence_sac, annual_equivalent_frequency_factor, species_trajectory_decay

def plot_prevalence_projections(
    initial_prev_sac: float,
    initial_prev_adult: float,
    time_horizon: int = 10,
    include_oscillations: bool = False,
    coverage_pct: float = 75.0,
    frequency: str = "Annual",
    species: str = "mansoni",
    p_min_pct: float = 0.5,
    target_multiplier: float = 1.0,
    include_no_mda: bool = True,
) -> alt.Chart:
    """Interactive prevalence projection plot for the selected MDA scenario."""
    years = np.linspace(0, time_horizon, int(time_horizon * 4 + 1))
    df = build_prevalence_projection_df(
        years,
        initial_prev_sac,
        initial_prev_adult,
        include_oscillations=include_oscillations,
        coverage_pct=coverage_pct,
        frequency=frequency,
        species=species,
        p_min_pct=p_min_pct,
        target_multiplier=target_multiplier,
        include_no_mda=include_no_mda,
    )

    value_cols = ["Prev_SAC", "Prev_Adult"]
    label_map = {"Prev_SAC": "SAC", "Prev_Adult": "Adult"}
    if float(target_multiplier) > 1.0 and "Prev_Combined" in df.columns:
        value_cols.append("Prev_Combined")
        label_map["Prev_Combined"] = "Effective target population"

    mda_long = df[["Year", *value_cols]].melt(
        "Year", var_name="Group", value_name="Prevalence (%)"
    )
    mda_long["Group"] = mda_long["Group"].map(label_map)
    mda_long["Scenario"] = f"Selected MDA ({coverage_pct:.0f}% {frequency.lower()})"

    frames = [mda_long]
    if include_no_mda:
        no_mda_cols = [f"{col}_no_mda" for col in value_cols if f"{col}_no_mda" in df.columns]
        no_mda_map = {f"{col}_no_mda": label_map[col] for col in value_cols if f"{col}_no_mda" in df.columns}
        no_mda_long = df[["Year", *no_mda_cols]].melt(
            "Year", var_name="Group", value_name="Prevalence (%)"
        )
        no_mda_long["Group"] = no_mda_long["Group"].map(no_mda_map)
        no_mda_long["Scenario"] = "No-MDA comparator"
        frames.append(no_mda_long)

    df_long = pd.concat(frames, ignore_index=True)

    return (
        alt.Chart(df_long)
        .mark_line(point=True, size=2)
        .encode(
            x=alt.X("Year:Q", title="Years from baseline"),
            y=alt.Y("Prevalence (%):Q", title="Prevalence (%)", scale=alt.Scale(zero=True)),
            color=alt.Color("Group:N", title="Population group"),
            strokeDash=alt.StrokeDash("Scenario:N", title="Scenario"),
            tooltip=[
                alt.Tooltip("Year:Q", title="Year from baseline", format=".1f"),
                "Scenario",
                "Group",
                alt.Tooltip("Prevalence (%)", format=".2f"),
            ],
        )
        .properties(
            title="Prevalence projection: selected MDA scenario vs no-MDA comparator",
            height=330,
            width=700,
        )
        .interactive()
    )

def plot_prevalence_trajectory(prevalence_df: pd.DataFrame) -> alt.Chart:
    """Plot no-MDA and MDA prevalence trajectories for SAC, adults, and combined target."""
    if prevalence_df is None or prevalence_df.empty:
        return (
            alt.Chart(pd.DataFrame({"message": ["No prevalence trajectory data available"]}))
            .mark_text(size=14)
            .encode(text="message:N")
            .properties(title="Prevalence trajectory", height=300)
        )

    value_cols = [col for col in ["Prev_SAC", "Prev_Adult", "Prev_Combined"] if col in prevalence_df.columns]
    label_map = {
        "Prev_SAC": "SAC",
        "Prev_Adult": "Adults",
        "Prev_Combined": "Effective target population",
    }
    df_long = prevalence_df[["Year", "Scenario", *value_cols]].melt(
        id_vars=["Year", "Scenario"],
        value_vars=value_cols,
        var_name="Population group",
        value_name="Prevalence (%)",
    )
    df_long["Population group"] = df_long["Population group"].map(label_map).fillna(df_long["Population group"])

    return (
        alt.Chart(df_long)
        .mark_line(point=True, size=2)
        .encode(
            x=alt.X("Year:Q", title="Years from baseline"),
            y=alt.Y("Prevalence (%):Q", title="Prevalence (%)", scale=alt.Scale(zero=True)),
            color=alt.Color("Scenario:N", title="Scenario"),
            strokeDash=alt.StrokeDash("Population group:N", title="Population group"),
            tooltip=[
                alt.Tooltip("Year:Q", title="Year from baseline", format=".0f"),
                "Scenario",
                "Population group",
                alt.Tooltip("Prevalence (%):Q", format=".2f"),
            ],
        )
        .properties(
            title="Prevalence trajectory: selected MDA scenario vs no-MDA counterfactual",
            height=330,
            width=700,
        )
        .interactive()
    )


def plot_health_sector_cost_trajectory(
    hs_savings_df: pd.DataFrame,
) -> alt.Chart:
    """Interactive chart showing health sector costs over time."""
    df_long = hs_savings_df[["Year", "HS_cost_no_mda", "HS_cost_with_mda"]].melt(
        "Year", var_name="Scenario", value_name="Annual cost (USD)"
    )
    df_long["Scenario"] = df_long["Scenario"].map({
        "HS_cost_no_mda": "No MDA",
        "HS_cost_with_mda": "With MDA",
    })
    
    return (
        alt.Chart(df_long)
        .mark_line(point=True, size=2.5)
        .encode(
            x=alt.X("Year:Q", title="Year"),
            y=alt.Y(
                "Annual cost (USD):Q",
                title="Annual health sector cost (discounted)",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color("Scenario:N", title="Scenario"),
            tooltip=[
                "Year",
                "Scenario",
                alt.Tooltip("Annual cost (USD)", format="$,.0f"),
            ],
        )
        .properties(
            title="Health sector cost trajectory: no-MDA vs MDA",
            height=300,
            width=600,
        )
        .interactive()
    )

def plot_health_sector_savings(
    hs_savings_df: pd.DataFrame,
) -> alt.Chart:
    """Stacked area chart showing annual and cumulative savings."""
    bars = (
        alt.Chart(hs_savings_df)
        .mark_bar(color="steelblue", opacity=0.6)
        .encode(
            x=alt.X("Year:Q", title="Year"),
            y=alt.Y(
                "Annual_savings_discounted:Q",
                title="Annual savings (USD)",
                axis=alt.Axis(format="$,.0f"),
            ),
            tooltip=[
                "Year",
                alt.Tooltip("Annual_savings_discounted:Q", format="$,.0f", title="Annual savings"),
            ],
        )
    )
    
    line = (
        alt.Chart(hs_savings_df)
        .mark_line(color="darkgreen", size=3, point=True)
        .encode(
            x=alt.X("Year:Q", title="Year"),
            y=alt.Y(
                "Cumulative_savings_discounted:Q",
                title="Cumulative savings (USD)",
                axis=alt.Axis(format="$,.0f"),
            ),
            tooltip=[
                "Year",
                alt.Tooltip(
                    "Cumulative_savings_discounted:Q",
                    format="$,.0f",
                    title="Cumulative savings",
                ),
            ],
        )
    )
    
    return (
        alt.layer(bars, line)
        .resolve_scale(y="independent")
        .properties(
            title="Health sector cost savings from MDA over time",
            height=350,
            width=600,
        )
        .interactive()
    )

def plot_prevalence_sensitivity(
    coverage_range: List[float],
    frequency_scenarios: Dict[str, float],
    initial_prev: float,
    time_horizon: int = 10,
    p_min_pct: float = 0.5,
    species: str = "mansoni",
    annual_decay_at_reference: float | None = None,
) -> alt.Chart:
    """Plot prevalence trajectories for multiple coverage/frequency scenarios."""
    years = np.linspace(0, time_horizon, int(time_horizon * 2 + 1))
    rows = []
    decay = species_trajectory_decay(species, "sac") if annual_decay_at_reference is None else annual_decay_at_reference

    for freq_name, freq_factor in frequency_scenarios.items():
        delivery_factor = annual_equivalent_frequency_factor(freq_name, freq_factor)
        for cov in coverage_range:
            prev = project_prevalence_sac(
                years,
                initial_prev,
                include_oscillations=False,
                min_prev=p_min_pct,
                coverage_pct=cov,
                frequency=freq_name,
                species=species,
                annual_decay_at_reference=decay,
                frequency_effect_factor=delivery_factor,
            )

            for y, p_val in zip(years, prev):
                rows.append({
                    "Year": y,
                    "Prevalence (%)": p_val,
                    "Coverage": f"{int(cov)}%",
                    "Frequency": freq_name,
                    "Frequency effect factor": delivery_factor,
                })

    df = pd.DataFrame(rows)

    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Year:Q", title="Years from baseline"),
            y=alt.Y("Prevalence (%):Q", title="Prevalence (%)", scale=alt.Scale(zero=True)),
            color=alt.Color("Coverage:N", title="MDA coverage"),
            strokeDash=alt.StrokeDash("Frequency:N", title="Delivery"),
            tooltip=["Year", "Coverage", "Frequency", alt.Tooltip("Frequency effect factor", format=".2f"), alt.Tooltip("Prevalence (%)", format=".2f")],
        )
        .properties(
            title="Coverage- and frequency-adjusted prevalence projections",
            height=350,
            width=700,
        )
        .interactive()
    )

def plot_cases_averted_sensitivity(
    sensitivity_df: pd.DataFrame,
    year: int = 10,
) -> alt.Chart:
    """Bar chart showing cases averted for each coverage/frequency combo at target year."""
    df = sensitivity_df[sensitivity_df["Year"] == year].copy()
    df = df.sort_values(["Frequency", "Coverage (%)"]).reset_index(drop=True)
    df["Scenario"] = df["Coverage (%)"].astype(str) + "% " + df["Frequency"]
    
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Scenario:N", title="Coverage & frequency", sort="-y"),
            y=alt.Y(
                "Cases_averted:Q",
                title="Cases averted",
                axis=alt.Axis(format=",.0f"),
            ),
            color=alt.Color("Frequency:N", title="Delivery"),
            tooltip=[
                "Scenario",
                alt.Tooltip("Cases_averted", format=",.0f", title="Cases averted"),
                alt.Tooltip("Prevalence (%)", format=".1f"),
            ],
        )
        .properties(
            title=f"Cases averted by year {year}: sensitivity to coverage & frequency",
            height=300,
            width=700,
        )
    )

def plot_cost_effectiveness_sensitivity(
    sensitivity_df: pd.DataFrame,
    year: int = 10,
) -> alt.Chart:
    """Scatter plot: cases averted vs net benefit."""
    df = sensitivity_df[sensitivity_df["Year"] == year].copy()
    
    return (
        alt.Chart(df)
        .mark_circle(size=150, opacity=0.7)
        .encode(
            x=alt.X(
                "Cases_averted:Q",
                title="Cases averted",
                axis=alt.Axis(format=",.0f"),
            ),
            y=alt.Y(
                "Net_benefit_USD:Q",
                title="Net benefit (HS savings - programme cost)",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color("Coverage (%):Q", scale=alt.Scale(scheme="viridis")),
            shape=alt.Shape("Frequency:N", title="Delivery"),
            tooltip=[
                alt.Tooltip("Coverage (%)", format=".0f"),
                "Frequency",
                alt.Tooltip("Cases_averted", format=",.0f"),
                alt.Tooltip("Net_benefit_USD", format="$,.0f"),
                alt.Tooltip("Prevalence (%)", format=".1f"),
            ],
        )
        .properties(
            title=f"Cost-effectiveness landscape (year {year}): cases averted vs net benefit",
            height=350,
            width=600,
        )
        .interactive()
    )

def plot_ceac(ceac_wtp: np.ndarray, ceac_prob: np.ndarray, cet: float) -> alt.Chart:
    df = pd.DataFrame(
        {
            "WTP (USD/DALY)": ceac_wtp,
            "Probability cost-effective": ceac_prob,
        }
    )
    base = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("WTP (USD/DALY):Q", title="Willingness-to-pay threshold (USD per DALY averted)"),
            y=alt.Y("Probability cost-effective:Q", axis=alt.Axis(format="%"), title="Probability cost-effective"),
            tooltip=["WTP (USD/DALY)", alt.Tooltip("Probability cost-effective", format=".1%")],
        )
        .properties(title="Cost-effectiveness acceptability curve (CEAC)", height=300)
    )
    threshold_line = alt.Chart(pd.DataFrame({"WTP": [float(cet)]})).mark_rule(strokeDash=[6, 3]).encode(x="WTP:Q")
    return (base + threshold_line).interactive()

def compute_confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    n_std: Optional[float] = None,
    n_points: int = 100,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Compute confidence-ellipse coordinates for a bivariate PSA cloud.

    The default confidence=0.95 uses sqrt(chi2.ppf(0.95, df=2)), which is
    approximately 2.45 standard deviations for a bivariate normal distribution.
    """
    x_arr = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]

    if x_arr.size < 3 or y_arr.size < 3:
        return pd.DataFrame(columns=["order", "x", "y"])

    if n_std is None:
        confidence = float(np.clip(confidence, 1e-6, 1.0 - 1e-6))
        n_std = float(np.sqrt(chi2.ppf(confidence, df=2)))
    else:
        n_std = float(n_std)

    cov = np.cov(x_arr, y_arr)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return pd.DataFrame(columns=["order", "x", "y"])

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    if np.allclose(eigenvalues, 0.0):
        return pd.DataFrame(columns=["order", "x", "y"])

    theta = np.linspace(0.0, 2.0 * np.pi, max(int(n_points), 20))
    radii = n_std * np.sqrt(eigenvalues)
    ellipse = np.vstack((radii[0] * np.cos(theta), radii[1] * np.sin(theta)))
    rotated = eigenvectors @ ellipse

    return pd.DataFrame(
        {
            "order": np.arange(theta.size),
            "x": rotated[0, :] + float(np.mean(x_arr)),
            "y": rotated[1, :] + float(np.mean(y_arr)),
        }
    )

def _padded_domain(lower: float, upper: float, padding: float = 0.10) -> list[float]:
    """Return a padded numeric Altair domain that always has non-zero width."""
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return [-1.0, 1.0]
    if np.isclose(lower, upper):
        half_span = max(abs(lower), 1.0) * padding
        return [lower - half_span, upper + half_span]
    pad = (upper - lower) * padding
    return [lower - pad, upper + pad]

def plot_ce_plane(
    icer_df: pd.DataFrame,
    effectiveness_col: str,
    cost_col: str,
    wtp_threshold: float,
    additional_thresholds: list[float] | None = None,
    include_ellipse: bool = True,
    max_points: int = 5_000,
) -> alt.Chart:
    """Plot a cost-effectiveness plane using PSA draw-level outputs."""
    missing = [col for col in [effectiveness_col, cost_col] if col not in icer_df.columns]
    if missing:
        message = f"Missing CE-plane columns: {', '.join(missing)}"
        return (
            alt.Chart(pd.DataFrame({"message": [message]}))
            .mark_text(size=14)
            .encode(text="message:N")
            .properties(title="Cost-effectiveness plane", height=300)
        )

    cols = [effectiveness_col, cost_col]
    if "draw" in icer_df.columns:
        cols.append("draw")

    icer_data = icer_df[cols].copy().rename(
        columns={effectiveness_col: "effectiveness", cost_col: "cost"}
    )
    icer_data["effectiveness"] = pd.to_numeric(icer_data["effectiveness"], errors="coerce")
    icer_data["cost"] = pd.to_numeric(icer_data["cost"], errors="coerce")
    icer_data = icer_data.replace([np.inf, -np.inf], np.nan).dropna(subset=["effectiveness", "cost"])

    if icer_data.empty:
        return (
            alt.Chart(pd.DataFrame({"message": ["No valid CE-plane draws to plot"]}))
            .mark_text(size=14)
            .encode(text="message:N")
            .properties(title="Cost-effectiveness plane", height=300)
        )

    wtp_threshold = float(wtp_threshold)
    icer_data["nqb"] = icer_data["effectiveness"] * wtp_threshold - icer_data["cost"]
    icer_data["cost_effective"] = np.where(icer_data["nqb"] > 0, "Cost-effective", "Not cost-effective")
    prob_ce = float((icer_data["nqb"] > 0).mean())

    chart_points = icer_data
    if len(chart_points) > int(max_points):
        chart_points = chart_points.sample(n=int(max_points), random_state=DEFAULT_SEED).sort_index()

    x_lower = min(0.0, float(icer_data["effectiveness"].min()))
    x_upper = max(0.0, float(icer_data["effectiveness"].max()))
    if np.isclose(x_lower, x_upper):
        x_upper = x_lower + 1.0

    all_thresholds = [wtp_threshold, *(additional_thresholds or [])]
    threshold_y = [x * t for x in (x_lower, x_upper) for t in all_thresholds]
    y_lower = min(0.0, float(icer_data["cost"].min()), *threshold_y)
    y_upper = max(0.0, float(icer_data["cost"].max()), *threshold_y)

    x_domain = _padded_domain(x_lower, x_upper)
    y_domain = _padded_domain(y_lower, y_upper)

    wtp_line = pd.concat([
        pd.DataFrame({
            "effectiveness": [x_domain[0], x_domain[1]],
            "cost": [x_domain[0] * t, x_domain[1] * t],
            "threshold": [f"${t:,.0f}/DALY"] * 2,
        })
        for t in all_thresholds
    ], ignore_index=True)

    x_enc = alt.X(
        "effectiveness:Q",
        title="Effectiveness (DALYs averted)",
        scale=alt.Scale(domain=x_domain),
        axis=alt.Axis(format=",.0f"),
    )
    y_enc = alt.Y(
        "cost:Q",
        title="Incremental cost (USD)",
        scale=alt.Scale(domain=y_domain),
        axis=alt.Axis(format="$,.0f"),
    )

    tooltip = []
    if "draw" in chart_points.columns:
        tooltip.append(alt.Tooltip("draw:Q", title="Draw", format=",.0f"))
    tooltip.extend(
        [
            alt.Tooltip("effectiveness:Q", title="DALYs averted", format=",.1f"),
            alt.Tooltip("cost:Q", title="Incremental cost", format="$,.0f"),
            alt.Tooltip("nqb:Q", title="Net monetary benefit", format="$,.0f"),
            alt.Tooltip("cost_effective:N", title="At threshold"),
        ]
    )

    points = (
        alt.Chart(chart_points)
        .mark_circle(size=30, opacity=0.40, color="steelblue")
        .encode(x=x_enc, y=y_enc, tooltip=tooltip)
    )

    line = (
        alt.Chart(wtp_line)
        .mark_line(strokeDash=[5, 5], color="red", strokeWidth=2)
        .encode(
            x=x_enc, 
            y=y_enc, 
            color=alt.Color("threshold:N", title="WTP threshold"),
            detail="threshold:N",
            tooltip=[alt.Tooltip("threshold:N", title="Threshold")]
        )
    )

    zero_x = (
        alt.Chart(pd.DataFrame({"x": [0.0]}))
        .mark_rule(color="black", strokeWidth=1)
        .encode(x=alt.X("x:Q", scale=alt.Scale(domain=x_domain)))
    )
    zero_y = (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(color="black", strokeWidth=1)
        .encode(y=alt.Y("y:Q", scale=alt.Scale(domain=y_domain)))
    )
    layers = [zero_x, zero_y, points, line]
    if include_ellipse and len(icer_data) >= 3:
        ellipse_df = compute_confidence_ellipse(
            icer_data["effectiveness"].to_numpy(dtype=float),
            icer_data["cost"].to_numpy(dtype=float),
            confidence=0.95,
        )
        if not ellipse_df.empty:
            ellipse = (
                alt.Chart(ellipse_df)
                .mark_line(color="darkred", opacity=0.60, strokeWidth=2)
                .encode(
                    x=alt.X("x:Q", title="Effectiveness (DALYs averted)", scale=alt.Scale(domain=x_domain)),
                    y=alt.Y("y:Q", title="Incremental cost (USD)", scale=alt.Scale(domain=y_domain)),
                    order=alt.Order("order:Q"),
                )
            )
            layers.append(ellipse)

    return (
        alt.layer(*layers)
        .resolve_scale(x="shared", y="shared")
        .properties(
            title=f"Cost-effectiveness plane | P(cost-effective) = {prob_ce:.1%} at ${wtp_threshold:,.0f}/DALY",
            width=500,
            height=500,
        )
        .interactive()
    )

def plot_elimination_trajectory(proj_df, target_threshold_pct, target_year, metric_col):
    long = proj_df[["Year", "Combined_overall_prev_pct", "SAC_heavy_prev_pct"]].melt(
        "Year", var_name="Metric", value_name="Prevalence (%)"
    )
    long["Metric"] = long["Metric"].map({
        "Combined_overall_prev_pct": "Overall (combined)",
        "SAC_heavy_prev_pct": "Heavy-intensity (SAC)",
    })
    lines = (
        alt.Chart(long).mark_line(point=True).encode(
            x=alt.X("Year:Q", axis=alt.Axis(format="d")),
            y=alt.Y("Prevalence (%):Q", scale=alt.Scale(zero=True)),
            color=alt.Color("Metric:N", title="Metric"),
            tooltip=["Year", "Metric", alt.Tooltip("Prevalence (%)", format=".2f")],
        )
    )
    thr = alt.Chart(pd.DataFrame({"y": [float(target_threshold_pct)]})).mark_rule(
        color="red", strokeDash=[6, 3]).encode(y="y:Q")
    yr = alt.Chart(pd.DataFrame({"x": [int(target_year)]})).mark_rule(
        color="green", strokeDash=[4, 4]).encode(x="x:Q")
    return (lines + thr + yr).properties(
        title="Endgame trajectory vs WHO target", height=320, width=650).interactive()

def plot_economic_impact(economic_df: pd.DataFrame) -> alt.Chart:
    """Plot discounted economic costs, cost offsets, and productivity gains."""
    df_long = economic_df[
        ["Year", "Total_prog_cost_USD", "Health_sector_savings_USD", "Economic_gains_USD"]
    ].melt("Year", var_name="Component", value_name="USD")
    label_map = {
        "Total_prog_cost_USD": "Programme cost",
        "Health_sector_savings_USD": "Health-sector savings",
        "Economic_gains_USD": "Productivity gains",
    }
    df_long["Component"] = df_long["Component"].map(label_map)
    bars = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("USD:Q", title="USD (discounted)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Component:N", title="Component"),
            tooltip=["Year", "Component", alt.Tooltip("USD:Q", format="$,.0f")],
        )
    )
    cum_line = (
        alt.Chart(economic_df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("Year:O"),
            y=alt.Y(
                "Cumulative_net_economic_benefit_USD:Q",
                title="Cumulative net economic benefit (USD)",
                axis=alt.Axis(format="$,.0f"),
            ),
            tooltip=["Year", alt.Tooltip("Cumulative_net_economic_benefit_USD:Q", format="$,.0f")],
        )
    )
    return (
        alt.layer(bars, cum_line)
        .resolve_scale(y="independent")
        .properties(title="Economic impact analysis: discounted costs and societal benefits", height=320)
        .interactive()
    )


def plot_budget_impact(budget_df: pd.DataFrame) -> alt.Chart:
    """Plot programme costs and morbidity-state health-sector budget offsets."""
    component_cols = ["Total_prog_cost_USD"]
    if "Clinical_management_budget_offset_USD" in budget_df.columns:
        component_cols.append("Clinical_management_budget_offset_USD")
    else:
        component_cols.append("Health_sector_budget_offset_USD")
    if (
        "Staff_time_value_included_USD" in budget_df.columns
        and pd.to_numeric(budget_df["Staff_time_value_included_USD"], errors="coerce").fillna(0.0).abs().sum() > 0
    ):
        component_cols.append("Staff_time_value_included_USD")
    df_long = budget_df[["Year", *component_cols]].melt("Year", var_name="Component", value_name="USD")
    label_map = {
        "Total_prog_cost_USD": "Gross programme cost",
        "Clinical_management_budget_offset_USD": "Clinical-management offset",
        "Health_sector_budget_offset_USD": "Health-sector budget offset",
        "Staff_time_value_included_USD": "Staff time value included",
    }
    df_long["Component"] = df_long["Component"].map(label_map)
    bars = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("USD:Q", title="USD (discounted)", axis=alt.Axis(format="$,.0f")),
            color=alt.Color("Component:N", title="Component"),
            tooltip=["Year", "Component", alt.Tooltip("USD:Q", format="$,.0f")],
        )
    )
    cum_line = (
        alt.Chart(budget_df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("Year:O"),
            y=alt.Y(
                "Cumulative_net_budget_impact_USD:Q",
                title="Cumulative net budget impact (USD)",
                axis=alt.Axis(format="$,.0f"),
            ),
            tooltip=["Year", alt.Tooltip("Cumulative_net_budget_impact_USD:Q", format="$,.0f")],
        )
    )
    return (
        alt.layer(bars, cum_line)
        .resolve_scale(y="independent")
        .properties(title="Budget impact analysis: programme budget, morbidity offsets, and staff time", height=320)
        .interactive()
    )


def plot_bia(bia_df: pd.DataFrame) -> alt.Chart:
    """Backward-compatible chart wrapper.

    Tables with productivity-gain columns are economic-impact outputs; strict
    budget-impact tables use health-sector budget offsets and no productivity.
    """
    if "Economic_gains_USD" in bia_df.columns:
        return plot_economic_impact(bia_df)
    return plot_budget_impact(bia_df)

def plot_daly_breakdown(daly_df: pd.DataFrame) -> alt.Chart:
    sub = daly_df[daly_df["Outcome"] != "Total"].copy()
    df_long = sub.melt(
        id_vars=["Outcome"],
        value_vars=["No-MDA mean", "MDA mean"],
        var_name="Scenario",
        value_name="DALYs",
    )
    return (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X("DALYs:Q", title="DALYs per year", axis=alt.Axis(format=",.0f")),
            y=alt.Y("Outcome:N", sort="-x"),
            color=alt.Color("Scenario:N"),
            tooltip=["Outcome", "Scenario", alt.Tooltip("DALYs:Q", format=",.0f")],
        )
        .properties(title="Annual DALYs by outcome: no-MDA vs MDA scenario", height=280)
    )
