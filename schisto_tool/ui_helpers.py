
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

def download_df(
    df: pd.DataFrame,
    label: str = "Download CSV",
    file_name: str = "results.csv",
) -> None:
    st.download_button(
        label,
        df.to_csv(index=False).encode("utf-8"),
        file_name=file_name,
        mime="text/csv",
    )

def styled_table(df: pd.DataFrame, fmt_cols: Optional[dict] = None) -> None:
    if fmt_cols:
        st.dataframe(df.style.format(fmt_cols), width="stretch")
    else:
        st.dataframe(df, width="stretch")


def format_daly_averted_calculation(icer_res: dict) -> str:
    """Human-readable DALYs-averted calculation for UI captions/help text."""
    return (
        "Computed DALYs averted p.a. = "
        f"{icer_res['daly_total_mean']:,.1f} no-MDA DALYs - "
        f"{icer_res['daly_total_mda_mean']:,.1f} MDA DALYs = "
        f"{icer_res['dalys_averted_point']:,.1f}."
    )
