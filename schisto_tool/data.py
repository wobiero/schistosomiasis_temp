
from __future__ import annotations
import streamlit as st
import pandas as pd
import os
import io
from pathlib import Path
from .cache import cache_data
from .config import DATA_DIR
from PIL import Image
folder_path = os.path.join(os.path.dirname(__file__), 'datasets')
@cache_data
def load_country_inputs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Hourly_PPP(Int$)" not in df.columns and {
        "Annual_PPP(Int$)",
        "Weekly_Work_Hours",
    }.issubset(df.columns):
        df["Hourly_PPP(Int$)"] = df["Annual_PPP(Int$)"] / (df["Weekly_Work_Hours"] * 52)
    return df

@cache_data
def load_inputs(data_dir = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load country economic inputs and ESPEN schistosomiasis data."""
    try:
        from schistosomiasis_data_uploader import load_espen_with_species
    except ImportError:
        from .schistosomiasis_data_uploader import load_espen_with_species

    country_inputs = load_country_inputs(str(data_dir / "df_gdp.csv"))
    espen_schisto = load_espen_with_species(
        str(data_dir / "consolidated_schisto.csv"),
        exclude_non_endemic=True,
    )
    return country_inputs, espen_schisto


@st.cache_data
def load_country_flag(country: str) -> Image.Image | None:
    flags_path = DATA_DIR / "flags"
    file_path = flags_path / f"{country}.png"

    if not file_path.exists():
        return None

    return Image.open(file_path)
