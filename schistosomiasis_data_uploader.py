"""ESPEN schistosomiasis data preparation helpers.

The Streamlit app imports ``load_espen_with_species`` to read
``datasets/consolidated_schisto.csv``. This module is intentionally free of
Streamlit imports so it can also be used in tests, notebooks, and command-line
QA checks.

The prepared dataframe keeps the original source columns and adds the canonical
columns used by the app:

    ADMIN0, ADMIN1, ADMIN2, IUs_NAME,
    PopReq, PopTreat, Prev_SAC, Prev_Adults, Sch_MDA_Rounds,
    sh_prev_pct, sm_prev_pct, sh_share, sm_share, sh_share_pct, sm_share_pct,
    species, species_status, species_source
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

PrevalenceScale = Literal["auto", "percent", "fraction"]
UnknownSpeciesStrategy = Literal["both", "none", "mansoni", "haematobium"]

GEO_COLS = ("ADMIN0", "ADMIN1", "ADMIN2", "IUs_NAME")
REQUIRED_OUTPUT_COLUMNS = (
    "ADMIN0",
    "ADMIN1",
    "ADMIN2",
    "IUs_NAME",
    "PopReq",
    "PopTreat",
    "Prev_SAC",
    "Prev_Adults",
    "Sch_MDA_Rounds",
    "sh_prev_pct",
    "sm_prev_pct",
    "sh_share",
    "sm_share",
    "sh_share_pct",
    "sm_share_pct",
    "species",
    "species_source",
)

TEXT_DEFAULTS = {
    "ADMIN0": "Unknown country",
    "ADMIN1": "Unknown ADMIN1",
    "ADMIN2": "Unknown ADMIN2",
    "IUs_NAME": "Unknown IU",
}

COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "ADMIN0": (
        "ADMIN0", "admin0", "admin_0", "country", "country_name", "COUNTRY",
        "name_0", "adm0_name", "admin0_name",
    ),
    "ADMIN1": (
        "ADMIN1", "admin1", "admin_1", "province", "region", "state",
        "county", "adm1", "adm1_name", "admin1_name", "name_1",
    ),
    "ADMIN2": (
        "ADMIN2", "admin2", "admin_2", "district", "subcounty", "sub_county",
        "lga", "adm2", "adm2_name", "admin2_name", "name_2",
    ),
    "IUs_NAME": (
        "IUs_NAME", "ius_name", "iu_name", "iu", "ius", "implementation_unit",
        "implementation unit", "implementation_unit_name", "implementing_unit",
        "implementing unit", "implementing_unit_name", "unit_name", "iu_name_clean",
    ),
    "PopReq": (
        "PopReq", "popreq", "pop_req", "population_requiring_mda",
        "population requiring mda", "population_requiring_pc", "population requiring pc",
        "population_required_mda", "population_required", "population required",
        "pop_required", "target_population", "target population", "population_at_risk",
        "at_risk_population", "pop_at_risk", "sac_population_requiring_mda",
        "sac_pop_req", "sac_population", "sac population",
    ),
    "PopTreat": (
        "PopTreat", "poptreat", "pop_treat", "population_treated",
        "population treated", "treated_population", "pop_treated", "number_treated",
        "n_treated", "people_treated", "treated",
    ),
    "Prev_SAC": (
        "Prev_SAC", "prev_sac", "sac_prev", "sac_prevalence", "sac prevalence",
        "prevalence_sac", "sac_prevalence_pct", "sac_prev_pct", "prevalence",
        "prev", "prev_pct", "prevalence_pct", "sch_prev_sac", "sch prevalence",
        "schisto_prevalence", "baseline_prevalence",
    ),
    "Prev_Adults": (
        "Prev_Adults", "prev_adults", "adult_prev", "adults_prev",
        "adult_prevalence", "adult prevalence", "prevalence_adults",
        "adults_prevalence", "adult_prevalence_pct", "adult_prev_pct", "prev_adult",
    ),
    "Sch_MDA_Rounds": (
        "Sch_MDA_Rounds", "sch_mda_rounds", "mda_rounds", "mda rounds", "rounds",
        "number_mda_rounds", "num_mda_rounds", "mda_round", "rounds_completed",
        "sch_rounds",
    ),
    "species_text": (
        "species", "Species", "schisto_species", "schistosoma_species",
        "parasite_species", "parasite", "species_reported", "endemic_species",
        "disease_species", "disease species",
    ),
}

SM_PREV_ALIASES = (
    "sm_prev_pct", "sm_prev", "sm_prevalence_pct", "sm_sac_prev_pct",
    "s_mansoni_prev", "s_mansoni_prev_pct", "s_mansoni_prevalence",
    "s_mansoni_prevalence_pct", "s. mansoni prevalence", "mansoni_prev",
    "mansoni_prev_pct", "mansoni_prevalence", "mansoni_prevalence_pct",
    "schistosoma_mansoni_prev", "schistosoma_mansoni_prev_pct",
    "schistosoma_mansoni_prevalence", "schistosoma_mansoni_prevalence_pct",
    "sm_sac_prev", "mansoni_sac_prev", "mansoni_sac_prev_pct",
    "s_japonicum_prev", "s_japonicum_prev_pct", "s_japonicum_prevalence",
    "japonicum_prev", "japonicum_prev_pct", "japonicum_prevalence",
    "intestinal_prev", "intestinal_prevalence",
)

SH_PREV_ALIASES = (
    "sh_prev_pct", "sh_prev", "sh_prevalence_pct", "sh_sac_prev_pct",
    "s_haematobium_prev", "s_haematobium_prev_pct", "s_haematobium_prevalence",
    "s_haematobium_prevalence_pct", "s. haematobium prevalence",
    "s_hematobium_prev", "s_hematobium_prev_pct", "s_hematobium_prevalence",
    "s_hematobium_prevalence_pct", "haematobium_prev", "haematobium_prev_pct",
    "haematobium_prevalence", "haematobium_prevalence_pct", "hematobium_prev",
    "hematobium_prev_pct", "hematobium_prevalence", "hematobium_prevalence_pct",
    "schistosoma_haematobium_prev", "schistosoma_haematobium_prev_pct",
    "schistosoma_haematobium_prevalence", "schistosoma_haematobium_prevalence_pct",
    "schistosoma_hematobium_prev", "schistosoma_hematobium_prev_pct",
    "schistosoma_hematobium_prevalence", "schistosoma_hematobium_prevalence_pct",
    "sh_sac_prev", "haematobium_sac_prev", "haematobium_sac_prev_pct",
    "urogenital_prev", "urogenital_prevalence", "urinary_prev", "urinary_prevalence",
)

SM_FLAG_ALIASES = (
    "sm", "s_mansoni", "mansoni", "schistosoma_mansoni", "s_japonicum",
    "japonicum", "mansoni_present", "sm_present", "intestinal_schistosomiasis",
    "intestinal_schisto",
)

SH_FLAG_ALIASES = (
    "sh", "s_haematobium", "s_hematobium", "haematobium", "hematobium",
    "schistosoma_haematobium", "schistosoma_hematobium", "haematobium_present",
    "hematobium_present", "sh_present", "urogenital_schistosomiasis",
    "urinary_schistosomiasis",
)


def _norm_name(name: object) -> str:
    text = str(name).strip().lower().replace("%", " pct ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _find_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lookup: dict[str, str] = {}
    for col in df.columns:
        lookup.setdefault(_norm_name(col), str(col))
    for alias in aliases:
        match = lookup.get(_norm_name(alias))
        if match is not None:
            return match
    return None


def _read_tabular(path_or_buffer: str | Path | Any) -> pd.DataFrame:
    name = getattr(path_or_buffer, "name", None)
    suffix = Path(str(name or path_or_buffer)).suffix.lower() if name or isinstance(path_or_buffer, (str, Path)) else ""
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path_or_buffer)
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path_or_buffer, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path_or_buffer)


def _to_numeric(values: object, default: float = 0.0, index: pd.Index | None = None) -> pd.Series:
    if isinstance(values, pd.Series):
        raw = values.copy()
    else:
        raw = pd.Series(values, index=index)
    if raw.empty:
        return pd.Series(dtype="float64", index=index)
    cleaned = (
        raw.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NA": pd.NA, "N/A": pd.NA})
    )
    out = pd.to_numeric(cleaned, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default))
    return out.astype(float)


def _numeric_column(df: pd.DataFrame, source: str | None, default: float = 0.0) -> pd.Series:
    if source is None:
        return pd.Series([float(default)] * len(df), index=df.index, dtype="float64")
    return _to_numeric(df[source], default=default).reindex(df.index).astype(float)


def _string_column(df: pd.DataFrame, source: str | None, default: str) -> pd.Series:
    if source is None:
        values = pd.Series([default] * len(df), index=df.index, dtype="string")
    else:
        values = df[source].astype("string").fillna("").str.strip().reindex(df.index)
    values = values.mask(values.isna() | values.eq(""), default)
    return values.astype(str)


def _looks_fractional(source_name: str | None, values: pd.Series) -> bool:
    if source_name is None:
        return False
    norm = _norm_name(source_name)
    if any(token in norm for token in ("fraction", "frac", "proportion", "prop")):
        return True
    if any(token in norm for token in ("pct", "percent", "percentage")):
        return False
    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    return bool(not finite.empty and finite.min() >= 0.0 and finite.max() <= 1.0)


def _as_percent(values: object, source_name: str | None, prevalence_scale: PrevalenceScale) -> pd.Series:
    """Return prevalence/coverage values in percentage points on a 0-100 scale.

    Accepts columns encoded as 25, "25%", or 0.25. In auto mode, columns with
    fractional-looking names or all values <=1 are converted as fractions. If a
    column mixes explicit percent strings and fractional decimals, the fractional
    rows are converted row-wise.
    """
    raw = values.copy() if isinstance(values, pd.Series) else pd.Series(values)
    had_percent_sign = raw.astype("string").fillna("").str.contains("%", regex=False)
    out = _to_numeric(raw, default=0.0).astype(float)
    if prevalence_scale == "fraction" or (prevalence_scale == "auto" and _looks_fractional(source_name, out)):
        out = out * 100.0
    elif prevalence_scale == "auto":
        if bool(had_percent_sign.any()):
            fractional_without_sign = (~had_percent_sign) & out.gt(0.0) & out.le(1.0)
            out.loc[fractional_without_sign] = out.loc[fractional_without_sign] * 100.0
    elif prevalence_scale == "percent":
        pass
    else:
        raise ValueError("prevalence_scale must be one of: auto, percent, fraction")
    return out.clip(lower=0.0, upper=100.0)


def _boolean_from_column(df: pd.DataFrame, source: str | None) -> pd.Series:
    if source is None:
        return pd.Series([False] * len(df), index=df.index, dtype=bool)
    raw = df[source]
    numeric = pd.to_numeric(raw, errors="coerce")
    out = pd.Series([False] * len(df), index=df.index, dtype=bool)
    numeric_mask = numeric.notna()
    out.loc[numeric_mask] = numeric.loc[numeric_mask].astype(float) > 0
    text = raw.astype("string").fillna("").str.strip().str.lower()
    true_values = {"true", "t", "yes", "y", "present", "positive", "pos", "endemic", "1"}
    out.loc[~numeric_mask] = text.loc[~numeric_mask].isin(true_values)
    return out


def _species_flags_from_text(df: pd.DataFrame, source: str | None) -> tuple[pd.Series, pd.Series, pd.Series]:
    has_sm = pd.Series([False] * len(df), index=df.index, dtype=bool)
    has_sh = pd.Series([False] * len(df), index=df.index, dtype=bool)
    has_text = pd.Series([False] * len(df), index=df.index, dtype=bool)
    if source is None:
        return has_sm, has_sh, has_text
    text = df[source].astype("string").fillna("").str.lower()
    text = text.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    has_text = text.ne("")
    mixed = text.str.contains(r"\b(?:both|mixed|coinfect|co infect|coendemic|co endemic|dual)\b", regex=True)
    has_sm = text.str.contains(r"\b(?:mansoni|japonicum|intestinal|sm|s m)\b", regex=True) | mixed
    has_sh = text.str.contains(r"\b(?:haematobium|hematobium|urogenital|urinary|sh|s h)\b", regex=True) | mixed
    return has_sm.astype(bool), has_sh.astype(bool), has_text.astype(bool)


def _weighted_average(values: pd.Series, weights: pd.Series, fallback: float = 0.5) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    wts = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    mask = vals.notna() & (wts > 0)
    if mask.any() and float(wts.loc[mask].sum()) > 0:
        return float((vals.loc[mask] * wts.loc[mask]).sum() / wts.loc[mask].sum())
    vals = vals.dropna()
    return float(vals.mean()) if not vals.empty else float(fallback)


def _first_non_empty(values: pd.Series) -> object:
    text = values.astype("string").fillna("").str.strip()
    non_empty = text[text.ne("")]
    return non_empty.iloc[0] if not non_empty.empty else np.nan


def _population_max(values: pd.Series) -> float:
    vals = _to_numeric(values, default=0.0)
    vals = vals[vals > 0]
    return float(vals.max()) if not vals.empty else 0.0


def _prevalence_max(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.max()) if not vals.empty else 0.0


def _infer_species(sm_prev: pd.Series, sh_prev: pd.Series, pop_req: pd.Series | None = None) -> pd.Series:
    has_sm = sm_prev > 0
    has_sh = sh_prev > 0
    species = pd.Series(["non_endemic"] * len(sm_prev), index=sm_prev.index, dtype="object")
    species.loc[has_sm & has_sh] = "both"
    species.loc[has_sm & ~has_sh] = "mansoni"
    species.loc[has_sh & ~has_sm] = "haematobium"
    if pop_req is not None:
        species.loc[(~has_sm & ~has_sh) & (pop_req > 0)] = "unspecified_endemic"
    return species


def _derive_species_shares(out: pd.DataFrame) -> pd.DataFrame:
    total_prev_raw = out["sm_prev_pct"] + out["sh_prev_pct"]
    total_prev = total_prev_raw.replace(0.0, np.nan)
    sm_share = (out["sm_prev_pct"] / total_prev).fillna(0.0).clip(0.0, 1.0)
    sh_share = (out["sh_prev_pct"] / total_prev).fillna(0.0).clip(0.0, 1.0)
    sm_share.loc[(out["sm_prev_pct"] > 0) & (out["sh_prev_pct"] <= 0)] = 1.0
    sh_share.loc[(out["sh_prev_pct"] > 0) & (out["sm_prev_pct"] <= 0)] = 1.0
    sm_share.loc[total_prev_raw <= 0] = 0.0
    sh_share.loc[total_prev_raw <= 0] = 0.0
    out["sm_share"] = sm_share.astype(float)
    out["sh_share"] = sh_share.astype(float)

    country_defaults: dict[str, float] = {}
    for country, sub in out.groupby("ADMIN0", dropna=False):
        species_positive = sub[(sub["sm_prev_pct"] + sub["sh_prev_pct"]) > 0]
        if species_positive.empty:
            default = 0.5
        else:
            default = _weighted_average(species_positive["sm_share"], species_positive["PopReq"], fallback=0.5)
        country_defaults[str(country)] = float(np.clip(default, 0.0, 1.0))
    out["sm_share_pct"] = out["ADMIN0"].map(country_defaults).fillna(0.5).astype(float) * 100.0
    out["sh_share_pct"] = 100.0 - out["sm_share_pct"]
    return out


def _consolidate_duplicate_geographies(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty or not out.duplicated(list(GEO_COLS), keep=False).any():
        return out
    aggregations: dict[str, Any] = {
        "PopReq": _population_max,
        "PopTreat": _population_max,
        "Prev_SAC": _prevalence_max,
        "Prev_Adults": _prevalence_max,
        "Sch_MDA_Rounds": _prevalence_max,
        "sh_prev_pct": _prevalence_max,
        "sm_prev_pct": _prevalence_max,
        "species_source": lambda s: "; ".join(sorted(set(str(x) for x in s.dropna() if str(x).strip()))),
    }
    for col in out.columns:
        if col in GEO_COLS or col in aggregations:
            continue
        aggregations[col] = _first_non_empty
    grouped = out.groupby(list(GEO_COLS), dropna=False, as_index=False).agg(aggregations)
    grouped["species"] = _infer_species(grouped["sm_prev_pct"], grouped["sh_prev_pct"], grouped["PopReq"])
    grouped = _derive_species_shares(grouped)
    return grouped


def prepare_espen_dataframe(
    df: pd.DataFrame,
    *,
    exclude_non_endemic: bool = True,
    prevalence_scale: PrevalenceScale = "auto",
    unknown_species_strategy: UnknownSpeciesStrategy = "both",
    country_name_map: Mapping[str, str] | None = None,
    consolidate_species_rows: bool = True,
) -> pd.DataFrame:
    """Clean a raw ESPEN-style dataframe for the costing tool."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(REQUIRED_OUTPUT_COLUMNS))
    if unknown_species_strategy not in {"both", "none", "mansoni", "haematobium"}:
        raise ValueError("unknown_species_strategy must be one of: both, none, mansoni, haematobium")
    if prevalence_scale not in {"auto", "percent", "fraction"}:
        raise ValueError("prevalence_scale must be one of: auto, percent, fraction")

    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]

    matches = {canonical: _find_column(out, aliases) for canonical, aliases in COLUMN_ALIASES.items()}
    for col in GEO_COLS:
        out[col] = _string_column(out, matches.get(col), TEXT_DEFAULTS[col])
    if country_name_map:
        out["ADMIN0"] = out["ADMIN0"].replace(dict(country_name_map))

    out["PopReq"] = _numeric_column(out, matches.get("PopReq"), 0.0).clip(lower=0.0)
    out["PopTreat"] = _numeric_column(out, matches.get("PopTreat"), 0.0).clip(lower=0.0)
    out["Sch_MDA_Rounds"] = _numeric_column(out, matches.get("Sch_MDA_Rounds"), 0.0).clip(lower=0.0)

    prev_sac_source = matches.get("Prev_SAC")
    prev_adults_source = matches.get("Prev_Adults")
    out["Prev_SAC"] = (
        pd.Series(0.0, index=out.index, dtype="float64")
        if prev_sac_source is None
        else _as_percent(out[prev_sac_source], prev_sac_source, prevalence_scale)
    )
    out["Prev_Adults"] = (
        out["Prev_SAC"].copy()
        if prev_adults_source is None
        else _as_percent(out[prev_adults_source], prev_adults_source, prevalence_scale)
    )

    sm_prev_source = _find_column(out, SM_PREV_ALIASES)
    sh_prev_source = _find_column(out, SH_PREV_ALIASES)
    sm_prev = _as_percent(out[sm_prev_source], sm_prev_source, prevalence_scale) if sm_prev_source else pd.Series(0.0, index=out.index)
    sh_prev = _as_percent(out[sh_prev_source], sh_prev_source, prevalence_scale) if sh_prev_source else pd.Series(0.0, index=out.index)

    sm_flag = _boolean_from_column(out, _find_column(out, SM_FLAG_ALIASES))
    sh_flag = _boolean_from_column(out, _find_column(out, SH_FLAG_ALIASES))
    text_sm, text_sh, has_species_text = _species_flags_from_text(out, matches.get("species_text"))
    sm_evidence = sm_flag | text_sm
    sh_evidence = sh_flag | text_sh

    if sm_prev_source is None:
        sm_prev = sm_prev.mask(sm_evidence, out["Prev_SAC"])
    if sh_prev_source is None:
        sh_prev = sh_prev.mask(sh_evidence, out["Prev_SAC"])

    unknown_positive = (out["Prev_SAC"] > 0) & (sm_prev <= 0) & (sh_prev <= 0) & ~sm_evidence & ~sh_evidence
    if unknown_species_strategy == "both":
        sm_prev = sm_prev.mask(unknown_positive, out["Prev_SAC"])
        sh_prev = sh_prev.mask(unknown_positive, out["Prev_SAC"])
    elif unknown_species_strategy == "mansoni":
        sm_prev = sm_prev.mask(unknown_positive, out["Prev_SAC"])
    elif unknown_species_strategy == "haematobium":
        sh_prev = sh_prev.mask(unknown_positive, out["Prev_SAC"])

    out["sm_prev_pct"] = sm_prev.clip(lower=0.0, upper=100.0).astype(float)
    out["sh_prev_pct"] = sh_prev.clip(lower=0.0, upper=100.0).astype(float)
    species_max = np.maximum(out["sm_prev_pct"], out["sh_prev_pct"])
    out["Prev_SAC"] = pd.Series(np.where(out["Prev_SAC"] > 0, out["Prev_SAC"], species_max), index=out.index).clip(0.0, 100.0)
    out["Prev_Adults"] = pd.Series(np.where(out["Prev_Adults"] > 0, out["Prev_Adults"], out["Prev_SAC"]), index=out.index).clip(0.0, 100.0)

    out["species"] = _infer_species(out["sm_prev_pct"], out["sh_prev_pct"], out["PopReq"])
    species_source = pd.Series("none", index=out.index, dtype="object")
    has_any_direct_prev = (sm_prev_source is not None) or (sh_prev_source is not None)
    if has_any_direct_prev:
        species_source.loc[:] = "species_prevalence_columns"
    else:
        species_source.loc[sm_flag | sh_flag] = "species_flag_columns"
        species_source.loc[~(sm_flag | sh_flag) & has_species_text] = "species_text_column"
    species_source.loc[unknown_positive] = f"general_prevalence_unknown_species_strategy_{unknown_species_strategy}"
    out["species_source"] = species_source

    out = _derive_species_shares(out)
    if consolidate_species_rows:
        out = _consolidate_duplicate_geographies(out)

    if exclude_non_endemic:
        out = out.loc[(out["sm_prev_pct"] > 0) | (out["sh_prev_pct"] > 0)].copy()

    for col in REQUIRED_OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    front = [col for col in REQUIRED_OUTPUT_COLUMNS if col in out.columns]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest].reset_index(drop=True)


def load_espen_with_species(
    path: str | Path | Any,
    *,
    exclude_non_endemic: bool = True,
    prevalence_scale: PrevalenceScale = "auto",
    unknown_species_strategy: UnknownSpeciesStrategy = "both",
    country_name_map: Mapping[str, str] | None = None,
    consolidate_species_rows: bool = True,
) -> pd.DataFrame:
    """Load and prepare an ESPEN schistosomiasis CSV/XLSX for the app."""
    raw = _read_tabular(path)
    return prepare_espen_dataframe(
        raw,
        exclude_non_endemic=exclude_non_endemic,
        prevalence_scale=prevalence_scale,
        unknown_species_strategy=unknown_species_strategy,
        country_name_map=country_name_map,
        consolidate_species_rows=consolidate_species_rows,
    )


def load_uploaded_espen(uploaded_file: Any, **kwargs: Any) -> pd.DataFrame:
    """Prepare a Streamlit-uploaded CSV/XLSX file without importing Streamlit."""
    return load_espen_with_species(uploaded_file, **kwargs)


def normalize_espen_dataframe(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Backward-compatible alias for ``prepare_espen_dataframe``."""
    return prepare_espen_dataframe(df, **kwargs)


def prepare_espen_with_species(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Backward-compatible alias for ``prepare_espen_dataframe``."""
    return prepare_espen_dataframe(df, **kwargs)


def validate_prepared_espen(df: pd.DataFrame) -> dict[str, Any]:
    """Return lightweight validation diagnostics for a prepared ESPEN dataframe."""
    missing = [col for col in REQUIRED_OUTPUT_COLUMNS if col not in df.columns]
    diagnostics: dict[str, Any] = {
        "rows": int(len(df)),
        "missing_required_columns": missing,
        "countries": int(df["ADMIN0"].nunique()) if "ADMIN0" in df.columns else 0,
        "total_popreq": float(pd.to_numeric(df.get("PopReq", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
        "total_poptreat": float(pd.to_numeric(df.get("PopTreat", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
    }
    if "species" in df.columns:
        diagnostics["species_counts"] = df["species"].value_counts(dropna=False).to_dict()
    if "species_source" in df.columns:
        diagnostics["species_source_counts"] = df["species_source"].value_counts(dropna=False).to_dict()
    return diagnostics


def validate_espen_dataframe(df: pd.DataFrame) -> list[str]:
    """Return validation issues as a list of human-readable strings."""
    diagnostics = validate_prepared_espen(df)
    issues: list[str] = []
    missing = diagnostics.get("missing_required_columns", [])
    if missing:
        issues.append("Missing required columns: " + ", ".join(map(str, missing)))
    if diagnostics.get("rows", 0) <= 0:
        issues.append("Prepared ESPEN dataframe has no rows.")
    return issues


def validate_espen_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Backward-compatible alias for validation diagnostics."""
    return validate_prepared_espen(df)


def summarize_espen(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize prepared data by country for quick QA."""
    if df.empty:
        return pd.DataFrame(columns=["ADMIN0", "rows", "PopReq", "PopTreat", "mean_sh_prev_pct", "mean_sm_prev_pct", "haematobium_rows", "mansoni_rows", "both_rows"])
    tmp = df.copy()
    tmp["_has_sh"] = pd.to_numeric(tmp["sh_prev_pct"], errors="coerce").fillna(0.0) > 0
    tmp["_has_sm"] = pd.to_numeric(tmp["sm_prev_pct"], errors="coerce").fillna(0.0) > 0
    tmp["_both"] = tmp["_has_sh"] & tmp["_has_sm"]
    return (
        tmp.groupby("ADMIN0", dropna=False)
        .agg(
            rows=("ADMIN0", "size"),
            PopReq=("PopReq", "sum"),
            PopTreat=("PopTreat", "sum"),
            mean_sh_prev_pct=("sh_prev_pct", "mean"),
            mean_sm_prev_pct=("sm_prev_pct", "mean"),
            haematobium_rows=("_has_sh", "sum"),
            mansoni_rows=("_has_sm", "sum"),
            both_rows=("_both", "sum"),
        )
        .reset_index()
        .sort_values("ADMIN0")
        .reset_index(drop=True)
    )


def summarize_espen_species(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for summarize_espen()."""
    return summarize_espen(df)


def save_prepared_espen(input_path: str | Path, output_path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load, clean, save, and return a prepared ESPEN dataframe."""
    prepared = load_espen_with_species(input_path, **kwargs)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".xlsx", ".xls"}:
        prepared.to_excel(output, index=False)
    else:
        prepared.to_csv(output, index=False)
    return prepared


def write_clean_espen(input_path: str | Path, output_path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Backward-compatible alias for ``save_prepared_espen``."""
    return save_prepared_espen(input_path, output_path, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ESPEN schistosomiasis data for the costing app.")
    parser.add_argument("input", help="Input CSV/XLSX path")
    parser.add_argument("output", nargs="?", help="Optional output CSV/XLSX path")
    parser.add_argument("--output", "-o", dest="output_option", help="Optional output CSV/XLSX path")
    parser.add_argument("--include-non-endemic", action="store_true", help="Keep rows without positive species prevalence")
    parser.add_argument("--prevalence-scale", choices=["auto", "percent", "fraction"], default="auto")
    parser.add_argument("--unknown-species-strategy", choices=["both", "none", "mansoni", "haematobium"], default="both")
    parser.add_argument("--no-consolidate", action="store_true", help="Do not consolidate duplicate geography/species rows")
    parser.add_argument("--head", type=int, default=0, help="Print the first N prepared rows")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    prepared = load_espen_with_species(
        args.input,
        exclude_non_endemic=not args.include_non_endemic,
        prevalence_scale=args.prevalence_scale,
        unknown_species_strategy=args.unknown_species_strategy,
        consolidate_species_rows=not args.no_consolidate,
    )
    print("Prepared ESPEN schistosomiasis data")
    print(validate_prepared_espen(prepared))
    print("\nCountry summary:")
    print(summarize_espen(prepared).to_string(index=False))
    if args.head > 0:
        print(f"\nFirst {args.head} rows:")
        print(prepared.head(args.head).to_string(index=False))
    output_arg = args.output_option or args.output
    if output_arg:
        output = Path(output_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() in {".xlsx", ".xls"}:
            prepared.to_excel(output, index=False)
        else:
            prepared.to_csv(output, index=False)
        print(f"\nWrote: {output}")
    return 0 if not validate_prepared_espen(prepared)["missing_required_columns"] else 1


__all__ = [
    "REQUIRED_OUTPUT_COLUMNS",
    "load_espen_with_species",
    "load_uploaded_espen",
    "prepare_espen_dataframe",
    "prepare_espen_with_species",
    "normalize_espen_dataframe",
    "save_prepared_espen",
    "write_clean_espen",
    "validate_prepared_espen",
    "validate_espen_dataframe",
    "validate_espen_schema",
    "summarize_espen",
    "summarize_espen_species",
]


if __name__ == "__main__":
    raise SystemExit(main())
