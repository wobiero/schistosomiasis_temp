from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schistosomiasis_data_uploader import (
    REQUIRED_OUTPUT_COLUMNS,
    load_espen_with_species,
    normalize_espen_dataframe,
    prepare_espen_with_species,
    summarize_espen,
    summarize_espen_species,
    validate_espen_dataframe,
    validate_espen_schema,
)


def main() -> None:
    raw = pd.DataFrame(
        {
            "Country": ["Kenya", "Kenya", "Kenya", "Kenya", "Zambia"],
            "Region": ["A", "A", "A", "A", "C"],
            "District": ["B", "B", "C", "C", "D"],
            "IU": ["IU1", "IU2", "IU3", "IU4", "IU5"],
            "Population requiring MDA": [1000, 2000, 1500, 500, 3000],
            "Population Treated": [800, 1000, 900, 0, 2400],
            "SAC Prevalence": [25, "30%", "20%", 0, 0],
            "Adult Prevalence": [10, "12%", "8%", 0, 0],
            "Species": ["S. mansoni", "S. haematobium", "Both species", "Non-endemic", "S. haematobium"],
            "MDA Rounds": [1, 2, 1, 0, 1],
        }
    )

    df = normalize_espen_dataframe(raw, exclude_non_endemic=True)
    assert len(df) == 3, df[["IUs_NAME", "species", "sm_prev_pct", "sh_prev_pct"]]
    assert set(REQUIRED_OUTPUT_COLUMNS).issubset(df.columns)
    assert float(df.loc[0, "sm_prev_pct"]) == 25.0
    assert float(df.loc[1, "sh_prev_pct"]) == 30.0
    assert float(df.loc[2, "sm_prev_pct"]) == 20.0
    assert float(df.loc[2, "sh_prev_pct"]) == 20.0
    assert df.loc[2, "species"] == "both"
    assert 0.0 <= float(df.loc[0, "sm_share_pct"]) <= 100.0
    assert validate_espen_dataframe(df) == []
    assert validate_espen_schema(df)["missing_required_columns"] == []

    country_summary = summarize_espen(df)
    assert not country_summary.empty
    assert float(country_summary["PopReq"].sum()) == 4500.0

    species_summary = summarize_espen_species(df)
    assert not species_summary.empty
    assert float(species_summary["PopReq"].sum()) == 4500.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir) / "raw.csv"
        raw.to_csv(tmp, index=False)
        loaded = load_espen_with_species(tmp)
        assert loaded.shape[0] == df.shape[0]

    raw_specific = pd.DataFrame(
        {
            "ADMIN0": ["Uganda", "Uganda"],
            "ADMIN1": ["X", "X"],
            "ADMIN2": ["Y", "Z"],
            "IUs_NAME": ["Y", "Z"],
            "PopReq": [100, 200],
            "PopTreat": [80, 160],
            "sm_prev_pct": [15, 0],
            "sh_prev_pct": [0, 12],
        }
    )
    specific = prepare_espen_with_species(raw_specific, exclude_non_endemic=True)
    assert specific.loc[0, "sm_prev_pct"] == 15.0
    assert specific.loc[1, "sh_prev_pct"] == 12.0
    assert specific.loc[0, "Prev_Adults"] == specific.loc[0, "Prev_SAC"]

    fraction_input = raw_specific.copy()
    fraction_input["sm_prev_pct"] = [0.15, 0]
    fraction_input["sh_prev_pct"] = [0, 0.12]
    fraction = prepare_espen_with_species(fraction_input, prevalence_scale="fraction")
    assert fraction.loc[0, "sm_prev_pct"] == 15.0
    assert fraction.loc[1, "sh_prev_pct"] == 12.0

    print("data uploader tests passed")


if __name__ == "__main__":
    main()
