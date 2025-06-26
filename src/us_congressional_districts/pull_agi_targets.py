import requests
from pathlib import Path

import numpy as np
import pandas as pd

from us_congressional_districts.utils import (
    get_data_directory,
    get_state_fips_codes,
)


"""Utilities to pull AGI targets from the IRS SOI data files."""


AGI_RENAME = {
    "Under $1": "under_1",
    "$1 under $10,000": "1_10k",
    "$10,000 under $25,000": "10k_25k",
    "$25,000 under $50,000": "25k_50k",
    "$50,000 under $75,000": "50k_75k",
    "$75,000 under $100,000": "75k_100k",
    "$100,000 under $200,000": "100k_200k",
    "$200,000 under $500,000": "200k_500k",
    "$500,000 or more": "500k_plus",
}

AGI_STUB_MAP = {i + 1: label for i, label in enumerate(AGI_RENAME)}

NON_VOTING_STATES = {"US", "AS", "GU", "MP", "PR", "VI", "DC", "OA"}


def get_code_name_map() -> dict:
    demographics = get_data_directory() / "input" / "demographics"
    age_district = pd.read_csv(demographics / "age_district.csv")
    age_state = pd.read_csv(demographics / "age_state.csv")
    age_national = pd.read_csv(demographics / "age_national.csv")

    for df in [age_district, age_state, age_national]:
        df = df[["GEO_ID", "NAME"]]

    combined = pd.concat(
        [age_district, age_state, age_national], ignore_index=True
    )
    return combined.set_index("GEO_ID")["NAME"].to_dict()


code_to_name = get_code_name_map()


def agi_wide(df_tall: pd.DataFrame, rename_cols: bool = True) -> pd.DataFrame:
    """Return wide-format AGI counts."""
    wide = (
        df_tall.pivot_table(
            index="GEO_ID",
            columns="agi_bracket",
            values="number_of_individuals",
            aggfunc="sum",
        )
        .reindex(columns=AGI_STUB_MAP.values())
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    wide.columns.name = None
    if rename_cols:
        wide = wide.rename(columns=AGI_RENAME)
    return wide


def pull_national_agi(out_dir: Path | None = None) -> pd.DataFrame:
    """Download and save national AGI totals."""
    df = pd.read_excel(
        "https://www.irs.gov/pub/irs-soi/22in54us.xlsx", skiprows=7
    )

    assert (
        np.abs(df.iloc[0, 1] - df.iloc[0, 2:12].sum()) < 100
    ), "Row 0 doesn’t add up — check the file."

    agi_values = df.iloc[0, 2:12].astype(int).to_numpy()
    agi_values = np.concatenate(
        [agi_values[:8], [agi_values[8] + agi_values[9]]]
    )

    out = pd.DataFrame(
        {
            "GEO_ID": "0100000US",
            "agi_bracket": AGI_RENAME.keys(),
            "number_of_individuals": agi_values,
        }
    )[["GEO_ID", "agi_bracket", "number_of_individuals"]]

    result = agi_wide(out)

    result["NAME"] = result["GEO_ID"].map(code_to_name)

    if out_dir is None:
        out_dir = Path(get_data_directory()) / "input" / "soi"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "agi_national.csv", index=False)
    return result


def pull_state_agi(out_dir: Path | None = None) -> pd.DataFrame:
    """Download and save state AGI totals."""
    df = pd.read_csv(
        "https://www.irs.gov/pub/irs-soi/22in55cmcsv.csv", thousands=","
    )

    merged = (
        df[df["AGI_STUB"].isin([9, 10])]
        .groupby("STATE", as_index=False)
        .agg({"N1": "sum"})
        .assign(AGI_STUB=9)
    )
    df = df[~df["AGI_STUB"].isin([9, 10])]
    df = pd.concat([df, merged], ignore_index=True)
    df = df[df["AGI_STUB"] != 0]

    df["agi_bracket"] = df["AGI_STUB"].map(AGI_STUB_MAP)
    state_fips = get_state_fips_codes()
    df["GEO_ID"] = "0400000US" + df["STATE"].str.lower().map(state_fips)
    out_df = df[["STATE", "GEO_ID", "agi_bracket", "N1"]].rename(
        columns={"N1": "number_of_individuals"}
    )
    out_df = out_df.loc[~out_df["STATE"].isin(NON_VOTING_STATES)].reset_index(
        drop=True
    )[["GEO_ID", "agi_bracket", "number_of_individuals"]]

    result = agi_wide(out_df)

    result["NAME"] = result["GEO_ID"].map(code_to_name)

    if out_dir is None:
        out_dir = Path(get_data_directory()) / "input" / "soi"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "agi_state.csv", index=False)
    return result


def pull_district_agi(out_dir: Path | None = None) -> pd.DataFrame:
    """Download and save congressional district AGI totals."""
    df = pd.read_csv("https://www.irs.gov/pub/irs-soi/22incd.csv")
    df = df[df["agi_stub"] != 0]

    df["STATEFIPS"] = df["STATEFIPS"].astype(int).astype(str).str.zfill(2)
    df["CONG_DISTRICT"] = (
        df["CONG_DISTRICT"].astype(int).astype(str).str.zfill(2)
    )
    df["GEO_ID"] = "5001800US" + df["STATEFIPS"] + df["CONG_DISTRICT"]

    df["agi_bracket"] = df["agi_stub"].map(AGI_STUB_MAP)
    df = df[["GEO_ID", "CONG_DISTRICT", "STATE", "agi_bracket", "N1"]].rename(
        columns={"N1": "number_of_individuals"}
    )

    df["state_fips"] = df["GEO_ID"].str[-4:-2]
    df["district"] = df["GEO_ID"].str[-2:]

    at_large_states = (
        df.groupby("state_fips")["district"]
        .nunique()
        .pipe(lambda s: s[s == 1].index)
    )
    clean_df = (
        df.loc[
            (df["district"] != "00") | (df["state_fips"].isin(at_large_states))
        ]
        .query("state_fips != '11'")
        .reset_index(drop=True)
    )
    assert clean_df["GEO_ID"].nunique() == 435

    out_df = clean_df[["GEO_ID", "agi_bracket", "number_of_individuals"]]
    result = agi_wide(out_df)

    result["NAME"] = result["GEO_ID"].map(code_to_name)

    if out_dir is None:
        out_dir = Path(get_data_directory()) / "input" / "soi"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "agi_district.csv", index=False)
    return result


def main() -> None:
    pull_national_agi()
    pull_state_agi()
    pull_district_agi()


if __name__ == "__main__":
    main()
