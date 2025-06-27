from pathlib import Path

from typing import Optional, Union

import numpy as np
import pandas as pd

from us_congressional_districts.utils import (
    get_data_directory,
    get_state_fips_codes,
)


"""Utilities to pull AGI targets from the IRS SOI data files."""


SOI_COLUMNS = [
    "Under $1",
    "$1 under $10,000",
    "$10,000 under $25,000",
    "$25,000 under $50,000",
    "$50,000 under $75,000",
    "$75,000 under $100,000",
    "$100,000 under $200,000",
    "$200,000 under $500,000",
    "$500,000 or more",
]

AGI_STUB_TO_BAND = {i + 1: band for i, band in enumerate(SOI_COLUMNS)}


AGI_BOUNDS = {
    "Under $1": (-np.inf, 1),
    "$1 under $10,000": (1, 10_000),
    "$10,000 under $25,000": (10_000, 25_000),
    "$25,000 under $50,000": (25_000, 50_000),
    "$50,000 under $75,000": (50_000, 75_000),
    "$75,000 under $100,000": (75_000, 100_000),
    "$100,000 under $200,000": (100_000, 200_000),
    "$200,000 under $500,000": (200_000, 500_000),
    "$500,000 or more": (500_000, np.inf),
}

NON_VOTING_STATES = {"US", "AS", "GU", "MP", "PR", "VI", "DC", "OA"}


def get_code_name_map() -> dict:
    demographics = get_data_directory() / "input" / "demographics"
    age_district = pd.read_csv(demographics / "age_district.csv")
    age_state = pd.read_csv(demographics / "age_state.csv")
    age_national = pd.read_csv(demographics / "age_national.csv")

    for df in [age_district, age_state, age_national]:
        df = df[["GEO_ID", "GEO_NAME"]]

    combined = pd.concat(
        [age_district, age_state, age_national], ignore_index=True
    )
    return combined.set_index("GEO_ID")["GEO_NAME"].to_dict()


code_to_name = get_code_name_map()


def pull_national_soi_variable(
    soi_variable_row: int,  # the national SOI xlsx file has a row for each target variable
    variable_name: Union[str, None],
    is_count: bool,
    national_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Download and save national AGI totals."""
    df = pd.read_excel(
        "https://www.irs.gov/pub/irs-soi/22in54us.xlsx", skiprows=7
    )

    assert (
        np.abs(
            df.iloc[soi_variable_row, 1]
            - df.iloc[soi_variable_row, 2:12].sum()
        )
        < 100
    ), "Row 0 doesn't add up — check the file."

    agi_values = df.iloc[soi_variable_row, 2:12].astype(int).to_numpy()
    agi_values = np.concatenate(
        [agi_values[:8], [agi_values[8] + agi_values[9]]]
    )

    agi_brackets = [
        AGI_STUB_TO_BAND[i] for i in range(1, len(SOI_COLUMNS) + 1)
    ]

    result = pd.DataFrame(
        {
            "GEO_ID": ["0100000US"] * len(agi_brackets),
            "AGI_LOWER_BOUND": [AGI_BOUNDS[b][0] for b in agi_brackets],
            "AGI_UPPER_BOUND": [AGI_BOUNDS[b][1] for b in agi_brackets],
            "VALUE": agi_values,
        }
    )

    result["GEO_NAME"] = result["GEO_ID"].map(code_to_name)
    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "AGI_LOWER_BOUND", "AGI_UPPER_BOUND", "VALUE"]
    ]
    result["IS_COUNT"] = int(is_count)
    result["VARIABLE"] = variable_name

    if national_df is not None:
        # If a DataFrame is passed, we append the new data to it.
        df = pd.concat([national_df, result], ignore_index=True)
        return df

    return result


def pull_state_soi_variable(
    soi_variable_col: str,  # the state SOI csv file has a column for each target variable
    variable_name: Union[str, None],
    is_count: bool,
    state_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Download and save state AGI totals."""
    df = pd.read_csv(
        "https://www.irs.gov/pub/irs-soi/22in55cmcsv.csv", thousands=","
    )

    merged = (
        df[df["AGI_STUB"].isin([9, 10])]
        .groupby("STATE", as_index=False)
        .agg({soi_variable_col: "sum"})
        .assign(AGI_STUB=9)
    )
    df = df[~df["AGI_STUB"].isin([9, 10])]
    df = pd.concat([df, merged], ignore_index=True)
    df = df[df["AGI_STUB"] != 0]

    df["agi_bracket"] = df["AGI_STUB"].map(AGI_STUB_TO_BAND)

    state_fips = get_state_fips_codes()
    df["GEO_ID"] = "0400000US" + df["STATE"].str.lower().map(state_fips)

    result = (
        df.loc[
            ~df["STATE"].isin(NON_VOTING_STATES),  # drop territories + DC
            ["GEO_ID", "agi_bracket", soi_variable_col],
        ]
        .rename(columns={soi_variable_col: "VALUE"})
        .reset_index(drop=True)
    )

    result["AGI_LOWER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][0]
    )
    result["AGI_UPPER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][1]
    )
    result["GEO_NAME"] = result["GEO_ID"].map(code_to_name)

    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "AGI_LOWER_BOUND", "AGI_UPPER_BOUND", "VALUE"]
    ]
    result["IS_COUNT"] = int(is_count)
    result["VARIABLE"] = variable_name

    if state_df is not None:
        # If a DataFrame is passed, we append the new data to it.
        df = pd.concat([state_df, result], ignore_index=True)
        return df

    return result


def pull_district_soi_variable(
    soi_variable_col: str,  # the district SOI csv file has a column for each target variable
    variable_name: Union[str, None],
    is_count: bool,
    district_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Download and save congressional district AGI totals."""
    df = pd.read_csv("https://www.irs.gov/pub/irs-soi/22incd.csv")
    df = df[df["agi_stub"] != 0]

    df["STATEFIPS"] = df["STATEFIPS"].astype(int).astype(str).str.zfill(2)
    df["CONG_DISTRICT"] = (
        df["CONG_DISTRICT"].astype(int).astype(str).str.zfill(2)
    )
    df["GEO_ID"] = "5001800US" + df["STATEFIPS"] + df["CONG_DISTRICT"]

    at_large_states = (
        df.groupby("STATEFIPS")["CONG_DISTRICT"]
        .nunique()
        .pipe(lambda s: s[s == 1].index)
    )
    df = (
        df.loc[
            (df["CONG_DISTRICT"] != "00")
            | (df["STATEFIPS"].isin(at_large_states))
        ]
        .query("STATEFIPS != '11'")  # drop DC
        .reset_index(drop=True)
    )
    assert df["GEO_ID"].nunique() == 435

    df["agi_bracket"] = df["agi_stub"].map(AGI_STUB_TO_BAND)
    result = df[
        ["GEO_ID", "CONG_DISTRICT", "STATE", "agi_bracket", soi_variable_col]
    ].rename(columns={soi_variable_col: "VALUE"})

    result["AGI_LOWER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][0]
    )
    result["AGI_UPPER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][1]
    )
    result["GEO_NAME"] = result["GEO_ID"].map(code_to_name)

    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "AGI_LOWER_BOUND", "AGI_UPPER_BOUND", "VALUE"]
    ]
    result["IS_COUNT"] = int(is_count)
    result["VARIABLE"] = variable_name

    if district_df is not None:
        # If a DataFrame is passed, we append the new data to it.
        df = pd.concat([district_df, result], ignore_index=True)
        return df

    return result


def main() -> None:
    national_agi_count_df = pull_national_soi_variable(
        soi_variable_row=0,  # Row 0 is the total number of returns (count) for AGI brackets
        variable_name="adjusted_gross_income_count",
        is_count=True,
    )
    national_df = pull_national_soi_variable(
        soi_variable_row=17,  # Row 17 is the total Adjusted Gross Income (amount) for AGI brackets
        variable_name="adjusted_gross_income",
        is_count=False,
        national_df=national_agi_count_df,
    )
    out_dir = Path(get_data_directory()) / "input" / "soi"
    out_dir.mkdir(parents=True, exist_ok=True)
    national_df.to_csv(out_dir / "agi_national.csv", index=False)

    state_agi_count_df = pull_state_soi_variable(
        soi_variable_col="N1",  # Column "N1" contains the total number of returns (count) for AGI brackets
        variable_name="adjusted_gross_income_count",
        is_count=True,
    )
    state_df = pull_state_soi_variable(
        soi_variable_col="A00100",  # Column "A00100" contains the the total Adjusted Gross Income (amount) for AGI brackets
        variable_name="adjusted_gross_income",
        is_count=False,
        state_df=state_agi_count_df,
    )
    out_dir = Path(get_data_directory()) / "input" / "soi"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_df.to_csv(out_dir / "agi_state.csv", index=False)

    district_agi_count_df = pull_district_soi_variable(
        soi_variable_col="N1",  # Column "N1" contains the total number of returns (count) for AGI brackets
        variable_name="adjusted_gross_income_count",
        is_count=True,
    )
    district_df = pull_district_soi_variable(
        soi_variable_col="A00100",  # Column "A00100" contains the the total Adjusted Gross Income (amount) for AGI brackets
        variable_name="adjusted_gross_income",
        is_count=False,
        district_df=district_agi_count_df,
    )
    out_dir = Path(get_data_directory()) / "input" / "soi"
    out_dir.mkdir(parents=True, exist_ok=True)
    district_df.to_csv(out_dir / "agi_district.csv", index=False)


if __name__ == "__main__":
    main()
