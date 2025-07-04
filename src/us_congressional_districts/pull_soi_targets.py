from pathlib import Path

from typing import Callable, Optional, Union

import logging
import numpy as np
import pandas as pd

from us_congressional_districts.utils import (
    get_data_directory,
    get_state_fips_codes,
)
from us_congressional_districts.pull_geography_ids import get_geography_ids
from us_congressional_districts.district_mapping import (
    get_district_mapping_matrix,
)


"""Utilities to pull AGI targets from the IRS SOI data files."""

logger = logging.getLogger(__name__)

GEO_ID_NAMES = get_geography_ids()
ID_TO_NAME = GEO_ID_NAMES.set_index("GEO_ID")["GEO_NAME"].to_dict()

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

NON_VOTING_STATES = {"US", "AS", "GU", "MP", "PR", "VI", "OA"}  # keep DC

### Add variables and their indices or column names to the dictionaries below to pull them from the SOI files. Make sure to add "_count" at the end of a varaible name if it is a count variable (instead of a total amount variable).

# after skipping the first 7 rows, the national SOI file has targets as row indices:
NATIONAL_VARIABLES = {
    "adjusted_gross_income/count": 0,
    "adjusted_gross_income/amount": 17,
    # "employment_income/count": 20,
    # "employment_income/amount": 21,
    # "self_employment_income/count": 32,
    # "self_employment_income/amount": 33,
    # "qualified_dividend_income/count": 28,
    # "qualified_dividend_income/amount": 29,
    # "taxable_interest_income/count": 22,
    # "taxable_interest_income/amount": 23,
    # "unemployment_compensation/count": 41,
    # "unemployment_compensation/amount": 42,
    # "taxable_pension_income/count": 38,
    # "taxable_pension_income/amount": 39,
    # "real_estate_taxes/count": 75,
    # "real_estate_taxes/amount": 76,
    # "qualified_business_income_deduction/count": 95,
    # "qualified_business_income_deduction/amount": 96,
}

# the state and district SOI file have targets as column names:
GEOGRAPHY_VARIABLES = {
    "adjusted_gross_income/count": "N1",
    "adjusted_gross_income/amount": "A00100",
    # "employment_income/count": "N00200",
    # "employment_income/amount": "A00200",
    # "self_employment_income/count": "N00900",
    # "self_employment_income/amount": "A00900",
    # "qualified_dividend_income/count": "N00650",
    # "qualified_dividend_income/amount": "A00650",
    # "taxable_interest_income/count": "N00300",
    # "taxable_interest_income/amount": "A00300",
    # "unemployment_compensation/count": "N02300",
    # "unemployment_compensation/amount": "A02300",
    # "taxable_pension_income/count": "N01700",
    # "taxable_pension_income/amount": "A01700",
    # "real_estate_taxes/count": "N18500",
    # "real_estate_taxes/amount": "A18500",
    # "qualified_business_income_deduction/count": "N04475",
    # "qualified_business_income_deduction/amount": "A04475",
}


def pull_national_soi_variable(
    soi_variable_ident: int,  # the national SOI xlsx file has a row for each target variable
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
            df.iloc[soi_variable_ident, 1]
            - df.iloc[soi_variable_ident, 2:12].sum()
        )
        < 100
    ), "Row 0 doesn't add up — check the file."

    agi_values = df.iloc[soi_variable_ident, 2:12].astype(int).to_numpy()
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

    result["GEO_NAME"] = ["US"] * len(agi_brackets)
    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "AGI_LOWER_BOUND", "AGI_UPPER_BOUND", "VALUE"]
    ]
    result["IS_COUNT"] = int(is_count)
    result["VARIABLE"] = variable_name

    result["VALUE"] = np.where(
        result["IS_COUNT"] == 0, result["VALUE"] * 1_000, result["VALUE"]
    )

    if national_df is not None:
        # If a DataFrame is passed, we append the new data to it.
        df = pd.concat([national_df, result], ignore_index=True)
        return df

    return result


def pull_state_soi_variable(
    soi_variable_ident: str,  # the state SOI csv file has a column for each target variable
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
        .agg({soi_variable_ident: "sum"})
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
            ~df["STATE"].isin(NON_VOTING_STATES),  # drop territories
            ["GEO_ID", "agi_bracket", soi_variable_ident],
        ]
        .rename(columns={soi_variable_ident: "VALUE"})
        .reset_index(drop=True)
    )

    result["AGI_LOWER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][0]
    )
    result["AGI_UPPER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][1]
    )
    result["GEO_NAME"] = result["GEO_ID"].map(ID_TO_NAME)

    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "AGI_LOWER_BOUND", "AGI_UPPER_BOUND", "VALUE"]
    ]
    result["IS_COUNT"] = int(is_count)
    result["VARIABLE"] = variable_name

    result["VALUE"] = np.where(
        result["IS_COUNT"] == 0, result["VALUE"] * 1_000, result["VALUE"]
    )

    if state_df is not None:
        # If a DataFrame is passed, we append the new data to it.
        df = pd.concat([state_df, result], ignore_index=True)
        return df

    return result


def pull_district_soi_variable(
    soi_variable_ident: str,  # the district SOI csv file has a column for each target variable
    variable_name: Union[str, None],
    is_count: bool,
    district_df: Optional[pd.DataFrame] = None,
    redistrict: bool = True,
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
    df = df.loc[
        (df["CONG_DISTRICT"] != "00") | (df["STATEFIPS"].isin(at_large_states))
    ].reset_index(drop=True)

    df["agi_bracket"] = df["agi_stub"].map(AGI_STUB_TO_BAND)
    result = df[
        ["GEO_ID", "CONG_DISTRICT", "STATE", "agi_bracket", soi_variable_ident]
    ].rename(columns={soi_variable_ident: "VALUE"})

    result["AGI_LOWER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][0]
    )
    result["AGI_UPPER_BOUND"] = result["agi_bracket"].map(
        lambda b: AGI_BOUNDS[b][1]
    )

    if redistrict:
        result = apply_redistricting(result, variable_name)

    result["GEO_NAME"] = result["GEO_ID"].map(ID_TO_NAME)

    assert df["GEO_ID"].nunique() == 436

    if redistrict:
        geo_id_df = pd.read_csv(
            Path(get_data_directory())
            / "input"
            / "geographies"
            / "geo_id_name.csv"
        )
        valid_district_codes = set(
            geo_id_df[geo_id_df["GEO_ID"].str.startswith("5001800US")][
                "GEO_ID"
            ]
        )

        # Check that all GEO_IDs are valid
        produced_codes = set(result["GEO_ID"])
        invalid_codes = produced_codes - valid_district_codes
        assert (
            not invalid_codes
        ), f"Invalid district codes after redistricting: {invalid_codes}"

        # Check we have exactly 436 districts
        assert (
            len(produced_codes) == 436
        ), f"Expected 436 districts after redistricting, got {len(produced_codes)}"

        # Check that all GEO_IDs successfully mapped to names
        missing_names = result[result["GEO_NAME"].isna()]["GEO_ID"].unique()
        assert (
            len(missing_names) == 0
        ), f"GEO_IDs without names in ID_TO_NAME mapping: {missing_names}"

    # final column order
    result = result[
        ["GEO_ID", "GEO_NAME", "AGI_LOWER_BOUND", "AGI_UPPER_BOUND", "VALUE"]
    ]
    result["IS_COUNT"] = int(is_count)
    result["VARIABLE"] = variable_name

    result["VALUE"] = np.where(
        result["IS_COUNT"] == 0, result["VALUE"] * 1_000, result["VALUE"]
    )

    if district_df is not None:
        # If a DataFrame is passed, we append the new data to it.
        df = pd.concat([district_df, result], ignore_index=True)
        return df

    return result


def apply_redistricting(
    df: pd.DataFrame,
    variable_name: str,
) -> pd.DataFrame:
    """Apply redistricting transformation to congressional district data."""
    mapping_matrix = get_district_mapping_matrix()
    mapping_df = pd.read_csv(
        Path(get_data_directory())
        / "input"
        / "geographies"
        / "district_mapping.csv"
    )

    # Get sorted lists of old and new codes (to match the matrix ordering)
    old_codes = sorted(mapping_df["code_old"].unique())
    new_codes = sorted(mapping_df["code_new"].unique())

    old_to_idx = {code: i for i, code in enumerate(old_codes)}

    assert mapping_matrix.shape == (
        436,
        436,
    ), f"Expected 436x436 matrix, got {mapping_matrix.shape}"
    assert np.allclose(
        mapping_matrix.sum(axis=1), 1.0
    ), "Mapping proportions don't sum to 1"

    # Process each AGI bracket separately
    result_dfs = []

    for bracket in (
        df[["AGI_LOWER_BOUND", "AGI_UPPER_BOUND"]]
        .drop_duplicates()
        .itertuples()
    ):
        bracket_df = df[
            (df["AGI_LOWER_BOUND"] == bracket.AGI_LOWER_BOUND)
            & (df["AGI_UPPER_BOUND"] == bracket.AGI_UPPER_BOUND)
        ].copy()

        # Create value vector for old districts (436 elements)
        old_values = np.zeros(436)
        for _, row in bracket_df.iterrows():
            geo_id = row["GEO_ID"]

            # Handle DC special case: SOI uses 1100, current map uses 1198
            if geo_id == "5001800US1100":
                geo_id = "5001800US1198"

            if geo_id in old_to_idx:
                idx = old_to_idx[geo_id]
                old_values[idx] = row["VALUE"]

        # Apply transformation: new = matrix^T @ old
        new_values = mapping_matrix.T @ old_values

        # Create new dataframe with redistributed values
        new_rows = []
        for i, new_code in enumerate(new_codes):
            state_fips = new_code[-4:-2]
            district = new_code[-2:]

            new_row = {
                "GEO_ID": new_code,
                "CONG_DISTRICT": district,
                "STATE": state_fips,  # This is FIPS code, not abbreviation
                "agi_bracket": bracket_df.iloc[0]["agi_bracket"],
                "AGI_LOWER_BOUND": bracket.AGI_LOWER_BOUND,
                "AGI_UPPER_BOUND": bracket.AGI_UPPER_BOUND,
                "VALUE": new_values[i],
            }
            new_rows.append(new_row)

        if new_rows:
            result_dfs.append(pd.DataFrame(new_rows))

    # Combine all brackets
    if result_dfs:
        result = pd.concat(result_dfs, ignore_index=True)
    else:
        # If no result_dfs, create empty DataFrame with proper structure
        result = pd.DataFrame(
            columns=[
                "GEO_ID",
                "CONG_DISTRICT",
                "STATE",
                "agi_bracket",
                "AGI_LOWER_BOUND",
                "AGI_UPPER_BOUND",
                "VALUE",
            ]
        )

    logger.info(f"Redistricting complete for {variable_name}")
    logger.info(
        f"Old districts: {len(old_codes)}, New districts: {len(new_codes)}"
    )

    # Verify total preservation
    old_total = df["VALUE"].sum()
    new_total = result["VALUE"].sum()
    if not np.isclose(old_total, new_total, rtol=1e-6):
        logger.error(
            f"Total value changed during redistricting: {old_total} -> {new_total}"
        )
        raise ValueError(f"Total value not preserved during redistricting")

    return result


def _get_soi_data(geo_level: str) -> pd.DataFrame:
    """
    geo_level ∈ {'National', 'State', 'District'}
    Returns a DataFrame with all SOI variables for the specified geography level
    """
    if geo_level == "National":
        var_indices = NATIONAL_VARIABLES
        variable_pull = pull_national_soi_variable
    elif geo_level == "State":
        var_indices = GEOGRAPHY_VARIABLES
        variable_pull = pull_state_soi_variable
    elif geo_level == "District":
        var_indices = GEOGRAPHY_VARIABLES
        variable_pull = pull_district_soi_variable
    else:
        raise ValueError("geo_level must be National, State or District")

    df = pd.DataFrame()
    for variable, identifyer in var_indices.items():
        variable_df = variable_pull(
            soi_variable_ident=identifyer,
            variable_name=variable,
            is_count=1 if variable.endswith("count") else 0,
        )
        df = pd.concat([df, variable_df], ignore_index=True)
    return df


def create_targets(
    var_indices: dict[str : Union[int, str]],
    variable_pull: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    """Create a DataFrame with AGI targets."""
    df = pd.DataFrame()
    for variable, identifyer in var_indices.items():
        variable_df = variable_pull(
            soi_variable_ident=identifyer,
            variable_name=variable,
            is_count=1 if variable.endswith("count") else 0,
        )
        df = pd.concat([df, variable_df], ignore_index=True)
    return df


def combine_geography_levels() -> None:
    """Combine SOI data across geography levels with validation and rescaling."""
    national = _get_soi_data("National")
    state = _get_soi_data("State")
    district = _get_soi_data("District")

    # Add state FIPS codes for validation
    state["STATEFIPS"] = state["GEO_ID"].str[-2:]
    district["STATEFIPS"] = district["GEO_ID"].str[-4:-2]

    # Get unique variables and AGI brackets for iteration
    variables = national["VARIABLE"].unique()
    agi_brackets = national[
        ["AGI_LOWER_BOUND", "AGI_UPPER_BOUND"]
    ].drop_duplicates()

    # Validate and rescale state totals against national totals
    for variable in variables:
        for _, bracket in agi_brackets.iterrows():
            lower, upper = (
                bracket["AGI_LOWER_BOUND"],
                bracket["AGI_UPPER_BOUND"],
            )

            # Get national total for this variable/bracket combination
            nat_mask = (
                (national["VARIABLE"] == variable)
                & (national["AGI_LOWER_BOUND"] == lower)
                & (national["AGI_UPPER_BOUND"] == upper)
            )
            us_total = national.loc[nat_mask, "VALUE"].iloc[0]

            # Get state total for this variable/bracket combination
            state_mask = (
                (state["VARIABLE"] == variable)
                & (state["AGI_LOWER_BOUND"] == lower)
                & (state["AGI_UPPER_BOUND"] == upper)
            )
            state_total = state.loc[state_mask, "VALUE"].sum()

            # Rescale states if they don't match national total
            if not np.isclose(state_total, us_total, rtol=1e-3):
                logger.warning(
                    f"States' sum does not match national total for {variable} "
                    f"in bracket [{lower}, {upper}]. Rescaling state targets."
                )
                state.loc[state_mask, "VALUE"] *= us_total / state_total

    # Validate and rescale district totals against state totals
    for variable in variables:
        for _, bracket in agi_brackets.iterrows():
            lower, upper = (
                bracket["AGI_LOWER_BOUND"],
                bracket["AGI_UPPER_BOUND"],
            )

            # Create masks for this variable/bracket combination
            state_mask = (
                (state["VARIABLE"] == variable)
                & (state["AGI_LOWER_BOUND"] == lower)
                & (state["AGI_UPPER_BOUND"] == upper)
            )
            district_mask = (
                (district["VARIABLE"] == variable)
                & (district["AGI_LOWER_BOUND"] == lower)
                & (district["AGI_UPPER_BOUND"] == upper)
            )

            # Get state totals indexed by STATEFIPS
            state_totals = state.loc[state_mask].set_index("STATEFIPS")[
                "VALUE"
            ]

            # Get district totals grouped by STATEFIPS
            district_totals = (
                district.loc[district_mask].groupby("STATEFIPS")["VALUE"].sum()
            )

            # Check and rescale districts for each state
            for fips, d_total in district_totals.items():
                s_total = state_totals.get(fips)

                if s_total is not None and not np.isclose(
                    d_total, s_total, rtol=1e-3
                ):
                    logger.warning(
                        f"Districts' sum does not match {fips} state total for {variable} "
                        f"in bracket [{lower}, {upper}]. Rescaling district targets."
                    )
                    rescale_mask = district_mask & (
                        district["STATEFIPS"] == fips
                    )
                    district.loc[rescale_mask, "VALUE"] *= s_total / d_total

    # Combine all data
    combined = pd.concat(
        [
            national,
            state.drop(columns="STATEFIPS"),
            district.drop(columns="STATEFIPS"),
        ],
        ignore_index=True,
    ).sort_values(["GEO_ID", "VARIABLE", "AGI_LOWER_BOUND"])

    # Save combined data
    out_dir = Path(get_data_directory()) / "input" / "soi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "soi_targets.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"Combined SOI targets saved to {out_path}")


def main() -> None:
    """Main function to generate combined SOI targets."""
    combine_geography_levels()


if __name__ == "__main__":
    main()
