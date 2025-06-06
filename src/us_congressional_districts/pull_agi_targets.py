import requests
import io
from pathlib import Path

import pandas as pd
import numpy as np

from us_congressional_districts.utils import get_data_directory, get_state_fips_codes


"""
https://www.irs.gov/pub/irs-soi/22incd.csv

Can't just sum up the district totals to get state totals
https://www.irs.gov/pub/irs-soi/congressional2022.zip

For state and National totals
https://www.irs.gov/statistics/soi-tax-stats-historic-table-2
"""


#AGI_ORDER = [
#    "Under $1",
#    "$1 under $10,000",
#    "$10,000 under $25,000",
#    "$25,000 under $50,000",
#    "$50,000 under $75,000",
#    "$75,000 under $100,000",
#    "$100,000 under $200,000",
#    "$200,000 under $500,000",
#    "$500,000 or more",
#]

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


def agi_wide(df_tall: pd.DataFrame, rename_cols: bool = True) -> pd.DataFrame:
    """Tall → wide pivot; drops the column-index name to hide ‘agi_bracket’."""
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


# National ----
national_soi_df = pd.read_excel(
    "https://www.irs.gov/pub/irs-soi/22in54us.xlsx", skiprows=7
)

# total-row sanity check
assert (
    np.abs(
        national_soi_df.iloc[0, 1] - national_soi_df.iloc[0, 2:12].sum()
    ) < 100
), "Row 0 doesn’t add up — check the file."

# grab the 10 bracket counts
agi_values = national_soi_df.iloc[0, 2:12].astype(int).to_numpy()

# combine the two highest brackets, as district data stops at $500k+
agi_values = np.concatenate([agi_values[:8], [agi_values[8] + agi_values[9]]])

## bracket names you want
#agi_labels = [
#    "Under $1",
#    "$1 under $10,000",
#    "$10,000 under $25,000",
#    "$25,000 under $50,000",
#    "$50,000 under $75,000",
#    "$75,000 under $100,000",
#    "$100,000 under $200,000",
#    "$200,000 under $500,000",
#    "$500,000 or more",  # <-- combined bucket
#]
#
out = (
    pd.DataFrame(
        {
            "GEO_ID": "0100000US",
            "agi_bracket": AGI_RENAME.keys(),
            "number_of_individuals": agi_values,
        }
    )
    [["GEO_ID", "agi_bracket", "number_of_individuals"]]
)
out_wide = agi_wide(out)

out_path = Path(get_data_directory()) / "input" / "soi" / "agi_national.csv"
out_wide.to_csv(out_path, index=False)

# State -------------------------------------
state_soi_df = pd.read_csv(
    "https://www.irs.gov/pub/irs-soi/22in55cmcsv.csv",
    thousands=","
)

np.sum(state_soi_df.loc[state_soi_df.STATE == "AL"].N1) / 2  # 2149560
df = state_soi_df.copy()
merged = (
    df[df["AGI_STUB"].isin([9, 10])]
    .groupby("STATE", as_index=False)
    .agg({"N1": "sum"})
    .assign(AGI_STUB=9)
)

# Remove old 9+10 and add merged
df = df[~df["AGI_STUB"].isin([9, 10])]
df = pd.concat([df, merged], ignore_index=True)

# Drop totals
df = df[df["AGI_STUB"] != 0]

df["agi_bracket"] = df["AGI_STUB"].map(AGI_STUB_MAP)
state_fips = get_state_fips_codes()
df["GEO_ID"] = "0400000US" + df["STATE"].str.lower().map(state_fips)
out_df = df[["STATE", "GEO_ID", "agi_bracket", "N1"]].rename(columns={"N1": "number_of_individuals"})

NON_VOTING_STATES = {"US", "AS", "GU", "MP", "PR", "VI", "DC", "OA"}

out_df = (
    out_df.loc[~out_df["STATE"].isin(NON_VOTING_STATES)]
          .reset_index(drop=True)
)[["GEO_ID", "agi_bracket", "number_of_individuals"]]

out_wide = agi_wide(out_df)

out_path = Path(get_data_directory()) / "input" / "soi" / "agi_state.csv"
out_wide.to_csv(out_path, index=False)


# Districts -----------------------------------

district_soi_df = pd.read_csv("https://www.irs.gov/pub/irs-soi/22incd.csv")

df = district_soi_df.copy()
# Step 1: Drop state-level aggregates
df = df[df["agi_stub"] != 0]

# Step 2: Construct GEO_ID (format: 5001800USSSDD)
df["STATEFIPS"] = df["STATEFIPS"].astype(int).astype(str).str.zfill(2)
df["CONG_DISTRICT"] = df["CONG_DISTRICT"].astype(int).astype(str).str.zfill(2)
df["GEO_ID"] = "5001800US" + df["STATEFIPS"] + df["CONG_DISTRICT"]

df["agi_bracket"] = df["agi_stub"].map(AGI_STUB_MAP)

df = df[["GEO_ID", "CONG_DISTRICT", "STATE", "agi_bracket", "N1"]]
df = df.rename(columns={"N1": "number_of_individuals"})

df["state_fips"] = df["GEO_ID"].str[-4:-2]
df["district"]   = df["GEO_ID"].str[-2:]

# 1. find states that really are at-large (just one district in the file)
at_large_states = (
    df.groupby("state_fips")["district"]
      .nunique()
      .pipe(lambda s: s[s == 1].index)
)

# 2.  build the clean list
clean_df = (
    df.loc[
        (df["district"] != "00")
        | (df["state_fips"].isin(at_large_states))
    ]
    .query("state_fips != '11'")  # DC
    .reset_index(drop=True)
)

# sanity-check
assert clean_df["GEO_ID"].nunique() == 435

out_df = clean_df[["GEO_ID", "agi_bracket", "number_of_individuals"]]

out_wide = agi_wide(out_df)
out_path = Path(get_data_directory()) / "input" / "soi" / "agi_district.csv"
out_wide.to_csv(out_path, index=False)
