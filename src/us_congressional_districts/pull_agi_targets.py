# TODOS;
#  uv pip install openpyxl
# data/input/geographies/districts.csv

from pathlib import Path

import pandas as pd
import requests
import io

from us_congressional_districts.utils import get_data_directory


district_lookup = pd.read_csv(Path(get_data_directory() / 'input' / 'geographies' / 'districts.csv'))
district_lookup["state_fips"]   = district_lookup["GEO_ID"].str[9:11]
district_lookup["district_code"] = (
    district_lookup["GEO_ID"].str[11:13].astype(int)
)

# May want to add these into a geography utils file later.
STATE_FIPS = {
    #  state :  FIPS
    "al": "01", "ak": "02", "az": "04", "ar": "05", "ca": "06",
    "co": "08", "ct": "09", "de": "10", "dc": "11", "fl": "12",
    "ga": "13", "hi": "15", "id": "16", "il": "17", "in": "18",
    "ia": "19", "ks": "20", "ky": "21", "la": "22", "me": "23",
    "md": "24", "ma": "25", "mi": "26", "mn": "27", "ms": "28",
    "mo": "29", "mt": "30", "ne": "31", "nv": "32", "nh": "33",
    "nj": "34", "nm": "35", "ny": "36", "nc": "37", "nd": "38",
    "oh": "39", "ok": "40", "or": "41", "pa": "42", "ri": "44",
    "sc": "45", "sd": "46", "tn": "47", "tx": "48", "ut": "49",
    "vt": "50", "va": "51", "wa": "53", "wv": "54", "wi": "55",
    "wy": "56",
}

AGI_ORDER = [
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

# optional: a simpler set of column names for the CSV header
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



def load_cd_agi(state_abbr: str) -> pd.DataFrame:
    """
    Fetch IRS SOI AGI-by-district for one state
    and return a tidy frame with columns
       GEO_ID | agi_bracket | number_of_individuals
    (district-0 statewide totals are dropped after an integrity check.)
    """
    state_abbr = state_abbr.lower()
    url = f"https://www.irs.gov/pub/irs-soi/22incd{state_abbr}.xlsx"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # ── read the three columns we care about ─────────────────────────
    cols = ["district", "agi_bracket", "number_of_individuals"]
    df = (
        pd.read_excel(
            io.BytesIO(resp.content),
            sheet_name="Sheet1",
            skiprows=6,
            usecols=[0, 1, 11],
            header=None,
            names=cols,
        )
        .query("district.notna() and agi_bracket.notna()")
        .assign(district=lambda d: d["district"].astype(int))
    )

    # ── integrity check: statewide row must equal Σ(district rows) ──
    totals = (
        df[df["district"] == 0]
        .set_index("agi_bracket")["number_of_individuals"]
    )
    parts  = (
        df[df["district"] != 0]
        .groupby("agi_bracket")["number_of_individuals"]
        .sum()
    )

    # 1) every bracket (except the all-bracket "Total") must match
    totals_brk = totals.drop(labels="Total", errors="ignore").sort_index()
    parts      = parts.sort_index()
    
    pd.testing.assert_series_equal(        # raises AssertionError if any differ
        totals_brk, parts,
        check_names=False,                 # we renamed cols, so ignore the name
    )
    
    # 2) the statewide "Total" must equal the sum of all bracket totals
    assert totals["Total"] == parts.sum(), (
        f"Grand totals disagree for {state_abbr.upper()}: "
        f"statewide row = {totals['Total']:,}, "
        f"sum(districts) = {parts.sum():,}"
    )

    # ── attach GEO_IDs ───────────────────────────────────────────────
    fips = STATE_FIPS[state_abbr]
    mapping = district_lookup.loc[
        district_lookup["state_fips"] == fips,
        ["district_code", "GEO_ID"],
    ]
    df = (
        df.merge(mapping, how="left",
                 left_on="district", right_on="district_code")
          .drop(columns="district_code")
    )

    # ── drop statewide row, reorder, clean columns ───────────────────
    df = (
        df[df["district"] != 0]                       # keep real CDs only
          .drop(columns="district")                  # no longer needed
          [["GEO_ID", "agi_bracket", "number_of_individuals"]]  # order
          .reset_index(drop=True)
    )
    return df


# 3.  ── example usage ────────────────────────────────────────────────
al = load_cd_agi("al")   # Alabama (districts 1–7 plus statewide row 0)
nc = load_cd_agi("nc")   # North Carolina (districts 1–14 plus 0)

def agi_wide(df_tall: pd.DataFrame, rename_cols: bool = True) -> pd.DataFrame:
    """Tall → wide pivot; drops the column-index name to hide ‘agi_bracket’."""
    wide = (
        df_tall.pivot_table(
            index="GEO_ID",
            columns="agi_bracket",
            values="number_of_individuals",
            aggfunc="sum",
        )
        .reindex(columns=AGI_ORDER)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    wide.columns.name = None          # ← remove the annoying header line
    if rename_cols:
        wide = wide.rename(columns=AGI_RENAME)
    return wide

agi_wide(al)

# TODO: some of these are failing their tests - find out why
# ── 2. loop through every workbook ────────────────────────────────────
frames = []
for abbr in STATE_FIPS:          # e.g. 'al', 'ak', …, 'wy', 'dc'
    print(f"Downloading {abbr.upper()} …")
    try:
        frames.append(load_cd_agi(abbr))
    except Exception as exc:
        print(f"  ⚠️  {abbr.upper()} skipped ({exc})")


# ── 3. stack & pivot ──────────────────────────────────────────────────
df_tall  = pd.concat(frames, ignore_index=True)
df_wide  = agi_wide(df_tall)

# TODO: yeah, where are my missing districts? 

# ── 4. save to the requested location ─────────────────────────────────
out_path = Path(get_data_directory()) / "input" / "soi" / "agi_district.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df_wide.to_csv(out_path, index=False)




# Ehh, are we going to do this later? 
from policyengine_us import Microsimulation
from policyengine_us_data.datasets.cps import EnhancedCPS_2024

sim = Microsimulation(dataset=EnhancedCPS_2024)

agi = (
    employment_income_last_year
    + self_employment_income_last_year
    + dividend_income
    + interest_income
    + rental_income
    + capital_gains
    + other_income_components
    - above_the_line_deductions
)

