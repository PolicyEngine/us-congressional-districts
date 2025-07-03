import logging
import numpy as np
import requests
import pandas as pd
from pathlib import Path

from us_congressional_districts.utils import get_data_directory
from us_congressional_districts.pull_geography_ids import get_geography_ids

YEAR = 2023

logger = logging.getLogger(__name__)

GEO_ID_NAMES = get_geography_ids()
ID_TO_NAME = GEO_ID_NAMES.set_index("GEO_ID")["GEO_NAME"].to_dict()

LABEL_TO_SHORT = {
    "Estimate!!Total!!Total population!!AGE!!Under 5 years": "0-4",
    "Estimate!!Total!!Total population!!AGE!!5 to 9 years": "5-9",
    "Estimate!!Total!!Total population!!AGE!!10 to 14 years": "10-14",
    "Estimate!!Total!!Total population!!AGE!!15 to 19 years": "15-19",
    "Estimate!!Total!!Total population!!AGE!!20 to 24 years": "20-24",
    "Estimate!!Total!!Total population!!AGE!!25 to 29 years": "25-29",
    "Estimate!!Total!!Total population!!AGE!!30 to 34 years": "30-34",
    "Estimate!!Total!!Total population!!AGE!!35 to 39 years": "35-39",
    "Estimate!!Total!!Total population!!AGE!!40 to 44 years": "40-44",
    "Estimate!!Total!!Total population!!AGE!!45 to 49 years": "45-49",
    "Estimate!!Total!!Total population!!AGE!!50 to 54 years": "50-54",
    "Estimate!!Total!!Total population!!AGE!!55 to 59 years": "55-59",
    "Estimate!!Total!!Total population!!AGE!!60 to 64 years": "60-64",
    "Estimate!!Total!!Total population!!AGE!!65 to 69 years": "65-69",
    "Estimate!!Total!!Total population!!AGE!!70 to 74 years": "70-74",
    "Estimate!!Total!!Total population!!AGE!!75 to 79 years": "75-79",
    "Estimate!!Total!!Total population!!AGE!!80 to 84 years": "80-84",
    "Estimate!!Total!!Total population!!AGE!!85 years and over": "85+",
}
AGE_COLS = list(LABEL_TO_SHORT.values())

BASE_URL = (
    f"https://api.census.gov/data/{YEAR}/acs/acs1/subject?get=group(S0101)"
)
DOCS_URL = (
    f"https://api.census.gov/data/{YEAR}/acs/acs1/subject/variables.json"
)

SAVE_DIR = Path(get_data_directory()) / "input" / "demographics"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _get_age_data(geo_level: str) -> pd.DataFrame:
    """
    geo_level ∈ {'National', 'State', 'District'}
    Returns a DataFrame with GEO_ID plus 18 age-band columns
    """
    if geo_level == "National":
        url = f"{BASE_URL}&for=us:*"
    elif geo_level == "State":
        url = f"{BASE_URL}&for=state:*"
    elif geo_level == "District":
        url = f"{BASE_URL}&for=congressional+district:*"
    else:
        raise ValueError("geo_level must be National, State or District")

    # Data API calls
    data = requests.get(url, timeout=60).json()
    docs = requests.get(DOCS_URL, timeout=60).json()

    df = pd.DataFrame(data[1:], columns=data[0])

    label_to_var = {
        v["label"]: k
        for k, v in docs["variables"].items()
        if v["group"] == "S0101"
        and v["concept"] == "Age and Sex"
        and v["label"] in LABEL_TO_SHORT
    }
    rename_map = {label_to_var[l]: LABEL_TO_SHORT[l] for l in LABEL_TO_SHORT}
    df = df.rename(columns=rename_map)
    df[AGE_COLS] = df[AGE_COLS].astype(int)

    # Keep rows present in the geo_id_names csv
    df = df[df["GEO_ID"].isin(ID_TO_NAME)].copy()
    df["GEO_NAME"] = df["GEO_ID"].map(ID_TO_NAME)

    return df[["GEO_ID", "GEO_NAME"] + AGE_COLS]


def combine_geography_levels() -> None:
    national = _get_age_data("National")
    state = _get_age_data("State")
    district = _get_age_data("District")

    state["STATEFIPS"] = state["GEO_ID"].str[-2:]
    district["STATEFIPS"] = district["GEO_ID"].str[-4:-2]

    for col in AGE_COLS:
        us_total = national[col].iloc[0]  # scalar
        state_total = state[col].sum()
        if not np.isclose(state_total, us_total):
            logger.warning(
                f"States' sum population does not match national total for age band: {col}. Reescaling state targets."
            )
            state[col] *= us_total / state_total

    for col in AGE_COLS:
        state_totals = state.set_index("STATEFIPS")[col]
        district_totals = district.groupby("STATEFIPS")[col].sum()

        for fips, d_total in district_totals.items():
            s_total = state_totals.get(fips)

            if not np.isclose(d_total, s_total):
                logger.warning(
                    f"Districts' sum population does not match {fips} state total for age band: {col}. Reescaling district targets."
                )
                mask = district["STATEFIPS"] == fips
                district.loc[mask, col] *= s_total / d_total

    combined = pd.concat(
        [
            national,
            state.drop(columns="STATEFIPS"),
            district.drop(columns="STATEFIPS"),
        ],
        ignore_index=True,
    ).sort_values("GEO_ID")

    # Ensure all age columns are numeric before saving
    for col in AGE_COLS:
        combined[col] = combined[col].round().astype(int)

    out_path = SAVE_DIR / "age.csv"
    combined.to_csv(out_path, index=False)


def main() -> None:
    """Main function to generate combined age targets."""
    combine_geography_levels()


if __name__ == "__main__":
    main()
