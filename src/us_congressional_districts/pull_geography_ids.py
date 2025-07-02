from pathlib import Path
import pandas as pd
import requests

from us_congressional_districts.utils import get_data_directory

"""Utilities to map every geography ID to its name, for nationa, state, and district levels."""

YEAR = 2023  # change when a newer ACS 1-year release is available

STATE_NAME_TO_ABBREV = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

NON_VOTING_GEO_IDS = {
    "0400000US72",  # Puerto Rico (state level)
    "5001800US7298",  # Puerto Rico
    "5001800US6098",  # American Samoa
    "5001800US6698",  # Guam
    "5001800US6998",  # Northern Mariana Islands
    "5001800US7898",  # U.S. Virgin Islands
}

BASE_URL = f"https://api.census.gov/data/{YEAR}/acs/acs1/subject?get=group(S0101)&for="


def fetch_geo_df(kind: str) -> pd.DataFrame:
    if kind == "National":
        url = f"{BASE_URL}us:*"
    elif kind == "State":
        url = f"{BASE_URL}state:*"
    elif kind == "District":
        url = f"{BASE_URL}congressional+district:*"
    else:
        raise ValueError("kind must be National, State, or District")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data[1:], columns=data[0])[["GEO_ID", "NAME"]]

    if kind == "National":
        df["GEO_NAME"] = "US"

    elif kind == "State":
        df = df[~df["GEO_ID"].isin(NON_VOTING_GEO_IDS)]
        df["GEO_NAME"] = df["NAME"].map(STATE_NAME_TO_ABBREV)

    elif kind == "District":
        # keep the 435 voting districts + DC:
        df = df[~df["GEO_ID"].isin(NON_VOTING_GEO_IDS)].copy()
        df["GEO_NAME"] = df["NAME"].apply(_district_to_abbrev)

        # sanity-check: should have exactly 436 districts
        assert df.shape[0] == 436, "District count mismatch"

    return df[["GEO_ID", "GEO_NAME"]]


def _district_to_abbrev(name: str) -> str:
    """
    'Congressional District 1 (118th Congress), Alabama' -> 'AL-01'
    'Congressional District (at Large) (118th Congress), Alaska' -> 'AK-01'
    """
    # Alaska, Wyoming, etc. have 'at Large' instead of a number
    if "at Large" in name:
        number = "01"
        state_full = name.split(",")[-1].strip()
    else:
        number = name.split("District ")[1].split(" ")[0].zfill(2)
        state_full = name.split(",")[-1].strip()

    return f"{STATE_NAME_TO_ABBREV[state_full]}-{number}"


def get_geography_ids() -> pd.DataFrame:
    """
    Fetches and returns a DataFrame with GEO_ID and GEO_NAME for all geographies.
    The DataFrame contains the following columns:
    - GEO_ID: Unique identifier for the geography
    - GEO_NAME: Name of the geography (e.g., state / district abbreviation or 'US' for national)
    """
    return pd.concat(
        [
            fetch_geo_df("National"),
            fetch_geo_df("State"),
            fetch_geo_df("District"),
        ],
        ignore_index=True,
    )


def main() -> None:
    out_dir = Path(get_data_directory()) / "input" / "geographies"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [
            fetch_geo_df("National"),
            fetch_geo_df("State"),
            fetch_geo_df("District"),
        ],
        ignore_index=True,
    )

    combined.to_csv(out_dir / "geo_id_name.csv", index=False)


if __name__ == "__main__":
    main()
