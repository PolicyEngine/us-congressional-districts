import requests
import zipfile
import io
import pathlib

import pandas as pd
import numpy as np


def read_bef(congress: int) -> pd.DataFrame:
    """Return a DataFrame with columns ['GEOID', f'CD{congress}'] (strings)."""
    base = ("https://www2.census.gov/programs-surveys/decennial/rdo/"
            "mapping-files/{cycle}/{cong}-congressional-district-bef/")
    cycle = "2023" if congress == 118 else "2025"   # 119 files live in 2025 dir
    url   = f"{base.format(cycle=cycle, cong=congress)}NationalCD{congress}.zip"

    zbytes = requests.get(url).content
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        fname = next(n for n in z.namelist() if n.endswith(".txt"))
        bef = pd.read_csv(z.open(fname), dtype=str)
    # column is CD118 or CDFP depending on file
    cd_col = [c for c in bef.columns if c.startswith("CD") or c == "CDFP"][0]
    return bef[["GEOID", cd_col]].rename(columns={cd_col: f"CD{congress}"})

def read_pl_pop() -> pd.DataFrame:
    """Return ['GEOID', 'POP20'] from the national P.L. 94-171 file."""
    url = ("https://www2.census.gov/programs-surveys/decennial/2020/pl/"
           "2020_PLSF_P1.zip")
    zbytes = requests.get(url).content
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        fname = next(n for n in z.namelist() if n.endswith(".csv"))
        pop = pd.read_csv(z.open(fname),
                          usecols=["GEOID", "P1_001N"],
                          dtype={"GEOID": str, "P1_001N": int})
    return pop.rename(columns={"P1_001N": "POP20"})

# DOESN'T WORK---------------------------------------------------------------------------
# # 1. Load the three building blocks -----------------------------------------
# df118 = read_bef(118)          # GEOID + CD118  (435-seat plan)
# df119 = read_bef(119)          # GEOID + CD119  (435-seat plan, 5 states changed)
# pop   = read_pl_pop()          # GEOID + POP20  (2020 resident population)
# 
# # ---------------------------------------------------------------------------
# # 2. Merge them onto the same row -------------------------------------------
# blocks = (
#     pop
#       .merge(df118, on="GEOID")
#       .merge(df119, on="GEOID")
# )
# 
# # Create 4-digit district GEOIDs: state-FIPS (first 2 chars) + 2-digit CD code
# blocks["CD118GEOID"] = blocks["GEOID"].str[:2] + blocks["CD118"]
# blocks["CD119GEOID"] = blocks["GEOID"].str[:2] + blocks["CD119"]
# 
# # ---------------------------------------------------------------------------
# # 3. Aggregate population to (old, new) pairs -------------------------------
# pair_pop = (
#     blocks.groupby(["CD118GEOID", "CD119GEOID"], as_index=False)["POP20"].sum()
#           .rename(columns={"POP20": "old_population_present"})
# )
# 
# # ---------------------------------------------------------------------------
# # 4. Convert to proportions (rows = 118th seats) ----------------------------
# tot_old = pair_pop.groupby("CD118GEOID")["old_population_present"].transform("sum")
# pair_pop["proportion"] = pair_pop["old_population_present"] / tot_old
# 
# # ---------------------------------------------------------------------------
# # 5. Build the 435×435 matrix -----------------------------------------------
# old_codes = sorted(pair_pop["CD118GEOID"].unique())
# new_codes = sorted(pair_pop["CD119GEOID"].unique())
# 
# old_idx = {c: i for i, c in enumerate(old_codes)}
# new_idx = {c: i for i, c in enumerate(new_codes)}
# 
# import numpy as np
# M = np.zeros((len(old_codes), len(new_codes)), dtype=float)
# M[
#     pair_pop["CD118GEOID"].map(old_idx).to_numpy(),
#     pair_pop["CD119GEOID"].map(new_idx).to_numpy()
# ] = pair_pop["proportion"].to_numpy()
# 
# assert np.allclose(M.sum(axis=1), 1.0)   # sanity check
# M = M.T                                   # so   new = M @ old

import numpy as np
import pandas as pd
from collections import Counter

# ---------------------------------------------------------------------------
# SETTINGS you may tweak
PREFIX        = "5001800US"           # your district GEO_ID prefix
BETA_PARAMS   = (0.4, 5.0)            # how lopsided the splits are
MAX_SPLITS    = 4                     # maximum # new seats one old seat splits into
SEED          = 42

# 2020-census seat changes:  (+) gained, (-) lost
SEAT_CHANGES = {
    "06": -1,  # CA 53→52
    "08": +1,  # CO 7→8
    "12": +1,  # FL 27→28
    "17": -1,  # IL 18→17
    "26": -1,  # MI 14→13
    "30": +1,  # MT 00→02
    "36": -1,  # NY 27→26
    "37": +1,  # NC 13→14
    "39": -1,  # OH 16→15
    "41": +1,  # OR 5→6
    "42": -1,  # PA 18→17
    "48": +2,  # TX 36→38
    "54": -1,  # WV 3→2
}

# ---------------------------------------------------------------------------
def build_old_code_list(new_codes: list[str]) -> list[str]:
    """
    Derive the 117-Congress code list from the 118 list by
    deleting gained seats and inventing lost-seat codes.
    """
    old_codes = set(new_codes)
    for state, diff in SEAT_CHANGES.items():
        # highest existing district number in the 118th codes for this state
        nums = [int(c[-2:]) for c in new_codes if c[len(PREFIX):len(PREFIX)+2] == state]
        max_new = max(nums) if nums else 0

        if diff > 0:                         # state gained seats → remove them from OLD list
            for k in range(diff):
                old_codes.discard(f"{PREFIX}{state}{max_new - k:02d}")
        else:                                # state lost seats → create phantom old seats
            for k in range(-diff):
                old_codes.add(f"{PREFIX}{state}{max_new + k + 1:02d}")

    return sorted(old_codes)

# ---------------------------------------------------------------------------
def simulate_117_to_118_crosswalk(new_codes: list[str],
                                  beta_params=BETA_PARAMS,
                                  max_splits=MAX_SPLITS,
                                  seed=SEED):
    rng = np.random.default_rng(seed)
    old_codes = build_old_code_list(new_codes)

    rows = []
    unchanged = set(new_codes) & set(old_codes)

    # identity rows for unchanged seats
    for c in unchanged:
        rows.append((c, c, 1.0))

    # states that lost seats → “old” seats with no match in new list
    lost = [c for c in old_codes if c not in new_codes]
    for old in lost:
        ss = old[len(PREFIX):len(PREFIX)+2]
        pool = [n for n in new_codes if n[len(PREFIX):len(PREFIX)+2] == ss]
        k = min(max_splits, len(pool))
        targets = rng.choice(pool, size=k, replace=False)
        weights = rng.beta(*beta_params, size=k)
        weights /= weights.sum()
        rows.extend((old, n, w) for n, w in zip(targets, weights))

    # states that gained seats → “new” seats with no match in old list
    gained = [n for n in new_codes if n not in old_codes]
    for new in gained:
        ss = new[len(PREFIX):len(PREFIX)+2]
        pool = [o for o in old_codes if o[len(PREFIX):len(PREFIX)+2] == ss]
        k = min(max_splits, len(pool))
        sources = rng.choice(pool, size=k, replace=False)
        weights = rng.beta(*beta_params, size=k)
        weights /= weights.sum()
        rows.extend((o, new, w) for o, w in zip(sources, weights))

    # ------------------------------------------------------------------
    # collapse duplicates, then renormalise column-wise
    cross = (
        pd.DataFrame(rows, columns=["code_old", "code_new", "proportion"])
          .groupby(["code_old", "code_new"], as_index=False)["proportion"].sum()
    )
    totals = cross.groupby("code_old")["proportion"].transform("sum")
    cross["proportion"] = cross["proportion"] / totals

    cross = cross.sort_values(["code_old", "proportion"],
                              ascending=[True, False]).reset_index(drop=True)

    # final sanity check
    assert np.allclose(cross.groupby("code_old")["proportion"].sum().max(), 1.0)

    return cross, old_codes

# ---------------------------------------------------------------------------
# EXAMPLE usage: replace with your real 118-Congress GEO_ID list
district_lookup = pd.read_csv("districts.csv")          # 435 rows from earlier step
new_codes = district_lookup["GEO_ID"].tolist()

mapping_matrix, old_code_list = simulate_117_to_118_crosswalk(new_codes)

print(mapping_matrix.head())
print("rows in cross-walk:", len(mapping_df))
print("old seats:", len(old_code_list), "  new seats:", len(new_codes))

from us_congressional_districts.utils import get_data_directory
from pathlib import Path

mapping_matrix.to_csv(
    Path(get_data_directory(), "input", "geographies", "district_mapping.csv"),
    index=False
)
