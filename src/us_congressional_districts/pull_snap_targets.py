import os
import requests
import zipfile
import io
import re
from pathlib import Path

import pandas as pd
import numpy as np
import us

from us_congressional_districts.utils import (
    get_data_directory,
    get_census_docs,
    investigate_census_targets,
    pull_subject_table,
    check_geographic_consistency,
    enforce_geographic_self_consistency,
    adjust_to_administrative_data
)


def extract_usda_snap_data(year=2023):
    """
    Downloads and extracts annual state-level SNAP data from the USDA FNS zip file.
    """
    url = "https://www.fns.usda.gov/sites/default/files/resource-files/snap-zip-fy69tocurrent-6.zip"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return None

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    filename = f'FY{str(year)[-2:]}.xlsx'
    with zip_file.open(filename) as f:
        xls = pd.ExcelFile(f)
        tab_results = [] 
        for sheet_name in ['NERO', 'MARO', 'SERO', 'MWRO', 'SWRO', 'MPRO', 'WRO']: 
            df_raw = pd.read_excel(
                xls,
                sheet_name=sheet_name,
                header=None,
                dtype={0: str}
            )

            state_row_mask = (
                df_raw[0].notna()
                & df_raw[1].isna()
                & ~df_raw[0].str.contains('Total', na=False)
                & ~df_raw[0].str.contains('Footnote', na=False)
            )

            df_raw['State'] = df_raw.loc[state_row_mask, 0]
            df_raw['State'] = df_raw['State'].ffill()
            total_rows = df_raw[df_raw[0].eq('Total')].copy()
            total_rows = total_rows.rename(
                columns={
                    1: 'Households',
                    2: 'Persons',
                    3: 'Cost',
                    4: 'Cost Per Household',
                    5: 'Cost Per Person'
                }
            )

            state_totals = (
                total_rows[['State', 'Households', 'Persons',
                            'Cost', 'Cost Per Household', 'Cost Per Person']]
            )

            tab_results.append(state_totals)

    results_df = pd.concat(tab_results)
 
    # Administrative totals for the 50 states
    fifty_states = us.states.STATES
    fips_lookup = {state.name: state.fips.zfill(2) for state in fifty_states}

    df_states = (
       results_df 
          .loc[results_df['State'].isin(fips_lookup.keys())]
          .copy()
    )
    df_states['STATE_FIPS'] = df_states['State'].map(fips_lookup)
    df_states = df_states.loc[~df_states['STATE_FIPS'].isna()].sort_values('STATE_FIPS').reset_index(drop=True)
    df_states['GEO_ID'] = '0400000US' + df_states['STATE_FIPS']

    # Administrative totals for the District of Columbia
    df_dc = results_df.loc[results_df['State'] == us.states.DC.name].copy()
    
    if not df_dc.empty:
        dc_fips = us.states.DC.fips.zfill(2) # Should be '11'
        df_dc['STATE_FIPS'] = dc_fips
        df_dc['GEO_ID'] = '0400000US' + dc_fips
    
    ordered_cols = ['GEO_ID', 'State', 'STATE_FIPS',
                    'Households', 'Persons', 'Cost',
                    'Cost Per Household', 'Cost Per Person']

    final_df_states = df_states[ordered_cols]
    final_df_dc = df_dc[ordered_cols] if not df_dc.empty else pd.DataFrame(columns=ordered_cols)

    for df in [final_df_states, final_df_dc]:
        df["overall"] = pd.to_numeric(df["Households"], errors="coerce").round().astype("Int64")

    return {
        "State": final_df_states,
        "DC": final_df_dc
    }
    

def process_snap_data(year):

    group = "S2201"
    concept = "Food Stamps/Supplemental Nutrition Assistance Program (SNAP)"
    estimate_type = "Households receiving food stamps/SNAP"
    label_to_short_name_mapping = {
        "Estimate!!Households receiving food stamps/SNAP!!Households": "overall",

        "Estimate!!Households receiving food stamps/SNAP!!Households!!No children under 18 years": "no_under18",
        "Estimate!!Households receiving food stamps/SNAP!!Households!!With children under 18 years": "with_under18",

        "Estimate!!Households receiving food stamps/SNAP!!Households!!No people in the household 60 years and over": "no_under60",
        "Estimate!!Households receiving food stamps/SNAP!!Households!!With one or more people in the household 60 years and over": "with_under60"
    }
    docs = get_census_docs(year) 
    label_to_variable_mapping = investigate_census_targets(docs, group, concept, estimate_type)

    rename_mapping = dict(
        [
            (label_to_variable_mapping[v], label_to_short_name_mapping[v])
            for v in label_to_short_name_mapping.keys()
        ]
    )

    raw_dfs = {}
    for geo in ["District", "State", "National"]:
        df = pull_subject_table(group, geo, year)
        df_data = df.rename(columns=rename_mapping)[
            ["GEO_ID", "NAME"] + list(label_to_short_name_mapping.values())
        ]
        if geo == "State":
            raw_dfs["DC"] = df_data[df_data["GEO_ID"].isin(["0400000US11"])]

        # Filter out non-voting geographies, e.g., DC and Puerto Rico
        df_geos = df_data[
            ~df_data["GEO_ID"].isin(
                [
                    "5001800US7298", # Puerto Rico Congressional District 
                    "5001800US1198", # Washington, D.C. Congressional District
                    "0400000US72",    # Puerto Rico (state-level FIPS code 72)
                    "0400000US11",    # Washington, D.C. (state-level FIPS code 11)
                ]
            )
        ].copy()
        raw_dfs[geo] = df_geos
        SAVE_DIR = Path(get_data_directory() / "input" / "demographics")
        df_geos.to_csv(SAVE_DIR / f"raw_snap_{geo}.csv", index=False)

    folder_path = (
        f"{get_data_directory()}/targets/edition=raw/"
        f"base_period={year}/reference_period={year}/"
        f"variable=snap_households/"
    )
    raw_out = pd.concat([
        raw_dfs['National'][['GEO_ID', 'overall']],
        raw_dfs['State'][['GEO_ID', 'overall']],
        raw_dfs['DC'][['GEO_ID', 'overall']],
        raw_dfs['District'][['GEO_ID', 'overall']]
    ]).rename({"GEO_ID": "geography_id", "overall": "value"}, axis=1)
        
    raw_out.to_csv(os.path.join(folder_path, "part-001.csv"), index=False)     

    additive_dfs = enforce_geographic_self_consistency(raw_dfs, 'overall')    
    usda_snap_df = extract_usda_snap_data()
    adjusted_dfs = adjust_to_administrative_data(additive_dfs, 'overall', usda_snap_df)
    assert check_geographic_consistency(adjusted_dfs, 'overall') 

    folder_path = (
        f"{get_data_directory()}/targets/edition=cleaned/"
        f"base_period={year}/reference_period={year}/"
        f"variable=snap_households/"
    )
 
    clean_out = pd.concat([
        adjusted_dfs['National'][['GEO_ID', 'overall']],
        adjusted_dfs['State'][['GEO_ID', 'overall']],
        adjusted_dfs['DC'][['GEO_ID', 'overall']],
        adjusted_dfs['District'][['GEO_ID', 'overall']]
    ]).rename({"GEO_ID": "geography_id", "overall": "value"}, axis=1)
 
    clean_out.to_csv(os.path.join(folder_path, "part-001.csv"), index=False)     


if __name__ == "__main__":
    process_snap_data(2023)
