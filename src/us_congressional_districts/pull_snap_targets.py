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


def reformat_cleaned_data():
    """Temporary conversion function"""
    benefits_dir = Path(get_data_directory() / 'input' / 'benefits')
   
    snap_filepath = Path(
        get_data_directory(),
        "targets",
        "edition=cleaned",
        "base_period=2023",
        "reference_period=2023",
        "variable=snap_households",
        "part-001.csv"
    )
    snap_data = pd.read_csv(snap_filepath)
    geo_hierarchies = pd.read_csv(Path(get_data_directory(), 'meta', 'geo_hierarchies.csv'))
    
    # Use Type II SCD to Filter geo_hierarchies for the year 2023
    geo_hierarchies['start_date'] = pd.to_datetime(geo_hierarchies['start_date'])
    geo_hierarchies['end_date'] = pd.to_datetime(geo_hierarchies['end_date'])
    geo_hierarchies_2023 = geo_hierarchies[
        (geo_hierarchies['start_date'] <= '2023-01-01') &
        (geo_hierarchies['end_date'] >= '2023-01-01')
    ]
    
    merged_data = pd.merge(snap_data, geo_hierarchies_2023, left_on='geography_id', right_on='geography_id')
    
    def create_cleaned_df(data, geo_name_map=None, geo_name_prefix=''):
        df = pd.DataFrame()
        df['GEO_ID'] = data['geography_id']
        if geo_name_map:
            df['GEO_NAME'] = data['geography_id'].map(geo_name_map)
        elif 'geography_name' in data.columns:
            df['GEO_NAME'] = data['geography_name']
        else:
            df['GEO_NAME'] = ''
    
        df['AGI_LOWER_BOUND'] = ''
        df['AGI_UPPER_BOUND'] = ''
        df['VALUE'] = data['value']
        df['IS_COUNT'] = 1
        df['VARIABLE'] = 'snap_households'
        return df
    
    # National data
    national_data = merged_data[merged_data['geography_type'] == 'nation'].copy()
    national_data['geography_name'] = 'US'
    cleaned_national = create_cleaned_df(national_data)
    cleaned_national.to_csv(Path(get_data_directory(), 'input', 'benefits', 'cleaned_snap_national.csv'), index=False)
    
    # State data
    state_data = merged_data[merged_data['geography_type'] == 'state-equivalent'].copy()
    # TODO: fix this redundancy if this becomes permanenent
    state_fips_map = {
        '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC',
        '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY',
        '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO', '30': 'MT',
        '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
        '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
        '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY'
    }
    state_data['state_fips'] = state_data['geography_id'].str[-2:]
    state_data['geography_name'] = state_data['state_fips'].map(state_fips_map)
    cleaned_state = create_cleaned_df(state_data)
    cleaned_state.to_csv(Path('us-congressional-districts/data/input/benefits/cleaned_snap_state.csv', index=False)
    cleaned_state.to_csv(Path(get_data_directory(), 'input', 'benefits', 'cleaned_snap_state.csv'), index=False)
    
    # District data
    district_data = merged_data[merged_data['geography_type'] == 'district'].copy()
    district_data['state_fips'] = district_data['geography_id'].str[9:11]
    district_data['district_num'] = district_data['geography_id'].str[11:]
    district_data['geography_name'] = district_data['state_fips'].map(state_fips_map) + ' - District ' + district_data['district_num']
    cleaned_district = create_cleaned_df(district_data)
    cleaned_district["VALUE"] = cleaned_district["VALUE"].round().astype(int)
    cleaned_district.to_csv(Path(get_data_directory(), 'input', 'benefits', 'cleaned_snap_district.csv'), index=False)



if __name__ == "__main__":
    process_snap_data(2023)
