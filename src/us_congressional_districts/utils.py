import pathlib
import requests

import pandas as pd
import numpy as np


def get_census_docs(year):
    docs_url = (
        f"https://api.census.gov/data/{year}/acs/acs1/subject/variables.json"
    )

    docs_response = requests.get(docs_url)
    docs_response.raise_for_status()

    return docs_response.json()


def pull_subject_table(group: str, geo: str, year: int) -> pd.DataFrame:
    """
    "group": e.g., 'S2201'
    "geo": 'National' | 'State' | 'District'
    """
    base = f"https://api.census.gov/data/{year}/acs/acs1/subject"
    geo_q = {
        "National": "us:*",
        "State": "state:*",
        "District": "congressional+district:*",
    }[geo]

    url = f"{base}?get=group({group})&for={geo_q}"

    data = requests.get(url).json()
    headers, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df


def investigate_census_targets(docs, group, concept, estimate_type = "Total"):
    record_gen = (
        (value["label"], key)
        for key, value in docs["variables"].items()
        if value["group"] == group 
        and value["concept"] == concept
    )
    records = []
    for record in record_gen:
        if estimate_type:
            if record[0].startswith("Estimate!!" + estimate_type):
                records.append(record) 
        else:
            records.append[record]
    return(dict(records)) 


def check_geographic_consistency(adjusted_dfs, variable_of_interest):
    """
    Checks for geographic rollup consistency in the adjusted data.

    This function verifies two things:
    1. The sum of the variable of interest for all districts in a state/DC
       equals the total for that state/DC.
    2. The sum of the variable of interest for all states plus DC equals the
       national total.

    Args:
        adjusted_dfs (dict): A dictionary of pandas DataFrames with keys 'National',
                             'State', 'District', and 'DC'.
        variable_of_interest (str): The name of the column to check.

    Returns:
        bool: True if all totals are consistent, False otherwise.
    """
    print("--- Running Geographic Consistency Check ---")
    is_consistent = True

    # --- Check 1: Rollup from States/DC to National ---
    print("Checking national rollup...")
    state_total = adjusted_dfs['State'][variable_of_interest].sum()
    dc_total = adjusted_dfs['DC'][variable_of_interest].sum()
    calculated_national_total = state_total + dc_total
    reported_national_total = adjusted_dfs['National'][variable_of_interest].iloc[0]

    if not np.isclose(calculated_national_total, reported_national_total):
        print(f"  [FAIL] National total is inconsistent.")
        print(f"    - Sum of States + DC: {calculated_national_total}")
        print(f"    - Reported National Total: {reported_national_total}")
        is_consistent = False
    else:
        print("  [OK] National total is consistent.")

    # --- Check 2: Rollup from Districts to States/DC ---
    print("\nChecking district-to-state rollup...")
    districts_consistent = True # Flag specific to this section
    
    # Create temporary copies to avoid modifying the original dataframes
    state_df = adjusted_dfs['State'].copy()
    district_df = adjusted_dfs['District'].copy()
    dc_df = adjusted_dfs['DC'].copy()

    # Add StateFIPS as a temporary key for grouping
    for df in [state_df, district_df, dc_df]:
        if 'GEO_ID' in df.columns:
            df['StateFIPS'] = df['GEO_ID'].str[9:11]

    # Combine states and DC for easier iteration
    state_level_df = pd.concat([state_df, dc_df], ignore_index=True)

    # Check each state/DC entity
    for _, state_row in state_level_df.iterrows():
        fips = state_row['StateFIPS']
        state_name = state_row.get('State', 'District of Columbia')
        state_total = state_row[variable_of_interest]
        
        # In this data structure, DC acts as its own district and has no sub-geographies
        # in the 'District' dataframe. Therefore, its "district sum" is its own total.
        if fips == '11': # FIPS code for DC
             districts_sum = state_total
        else:
            # For regular states, sum all districts corresponding to the state's FIPS code
            districts_in_state = district_df[district_df['StateFIPS'] == fips]
            districts_sum = districts_in_state[variable_of_interest].sum()

        if not np.isclose(districts_sum, state_total):
            print(f"  [FAIL] Mismatch in {state_name} (FIPS: {fips})")
            print(f"    - Sum of Districts: {districts_sum}")
            print(f"    - Reported State Total: {state_total}")
            is_consistent = False
            districts_consistent = False

    if districts_consistent:
        print("  [OK] All districts correctly roll up to their state/DC total.")

    print("\n--- Check Complete ---")
    if is_consistent:
        print("Result: All geographic levels are consistent.")
    else:
        print("Result: Inconsistencies found.")
        
    return is_consistent


def enforce_geographic_self_consistency(raw_dfs, variable_of_interest):
    """
    Proportionally scales state and district data to match national and state totals.

    This function ensures that the sum of a given variable in the state-level data
    matches the national total (after accounting for D.C.) and that the sum of
    the same variable for congressional districts within each state matches the
    (potentially new) scaled state total.

    Args:
        raw_dfs (dict): A dictionary of pandas DataFrames with keys 'National', 'State', 
                        and 'District'. The DataFrames are expected to have 'GEO_ID' 
                        and 'NAME' columns, in addition to the `variable_of_interest`.
        variable_of_interest (str): The name of the column to be scaled (e.g., 'overall').

    Returns:
        dict: A new dictionary with the scaled DataFrames. If no scaling is needed,
              the dataframes in the returned dictionary will be identical to the originals.
    """
    # Create a deep copy of the dictionary and dataframes to avoid modifying the originals
    scaled_dfs = {key: df.copy() for key, df in raw_dfs.items()}
    for key in scaled_dfs.keys():
        scaled_dfs[key][variable_of_interest] = pd.to_numeric(scaled_dfs[key][variable_of_interest])

    national_df = scaled_dfs['National']
    state_df = scaled_dfs['State']
    district_df = scaled_dfs['District']
    dc_df = scaled_dfs['DC']

    national_total = national_df[variable_of_interest].iloc[0]
    dc_state_total = dc_df[variable_of_interest].iloc[0]

    sum_of_states = state_df[variable_of_interest].sum()
    
    national_total_for_states = national_total - dc_state_total

    # Scale the 50 states if their sum does not match the national target
    if not np.isclose(sum_of_states, national_total_for_states):
        scaling_factor = national_total_for_states / sum_of_states
        state_df[variable_of_interest] *= scaling_factor

    # District-to-State Scaling ------------
    state_df['StateFIPS'] = state_df['GEO_ID'].str[9:11]
    district_df['StateFIPS'] = district_df['GEO_ID'].str[9:11]

    scaled_districts_list = []
    
    for fips in state_df['StateFIPS'].unique():
        state_row = state_df[state_df['StateFIPS'] == fips]
        state_total = state_row[variable_of_interest].iloc[0]

        districts_in_state = district_df[district_df['StateFIPS'] == fips].copy()
        sum_of_districts = districts_in_state[variable_of_interest].sum()
        
        # Scale districts if their sum does not match the state total
        if not np.isclose(sum_of_districts, state_total):
            scaling_factor = state_total / sum_of_districts
            districts_in_state[variable_of_interest] *= scaling_factor
        
        scaled_districts_list.append(districts_in_state)

    scaled_dfs['District'] = pd.concat(scaled_districts_list).drop(columns=['StateFIPS'])

    scaled_dfs['State'] = scaled_dfs['State'].drop(columns=['StateFIPS'])
    
    return scaled_dfs


def adjust_to_administrative_data(raw_dfs, variable_of_interest, admin_data):
    """
    Adjusts survey data to match administrative totals, ensuring geographic rollup
    consistency, including the District of Columbia.

    Args:
        raw_dfs (dict): A dictionary of pandas DataFrames. Must contain keys:
                        'National', 'State', 'District', and 'DC'.
        variable_of_interest (str): The name of the column containing the data to be adjusted.
        admin_data (int, float, or dict): The administrative data.
            - If int or float: The authoritative national total.
            - If dict: A dictionary with keys 'State' and 'DC', containing DataFrames
              with authoritative totals for states and DC, respectively. These DataFrames
              must have a 'GEO_ID' column and a column with the same name as
              `variable_of_interest`.

    Returns:
        dict: A new dictionary with the adjusted and scaled DataFrames, where
              the national total is the sum of all state-level entities (incl. DC),
              and congressional districts sum to their respective state/DC total.
    """
    # Create a deep copy to avoid modifying original dataframes
    adjusted_dfs = {key: df.copy() for key, df in raw_dfs.items()}
    # Ensure the target variable is a numeric type for calculations
    for key in adjusted_dfs:
        adjusted_dfs[key][variable_of_interest] = pd.to_numeric(adjusted_dfs[key][variable_of_interest])

    # --- Scenario 1: Adjusting to a single National Administrative Total ---
    if isinstance(admin_data, (int, float)):
        national_admin_total = admin_data

        # The survey total is the sum of all state values plus the DC value.
        current_survey_total = (adjusted_dfs['State'][variable_of_interest].sum() +
                                adjusted_dfs['DC'][variable_of_interest].sum())

        # If the survey total doesn't match the admin total, we scale everything.
        if not np.isclose(current_survey_total, national_admin_total):
            scaling_factor = national_admin_total / current_survey_total

            # Apply the same scaling factor to all geographic levels to maintain consistency.
            for key in ['State', 'District', 'DC']:
                adjusted_dfs[key][variable_of_interest] *= scaling_factor

        # Set the national total directly to the administrative total.
        adjusted_dfs['National'][variable_of_interest] = national_admin_total

    # --- Scenario 2: Adjusting to State-Level and DC Administrative Totals ---
    elif isinstance(admin_data, dict):
        # Extract admin data for states and DC from the input dictionary.
        admin_states_df = admin_data['State'].copy()
        admin_dc_df = admin_data['DC'].copy()
        admin_combined_df = pd.concat([admin_states_df, admin_dc_df], ignore_index=True)

        for df in [admin_combined_df, adjusted_dfs['State'], adjusted_dfs['District'], adjusted_dfs['DC']]:
            df['StateFIPS'] = df['GEO_ID'].str[9:11]

        survey_state_level_df = pd.concat([adjusted_dfs['State'], adjusted_dfs['DC']], ignore_index=True)

        merged_df = pd.merge(
            survey_state_level_df,
            admin_combined_df[['StateFIPS', variable_of_interest]],
            on='StateFIPS',
            how='left',
            suffixes=('', '_admin')
        )
        # Replace survey values with admin values where available.
        merged_df[variable_of_interest] = merged_df[variable_of_interest + '_admin'].fillna(merged_df[variable_of_interest])
        merged_df.drop(columns=[variable_of_interest + '_admin'], inplace=True)
        
        # This dataframe now holds the adjusted totals for all states and DC.
        adjusted_state_level_df = merged_df

        # --- Scale Congressional Districts to Match New State/DC Totals ---
        scaled_districts_list = []
        district_df = adjusted_dfs['District']

        # Iterate through each state/DC FIPS to scale its districts.
        for fips in adjusted_state_level_df['StateFIPS'].unique():
            # Get the new authoritative total for this state/DC.
            state_admin_total = adjusted_state_level_df.loc[
                adjusted_state_level_df['StateFIPS'] == fips, variable_of_interest
            ].iloc[0]

            # Get all congressional districts belonging to this state/DC.
            districts_in_state = district_df[district_df['StateFIPS'] == fips].copy()
            
            sum_of_districts_survey = districts_in_state[variable_of_interest].sum()

            if sum_of_districts_survey > 0 and not np.isclose(sum_of_districts_survey, state_admin_total):
                scaling_factor = state_admin_total / sum_of_districts_survey
                districts_in_state[variable_of_interest] *= scaling_factor
           
            scaled_districts_list.append(districts_in_state)

        # --- Reassemble the Final Adjusted DataFrames ---
        # DC FIPS code is '11'.
        adjusted_dfs['State'] = adjusted_state_level_df[adjusted_state_level_df['StateFIPS'] != '11']
        adjusted_dfs['DC'] = adjusted_state_level_df[adjusted_state_level_df['StateFIPS'] == '11']

        # Reassemble the scaled district data.
        if scaled_districts_list:
            adjusted_dfs['District'] = pd.concat(scaled_districts_list)

        # Drop the temporary 'StateFIPS' column from all dataframes.
        for key in adjusted_dfs:
            if 'StateFIPS' in adjusted_dfs[key].columns:
                adjusted_dfs[key].drop(columns=['StateFIPS'], inplace=True)

        # Finally, update the national total. It is now the sum of all adjusted state
        # and DC totals, ensuring the rollup is consistent.
        new_national_total = (adjusted_dfs['State'][variable_of_interest].sum() +
                              adjusted_dfs['DC'][variable_of_interest].sum())
        adjusted_dfs['National'][variable_of_interest] = new_national_total

    else:
        raise TypeError("`admin_data` must be a number (for national total) or a dictionary (for state/DC totals).")

    return adjusted_dfs


# TODO use the us library to avoid having this here
# You know what might be good? Go ahead and make the Geography table with all the parent and child relationships and
# also have this information in there.
def get_state_fips_codes():
    return {
        "al": "01",
        "ak": "02",
        "az": "04",
        "ar": "05",
        "ca": "06",
        "co": "08",
        "ct": "09",
        "de": "10",
        "fl": "12",
        "ga": "13",
        "hi": "15",
        "id": "16",
        "il": "17",
        "in": "18",
        "ia": "19",
        "ks": "20",
        "ky": "21",
        "la": "22",
        "me": "23",
        "md": "24",
        "ma": "25",
        "mi": "26",
        "mn": "27",
        "ms": "28",
        "mo": "29",
        "mt": "30",
        "ne": "31",
        "nv": "32",
        "nh": "33",
        "nj": "34",
        "nm": "35",
        "ny": "36",
        "nc": "37",
        "nd": "38",
        "oh": "39",
        "ok": "40",
        "or": "41",
        "pa": "42",
        "ri": "44",
        "sc": "45",
        "sd": "46",
        "tn": "47",
        "tx": "48",
        "ut": "49",
        "vt": "50",
        "va": "51",
        "wa": "53",
        "wv": "54",
        "wi": "55",
        "wy": "56",
    }


def get_data_directory() -> pathlib.Path:
    """
    Determines the absolute path to the 'us-congressional-districts/inputs' directory,
    assuming the script is located within 'us-congressional-districts/src/us_congressional_districts'.
    """
    script_path = pathlib.Path(__file__).resolve()
    repo_root = script_path.parent.parent.parent
    inputs_dir = repo_root / "data"

    return inputs_dir
