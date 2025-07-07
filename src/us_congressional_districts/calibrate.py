import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import h5py
from huggingface_hub import hf_hub_download
from typing import Optional

from policyengine_core.data import Dataset
from policyengine_us import Microsimulation
from policyengine_us.system import system
from us_congressional_districts.utils import (
    get_data_directory,
    state_abbr_from_fips,
)
import pandas as pd
import numpy as np
import torch
from microcalibrate import Calibration
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_dataset(dataset: str = "cps_2023", time_period=2023) -> pd.DataFrame:
    """
    Get the dataset from the huggingface hub.
    """
    dataset_path = hf_hub_download(
        repo_id="policyengine/policyengine-us-data",
        filename=f"{dataset}.h5",
        local_dir=get_data_directory() / "input" / "cps",
    )

    return Dataset.from_file(dataset_path, time_period=time_period)


def get_agi_band_label(lower: float, upper: float) -> str:
    """Get the label for the AGI band based on lower and upper bounds."""
    if lower <= 0:
        return f"-inf_{int(upper)}"
    elif np.isposinf(upper):
        return f"{int(lower)}_inf"
    else:
        return f"{int(lower)}_{int(upper)}"


def get_state_abbr_from_fips(fips_code: str) -> str:
    """Get the state abbreviation from the FIPS code."""
    state_abbr_dict = state_abbr_from_fips()
    return state_abbr_dict.get(fips_code)


def create_metric_matrix(
    sim: Microsimulation,
    sim_calculations: dict,
    ages: pd.DataFrame,
    soi_targets: pd.DataFrame,
    households: pd.DataFrame,
):
    """
    Create metric matrix for multi-level calibration (national, state, district).

    Args:
        dataset: Dataset to use for simulation
        ages: DataFrame with age targets for all geographic levels
        soi_targets: DataFrame with SOI targets for all geographic levels
        time_period: Year for calculation

    Returns:
        DataFrame with metrics for each household, including geographic identifiers
    """
    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    # Use passed simulation data
    age = sim.calculate("age").values
    state_code = sim.calculate("state_code").values
    state_fips = sim.calculate("state_fips").values

    matrix = pd.DataFrame()

    for i, age_range in enumerate(age_ranges):
        if age_range != "85+":
            lower_age, upper_age = age_range.split("-")
            in_age_band = (age >= int(lower_age)) & (age <= int(upper_age))
        else:
            in_age_band = age >= 85

        # Map age band to household level
        in_age_band = sim.map_result(
            in_age_band, "person", "household", how="sum"
        )

        # Create age metrics for each geographic level
        unique_geo_ids = ages["GEO_ID"].unique()
        for j, geo_id in enumerate(unique_geo_ids):
            if geo_id.startswith("0100000US"):
                level_prefix = "national"
                geo_mask = np.ones(len(in_age_band), dtype=bool)
            elif geo_id.startswith("0400000US"):
                state_fips_code = geo_id[9:11]
                level_prefix = (
                    f"state_{get_state_abbr_from_fips(state_fips_code)}"
                )
                geo_mask = state_fips == int(state_fips_code)
            elif geo_id.startswith("5001800US"):
                district_code = geo_id[11:13]
                state_fips_code = geo_id[9:11]
                level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
                # Create mask for households in this specific state AND district
                state_mask = state_fips == int(state_fips_code)
                district_int = int(district_code)
                hhs = households.loc[
                    households["district"] == district_int, "household_id"
                ]
                geo_mask = np.zeros(len(in_age_band), dtype=bool)
                # Use household IDs to create boolean mask - find positions where household index matches
                household_mask = np.isin(
                    np.arange(len(state_mask)), hhs.values
                )
                geo_mask = household_mask & state_mask
            else:
                continue

            combined_mask = in_age_band * geo_mask.astype(float)
            matrix[f"age/{level_prefix}/{age_range}"] = combined_mask

    agi_long = (
        soi_targets[
            [
                "GEO_ID",
                "AGI_LOWER_BOUND",
                "AGI_UPPER_BOUND",
                "VARIABLE",
                "IS_COUNT",
            ]
        ]
        .drop_duplicates()
        .sort_values(["IS_COUNT", "VARIABLE", "AGI_LOWER_BOUND"])
    )

    for _, row in agi_long.iterrows():
        lower, upper = row.AGI_LOWER_BOUND, row.AGI_UPPER_BOUND
        band = get_agi_band_label(lower, upper)
        var = row.VARIABLE.replace("/count", "").replace("/amount", "")
        is_count = row.IS_COUNT
        geo_id = row.GEO_ID
        var_values = sim_calculations[var]

        mask = (sim_calculations["adjusted_gross_income"] > lower) & (
            sim_calculations["adjusted_gross_income"] <= upper
        )

        # Determine geographic level and create appropriate mask
        if geo_id.startswith("0100000US"):
            geo_mask = np.ones(len(mask), dtype=bool)
            level_prefix = "national"
        elif geo_id.startswith("0400000US"):
            state_fips_code = geo_id[9:11]
            geo_mask = state_fips == int(state_fips_code)
            level_prefix = f"state_{get_state_abbr_from_fips(state_fips_code)}"
        elif geo_id.startswith("5001800US"):
            district_code = geo_id[11:13]
            state_fips_code = geo_id[9:11]
            level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
            # Create mask for households in this specific state AND district
            state_mask = state_fips == int(state_fips_code)
            district_int = int(district_code)
            hhs = households.loc[
                households["district"] == district_int, "household_id"
            ]
            geo_mask = np.zeros(len(state_fips), dtype=bool)
            # Use household IDs to create boolean mask - find positions where household index matches
            household_mask = np.isin(np.arange(len(state_mask)), hhs.values)
            geo_mask = household_mask & state_mask
        else:
            continue

        # Map geographic mask to tax_unit level
        geo_mask = sim.map_result(
            geo_mask.astype(float), "household", "tax_unit"
        )
        combined_mask = mask & (geo_mask > 0)

        if is_count:
            col = f"soi/{level_prefix}/{var}/count/{band}"
            metric = combined_mask * (var_values > 0).astype(float)
            metric = sim.map_result(metric, "tax_unit", "household")
        else:
            col = f"soi/{level_prefix}/{var}/amount/{band}"
            metric = var_values * combined_mask
            metric = sim.map_result(metric, "tax_unit", "household")

        matrix[col] = metric

    matrix["state_code"] = state_code
    matrix["state_fips"] = state_fips

    return matrix


def create_target_matrix(ages, soi_targets):
    """
    Create an aggregate target matrix for multi-level calibration (national, state, district).

    Args:
        ages: DataFrame containing GEO_ID and GEO_NAME,
          with target variables afterwards for all geographic levels
        soi_targets: DataFrame containing GEO_ID and GEO_NAME with SOI targets for all geographic levels
    """
    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    # Initialize target dictionary
    targets_dict = {}

    # Create age targets for each geographic level
    for idx, row in ages.iterrows():
        geo_id = row["GEO_ID"]

        if geo_id.startswith("0100000US"):
            level_prefix = "national"
        elif geo_id.startswith("0400000US"):
            state_fips_code = geo_id[9:11]
            level_prefix = f"state_{get_state_abbr_from_fips(state_fips_code)}"
        elif geo_id.startswith("5001800US"):
            district_code = geo_id[11:13]
            state_fips_code = geo_id[9:11]
            level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
        else:
            continue

        for age_range in age_ranges:
            col_name = f"age/{level_prefix}/{age_range}"
            targets_dict[col_name] = row[age_range]

    # Create SOI targets with geographic level indicators
    agi_with_labels = soi_targets.assign(
        band=lambda df: df.apply(
            lambda r: get_agi_band_label(r.AGI_LOWER_BOUND, r.AGI_UPPER_BOUND),
            axis=1,
        )
    )
    agi_with_labels = agi_with_labels.sort_values(
        ["IS_COUNT", "VARIABLE", "AGI_LOWER_BOUND"]
    )

    for _, row in agi_with_labels.iterrows():
        geo_id = row["GEO_ID"]
        variable = row["VARIABLE"]
        band = row["band"]
        value = row["VALUE"]

        if geo_id.startswith("0100000US"):
            level_prefix = "national"
        elif geo_id.startswith("0400000US"):
            state_fips_code = geo_id[9:11]
            level_prefix = f"state_{get_state_abbr_from_fips(state_fips_code)}"
        elif geo_id.startswith("5001800US"):
            district_code = geo_id[11:13]
            state_fips_code = geo_id[9:11]
            level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
        else:
            continue

        col_name = f"soi/{level_prefix}/{variable}/{band}"
        targets_dict[col_name] = value

    # Convert to DataFrame
    y = pd.DataFrame([targets_dict])

    return y


def create_state_mask(
    dataset: str = None,
    districts: pd.Series = pd.Series(["5001800US5600"]),
    time_period: int = 2023,
) -> np.ndarray:
    """
    Create a matrix R to accompany the loss matrix M s.t. (W x M) x R = Y_
    where Y_ is the target matrix s.t. no target is constructed
    from weights from a different state.
    """

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period

    household_states = sim.calculate("state_fips").values
    district_states = districts.str[9:11].astype(np.int32)
    r = np.zeros((len(districts), len(household_states)))

    for i in range(len(districts)):
        r[i] = household_states == district_states[i]

    return r


def create_district_to_state_matrix():
    """Create [50, 450] sparse binary matrix mapping states to districts"""

    districts = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age_district.csv"
    ).GEO_ID

    states = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age_state.csv"
    ).GEO_ID

    num_districts = len(districts)
    num_states = len(states)

    district_state_codes = [dist_id[9:11] for dist_id in districts]
    state_codes = [state_id[9:11] for state_id in states]

    # Create mapping from state code to state index (position in the states Series)
    state_code_to_idx = {code: idx for idx, code in enumerate(state_codes)}

    # Create indices and values for sparse tensor
    indices = []
    for dist_idx, state_code in enumerate(district_state_codes):
        if state_code in state_code_to_idx:  # Safety check
            state_idx = state_code_to_idx[state_code]
            indices.append([state_idx, dist_idx])

    # Check if we have any valid mappings
    if not indices:
        raise ValueError(
            "No valid district-to-state mappings found. Check the ID formats."
        )

    # Convert to tensors
    indices = torch.tensor(indices, dtype=torch.long).t()
    values = torch.ones(len(indices[0]), dtype=torch.float)

    # Create sparse tensor
    mapping_matrix = torch.sparse.FloatTensor(
        indices, values, torch.Size([num_states, num_districts])
    )

    return mapping_matrix


def create_households(
    sample_per_district: int,
    age_data_by_district: pd.DataFrame,
    target_names: list,
    dataset: str = "cps_2023",
    time_period: int = 2023,
    states: Optional[pd.Series] = None,  # fips code
):
    """
    Create household assignments and simulation data needed for metric matrix creation.

    Returns:
        tuple: (households_df, sim_calculations_dict, sim_object)
    """
    sim = Microsimulation(dataset=get_dataset(dataset, time_period))
    sim.default_calculation_period = time_period

    if states is not None:
        states = set(states.astype(int))

    # Extract needed variables from target names
    needed_variables = set()
    for target_name in target_names:
        if target_name.startswith("soi/"):
            # Extract variable name from soi target format
            parts = target_name.split("/")
            if len(parts) >= 3:
                var_name = (
                    parts[2].replace("/count", "").replace("/amount", "")
                )
                needed_variables.add(var_name)

    # Always include AGI for SOI filtering
    needed_variables.add("adjusted_gross_income")

    # Calculate all needed variables
    sim_calculations = {}
    for variable in needed_variables:
        if variable in system.variables:
            values = sim.calculate(variable).values
            values_entity = system.variables[variable].entity.key
            if values_entity == "tax_unit":
                sim_calculations[variable] = values
            else:
                sim_calculations[variable] = sim.map_result(
                    values, values_entity, "tax_unit"
                )

    # Create basic household data DataFrame
    data_by_household = pd.DataFrame(
        {
            "state_fips": sim.calculate("state_fips").values,
            "state_code": sim.calculate("state_code").values,
            "cps_weight": sim.calculate("household_weight").values,
        }
    )

    synth_households = []
    for geo_id in age_data_by_district["GEO_ID"]:
        state_fips_code = int(geo_id[9:11])
        district_code = int(geo_id[11:13])

        if states is not None and state_fips_code not in states:
            continue

        pool = data_by_household[
            data_by_household["state_fips"] == state_fips_code
        ]
        sample_ids = pool.sample(sample_per_district, replace=True).index
        synth_households.append(
            pd.DataFrame(
                {
                    "household_id": sample_ids,
                    "district": district_code,
                    "weight": data_by_household.loc[
                        sample_ids, "cps_weight"
                    ].values,
                }
            )
        )

    synth_households = pd.concat(synth_households, ignore_index=True)
    return synth_households, sim_calculations, sim


def calibrate():
    logger.info("Starting calibration...")

    logger.info("Loading data files...")
    age_data_all_levels = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age.csv"
    )
    agi_data_all_levels = pd.read_csv(
        get_data_directory() / "input" / "soi" / "soi_targets.csv"
    )

    logger.info(
        f"Loaded {len(age_data_all_levels)} age rows, {len(agi_data_all_levels)} SOI rows"
    )

    # Focus on specific states and districts to reduce data size
    states = pd.Series(["10"])

    logger.info(
        f"Filtering to {[get_state_abbr_from_fips(fips) for fips in states]} states and their districts..."
    )

    # Create regex pattern from states variable
    state_fips_pattern = "|".join(
        [f"0400000US{fips.zfill(2)}" for fips in states]
    )
    district_fips_pattern = "|".join(
        [f"5001800US{fips.zfill(2)}" for fips in states]
    )
    combined_pattern = f"^({state_fips_pattern}|{district_fips_pattern})"

    age_data_subset = age_data_all_levels[
        age_data_all_levels["GEO_ID"].str.match(combined_pattern)
    ].reset_index(drop=True)

    agi_data_subset = agi_data_all_levels[
        agi_data_all_levels["GEO_ID"].str.match(combined_pattern)
    ].reset_index(drop=True)

    logger.info(
        f"Filtered to {len(age_data_subset)} age rows, {len(agi_data_subset)} SOI rows"
    )

    # Keep district-level data for household creation logic
    age_data_by_district = age_data_subset.loc[
        lambda df: df["GEO_ID"].str.startswith("5001800US")
    ].reset_index(drop=True)

    logger.info(
        f"Creating metric matrix for {len(age_data_subset)} geographic areas..."
    )
    # Create target matrix
    logger.info("Creating target matrix...")
    targets = create_target_matrix(age_data_subset, agi_data_subset)
    target_names = list(targets.columns)

    # Create households and simulation data based on target requirements
    logger.info("Creating households and simulation data...")
    households, sim_calculations, sim = create_households(
        sample_per_district=500,
        age_data_by_district=age_data_by_district,
        target_names=target_names,
        dataset="cps_2023",
        time_period=2023,
        states=states,
    )

    # Create metric matrix
    logger.info("Creating metric matrix with household filtering...")
    data_by_household = create_metric_matrix(
        sim=sim,
        sim_calculations=sim_calculations,
        ages=age_data_subset,
        soi_targets=agi_data_subset,
        households=households,
    )

    logger.info(f"Metric matrix created with shape: {data_by_household.shape}")

    weights = households["weight"].to_numpy(copy=True)

    device = "mps:0" if torch.backends.mps.is_available() else "cpu"

    data_by_household_tensor = torch.tensor(
        data_by_household.drop(columns=["state_code", "state_fips"])
        .astype(float)
        .values,
        dtype=torch.float32,
        device=device,
    )
    households_tensor = torch.tensor(
        households.values, dtype=torch.int64, device=device
    )
    targets = targets.values.flatten()

    def estimate_targets(weights: torch.Tensor) -> torch.Tensor:
        """
        Estimate targets based on the weights for multi-level calibration.

        Args:
            weights: Shape [N] - one weight per (district, household) pair

        Returns:
            Shape [count_targets] - flattened estimated targets for all geographic levels
        """
        # Extract household indices
        household_indices = households_tensor[:, 0]

        # Get household data for the sampled households
        sampled_household_data = data_by_household_tensor[household_indices]

        # Apply weights: multiply each household's demographics by its weight
        weighted_household_data = weights.unsqueeze(1) * sampled_household_data

        # Sum weighted household data across all households
        # This gives us the estimated values for each metric column
        estimated_values = weighted_household_data.sum(dim=0)

        return estimated_values

    def create_target_normalization_factor() -> torch.Tensor:
        """
        Create a normalization factor for the targets based on the geography level of each target.

        This ensures that national, state, and district targets each have equal total weight,
        making the calibration robust to random sampling of targets.

        Returns:
            Normalization factor as a torch tensor.
        """
        target_names_array = np.array(target_names)

        # Identify different geographic levels
        is_national = np.array(
            ["/national" in name for name in target_names_array]
        )
        is_state = np.array(["/state_" in name for name in target_names_array])
        is_district = np.array(
            ["/district_" in name for name in target_names_array]
        )

        # Calculate normalization factors for each level
        national_factor = is_national * (1 / max(is_national.sum(), 1))
        state_factor = is_state * (1 / max(is_state.sum(), 1))
        district_factor = is_district * (1 / max(is_district.sum(), 1))

        # Each geographic level gets equal total weight
        normalization_factor = np.where(
            is_national,
            national_factor,
            np.where(is_state, state_factor, district_factor),
        )

        return torch.tensor(
            normalization_factor, dtype=torch.float32, device=device
        )

    normalization_factor = create_target_normalization_factor()

    # Set to warning logging level

    # logging.basicConfig(level=logging.ERROR)

    logging.info("Initializing calibration...")
    calibration = Calibration(
        targets=targets,
        weights=weights,
        target_names=target_names,
        estimate_function=estimate_targets,
        epochs=512,
        learning_rate=0.2,
        normalization_factor=normalization_factor,
    )

    calibration.calibrate()
    calibration.performance_df.to_csv("calibration_log.csv", index=False)

    return calibration.performance_df


if __name__ == "__main__":
    calibrate()
