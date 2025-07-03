import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import h5py
from huggingface_hub import hf_hub_download

from policyengine_core.data import Dataset
from policyengine_us import Microsimulation
from policyengine_us.system import system
from us_congressional_districts.utils import get_data_directory
import pandas as pd
import numpy as np
import torch
from microcalibrate import Calibration
import logging


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


def create_metric_matrix(
    dataset: str = None,
    ages: pd.DataFrame = pd.DataFrame(),
    soi_targets: pd.DataFrame = pd.DataFrame(),
    time_period: int = 2023,
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

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period

    age = sim.calculate("age").values

    soi_target_variables = (
        soi_targets["VARIABLE"]
        .str.replace(r"/(count|amount)", "", regex=True)
        .unique()
    )

    sim_calculations = {}
    for variable in soi_target_variables:
        values = sim.calculate(variable).values
        values_entity = system.variables[variable].entity.key
        if values_entity == "tax_unit":
            sim_calculations[variable] = values
        else:
            # ensure all variables are mapped to household level (calibration happens for household weights)
            sim_calculations[variable] = sim.map_result(
                values, values_entity, "tax_unit"
            )

    state_code = sim.calculate("state_code").values
    state_fips = sim.calculate("state_fips").values

    matrix = pd.DataFrame()

    for age_range in age_ranges:
        if age_range != "85+":
            lower_age, upper_age = age_range.split("-")
            in_age_band = (age >= int(lower_age)) & (age < int(upper_age))
        else:
            in_age_band = age >= 85

        # First map age band to household level
        in_age_band = sim.map_result(in_age_band, "person", "household")

        # Create age metrics for each geographic level
        for geo_id in ages["GEO_ID"].unique():
            if geo_id.startswith("0100000US"):
                level_prefix = "national"
                geo_mask = np.ones(len(in_age_band), dtype=bool)
            elif geo_id.startswith("0400000US"):
                state_fips_code = int(geo_id[9:11])
                level_prefix = f"state_{state_fips_code:02d}"
                geo_mask = state_fips == state_fips_code
            elif geo_id.startswith("5001800US"):
                district_code = geo_id[9:13]
                state_fips_code = int(geo_id[9:11])
                level_prefix = f"district_{district_code}"
                geo_mask = state_fips == state_fips_code
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
            geo_mask = np.ones(len(state_fips), dtype=bool)
            level_prefix = "national"
        elif geo_id.startswith("0400000US"):
            state_fips_code = int(geo_id[9:11])
            geo_mask = state_fips == state_fips_code
            level_prefix = f"state_{state_fips_code:02d}"
        elif geo_id.startswith("5001800US"):
            district_code = geo_id[9:13]
            state_fips_code = int(geo_id[9:11])
            geo_mask = state_fips == state_fips_code
            level_prefix = f"district_{district_code}"
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
            state_fips_code = int(geo_id[9:11])
            level_prefix = f"state_{state_fips_code:02d}"
        elif geo_id.startswith("5001800US"):
            district_code = geo_id[9:13]
            level_prefix = f"district_{district_code}"
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
            state_fips_code = int(geo_id[9:11])
            level_prefix = f"state_{state_fips_code:02d}"
        elif geo_id.startswith("5001800US"):
            district_code = geo_id[9:13]
            level_prefix = f"district_{district_code}"
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
    data_by_household: pd.DataFrame,
    age_data_by_district: pd.DataFrame,
):
    synth_households = pd.DataFrame()
    state_codes = age_data_by_district.GEO_NAME.apply(lambda x: x[:2])
    for district in age_data_by_district.index:
        state_subset = data_by_household[
            data_by_household["state_code"] == state_codes[district]
        ]
        households_in_district = pd.DataFrame(
            {
                "household_id": state_subset.sample(
                    sample_per_district, replace=True
                ).index.values,
            }
        )
        households_in_district["district"] = district
        synth_households = pd.concat(
            [synth_households, households_in_district]
        )

    return synth_households


def calibrate():
    age_data_all_levels = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age.csv"
    )
    agi_data_all_levels = pd.read_csv(
        get_data_directory() / "input" / "soi" / "soi_targets.csv"
    )

    # Keep district-level data for household creation logic
    age_data_by_district = age_data_all_levels.loc[
        lambda df: df["GEO_ID"].str.startswith("5001800US")
    ].reset_index(drop=True)

    data_by_household = create_metric_matrix(
        dataset=get_dataset("cps_2023", 2023),
        ages=age_data_all_levels,
        soi_targets=agi_data_all_levels,
        time_period=2023,
    )

    targets = create_target_matrix(age_data_all_levels, agi_data_all_levels)

    target_names = list(targets.columns)

    households = create_households(
        sample_per_district=1_000,
        data_by_household=data_by_household,
        age_data_by_district=age_data_by_district,
    )
    weights = np.ones(len(households)) * (150e6 / len(households))

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

    # Set to warning logging level

    logging.basicConfig(level=logging.ERROR)

    calibration = Calibration(
        targets=targets,
        weights=weights,
        target_names=target_names,
        estimate_function=estimate_targets,
        epochs=512,
        learning_rate=0.2,
    )

    calibration.calibrate()
    calibration.performance_df.to_csv("calibration_log.csv", index=False)

    return calibration.performance_df


if __name__ == "__main__":
    calibrate()
