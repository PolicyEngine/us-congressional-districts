import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import h5py
from huggingface_hub import hf_hub_download

from policyengine_core.data import Dataset
from policyengine_us import Microsimulation
from us_congressional_districts.utils import get_data_directory
import pandas as pd
import numpy as np
import torch
from microcalibrate import Calibration
import logging

# TODO (baogorek): A task is to use the mapping matrix
from us_congressional_districts.district_mapping import (
    get_district_mapping_matrix,
)


matrix_path = Path(
    get_data_directory(), "input", "geographies", "district_mapping.csv"
)

# Mapping matrix logic -----
mapping_df = pd.read_csv(matrix_path)
old_codes = sorted(mapping_df.code_old.unique())
new_codes = sorted(mapping_df.code_new.unique())

assert (
    len(old_codes) == len(new_codes) == 435
), "Still not 435×435 after filtering!"

old_index = {c: i for i, c in enumerate(old_codes)}
new_index = {c: j for j, c in enumerate(new_codes)}

# 3)  Allocate the empty matrix and populate it row-by-row ──────────────────
mapping_matrix = np.zeros((435, 435), dtype=float)

for row in mapping_df.itertuples(index=False):
    i = old_index[row.code_old]
    j = new_index[row.code_new]
    mapping_matrix[i, j] = row.proportion

assert np.allclose(mapping_matrix.sum(axis=1), 1.0), "Row totals aren't 1.0"


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
    if np.isneginf(lower) or lower <= 0:
        return f"under_{int(upper)}"
    elif np.isposinf(upper):
        return f"{int(lower)}_plus"
    else:
        return f"{int(lower)}_{int(upper)}"


def create_district_metric_matrix(
    dataset: str = None,
    ages: pd.DataFrame = pd.DataFrame(),
    agi_targets: pd.DataFrame = pd.DataFrame(),
    time_period: int = 2023,
):
    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period

    age = sim.calculate("age").values
    agi = sim.calculate("adjusted_gross_income").values
    state_code = sim.calculate("state_code").values

    matrix = pd.DataFrame()

    for age_range in age_ranges:
        if age_range != "85+":
            lower_age, upper_age = age_range.split("-")
            in_age_band = (age >= int(lower_age)) & (age < int(upper_age))
        else:
            in_age_band = age >= 85

        matrix[f"age/{age_range}"] = sim.map_result(
            in_age_band, "person", "household"
        )

    agi_long = agi_targets[
        ["AGI_LOWER_BOUND", "AGI_UPPER_BOUND", "VARIABLE"]
    ].drop_duplicates()  # drop duplicates to avoid redundancy calculating the same band multiple times

    for _, row in agi_long.iterrows():
        lower, upper, variable = (
            row.AGI_LOWER_BOUND,
            row.AGI_UPPER_BOUND,
            row.VARIABLE,
        )
        band = get_agi_band_label(lower, upper)

        in_band = (agi > lower) & (agi <= upper)
        if variable is not None:
            matrix[f"agi/{variable}/{band}"] = sim.map_result(
                in_band, "tax_unit", "household"
            )
        else:
            matrix[f"agi/{band}"] = sim.map_result(
                in_band, "tax_unit", "household"
            )

    matrix["state_code"] = state_code

    return matrix


def create_target_matrix(ages, agi_targets):
    """
    Create an aggregate target matrix for the appropriate geographic area

    Args:
        ages: a data frame containing GEO_ID and GEO_NAME as the first two columns,
          with target variables afterwards
        agi_targets: a data frame containing GEO_ID and GEO_NAME as the first and last columns,
    """
    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    y = pd.DataFrame()
    for age_range in age_ranges:
        y[f"age/{age_range}"] = ages[age_range]

    agi_with_labels = agi_targets.assign(
        band=lambda df: df.apply(
            lambda r: get_agi_band_label(r.AGI_LOWER_BOUND, r.AGI_UPPER_BOUND),
            axis=1,
        )
    )

    for variable, df_var in agi_with_labels.groupby("VARIABLE"):
        for band, df_band in df_var.groupby("band"):
            if variable is not None:
                y[f"agi/{variable}/{band}"] = df_band["VALUE"].values
            else:
                y[f"agi/{band}"] = df_band["VALUE"].values

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


def create_target_names(
    targets_by_district: pd.DataFrame, district_names: np.ndarray
) -> list:
    targets = []
    for district in district_names:
        for age_band in targets_by_district.columns:
            targets.append(f"{district}/{age_band}")
    return np.array(targets)


def calibrate():
    age_data_by_district = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age_district.csv"
    )
    agi_data_by_district = pd.read_csv(
        get_data_directory() / "input" / "soi" / "agi_district.csv"
    )

    target_district_names = age_data_by_district.GEO_NAME

    data_by_household = create_district_metric_matrix(
        dataset=get_dataset("cps_2023", 2023),
        ages=age_data_by_district,
        agi_targets=agi_data_by_district,
        time_period=2023,
    )

    targets_by_district = create_target_matrix(
        age_data_by_district, agi_data_by_district
    )
    count_districts, count_targets = targets_by_district.shape
    target_names = create_target_names(
        targets_by_district, target_district_names
    )

    households = create_households(
        sample_per_district=1_000,
        data_by_household=data_by_household,
        age_data_by_district=age_data_by_district,
    )
    weights = np.ones(len(households)) * (150e6 / len(households))

    device = "mps:0" if torch.backends.mps.is_available() else "cpu"

    data_by_household_tensor = torch.tensor(
        data_by_household.drop(columns=["state_code"]).values,
        dtype=torch.float32,
        device=device,
    )
    households_tensor = torch.tensor(
        households.values, dtype=torch.int64, device=device
    )
    targets = targets_by_district.values.flatten()

    def estimate_targets(weights: torch.Tensor) -> torch.Tensor:
        """
        Estimate targets based on the weights.

        Args:
            weights: Shape [43500] - one weight per (district, household) pair

        Returns:
            Shape [435*36] - flattened estimated targets for all districts and target variables (currently age, agi count, agi amount)
        """
        # Extract household and district indices (note the order!)
        household_indices = households_tensor[
            :, 0
        ]  # Shape: [435000] - actual household IDs
        district_indices = households_tensor[
            :, 1
        ]  # Shape: [435000] - district indices (0-434)

        # Get household data for the sampled households
        sampled_household_data = data_by_household_tensor[household_indices]

        # Apply weights: multiply each household's demographics by its weight
        weighted_household_data = weights.unsqueeze(1) * sampled_household_data

        # Sum weighted household data by district
        estimated_targets = torch.zeros(
            count_districts,
            count_targets,
            dtype=torch.float32,
            device=weights.device,
        )
        estimated_targets.scatter_add_(
            0,
            district_indices.unsqueeze(1).expand(-1, count_targets),
            weighted_household_data,
        )

        return estimated_targets.flatten()

    # Set to warning logging level

    logging.basicConfig(level=logging.ERROR)

    calibration = Calibration(
        targets=targets,
        weights=weights,
        target_names=target_names,
        estimate_function=estimate_targets,
        epochs=128,
        learning_rate=0.2,
    )

    calibration.calibrate()
    calibration.performance_df.to_csv("calibration_log.csv", index=False)


if __name__ == "__main__":
    calibrate()
