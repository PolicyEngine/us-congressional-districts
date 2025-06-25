from us_congressional_districts.calibrate import (
    create_district_metric_matrix,
    get_dataset,
    create_target_matrix,
    get_data_directory,
)
import pandas as pd
import numpy as np
import torch


def create_households(
    sample_per_district: int, data_by_household: pd.DataFrame
):
    synth_households = pd.DataFrame()
    for district in age_data_by_district.index:
        households_in_district = pd.DataFrame(
            {
                "household_id": data_by_household.sample(
                    sample_per_district
                ).index.values,
            }
        )
        households_in_district["district"] = district
        synth_households = pd.concat(
            [synth_households, households_in_district]
        )

    return synth_households


def create_target_names(targets_by_district: pd.DataFrame) -> list:
    targets = []
    for district in targets_by_district.index:
        for age_band in targets_by_district.columns:
            targets.append(f"{district}/{age_band}")
    return np.array(targets)


age_data_by_district = pd.read_csv(
    get_data_directory() / "input" / "demographics" / "age_district.csv"
)

data_by_household = create_district_metric_matrix(
    dataset=get_dataset("cps_2023", 2023),
    ages=age_data_by_district,
    time_period=2023,
)

targets_by_district = create_target_matrix(age_data_by_district)
count_districts, count_targets = targets_by_district.shape
target_names = create_target_names(targets_by_district)

households = create_households(
    sample_per_district=100, data_by_household=data_by_household
)
weights = np.ones(len(households)) * (150e6 / len(households))

device = "mps:0" if torch.backends.mps.is_available() else "cpu"

weights = torch.tensor(weights, dtype=torch.float32, device=device)
data_by_household = torch.tensor(
    data_by_household.values, dtype=torch.float32, device=device
)
targets_by_district = torch.tensor(
    targets_by_district.values, dtype=torch.float32, device=device
)
households = torch.tensor(households.values, dtype=torch.int64, device=device)
targets = targets_by_district.flatten()


def estimate_targets(weights: torch.Tensor) -> torch.Tensor:
    """
    Estimate targets based on the weights.

    Args:
        weights: Shape [43500] - one weight per (district, household) pair

    Returns:
        Shape [435*18] - flattened estimated targets for all districts and age bands
    """
    # Extract household and district indices (note the order!)
    household_indices = households[
        :, 0
    ]  # Shape: [43500] - actual household IDs
    district_indices = households[
        :, 1
    ]  # Shape: [43500] - district indices (0-434)

    # Get household data for the sampled households
    sampled_household_data = data_by_household[household_indices]

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


estimates = estimate_targets(weights)

targets = targets.cpu().numpy()
weights = weights.cpu().numpy()

from microcalibrate import Calibration
import logging

# Set to warning logging level

logging.basicConfig(level=logging.ERROR)

calibration = Calibration(
    targets=targets,
    weights=weights,
    target_names=target_names,
    estimate_function=estimate_targets,
    epochs=128,
    learning_rate=0.1,
)

calibration.calibrate()
calibration.performance_df.to_csv("calibration_log.csv", index=False)
