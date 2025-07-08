import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import h5py
from huggingface_hub import hf_hub_download
from typing import Optional, Union

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

    targets_dict = {}

    # Create age targets for required geographic levels
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


def create_households(
    sample_per_district: int,
    age_data_subset: pd.DataFrame,
    target_names: list,
    how: list[str],
    dataset: str = "cps_2023",
    time_period: int = 2023,
):
    """
    Create household assignments based on the most granular geography level needed.

    If the district and state levels are needed, the synthetic households created accoding to districting logic will contribute to state targets. This logic applies to all geographic levels: District → State → National.
    """
    sim = Microsimulation(dataset=get_dataset(dataset, time_period))
    sim.default_calculation_period = time_period

    # Calculate needed variables based on target names
    needed_variables = set()
    for target_name in target_names:
        if target_name.startswith("soi/"):
            parts = target_name.split("/")
            if len(parts) >= 3:
                var_name = (
                    parts[2].replace("/count", "").replace("/amount", "")
                )
                needed_variables.add(var_name)
    needed_variables.add("adjusted_gross_income")

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

    # Get base household data
    data_by_household = pd.DataFrame(
        {
            "state_fips": sim.calculate("state_fips").values,
            "state_code": sim.calculate("state_code").values,
            "household_weight": sim.calculate("household_weight").values,
        }
    )

    # Determine the most granular level needed
    needs_district = "district" in how or any(h.isdigit() for h in how)
    needs_state = (
        "state" in how or any(h.isdigit() for h in how) and not needs_district
    )
    needs_national = (
        "national" in how and not needs_state and not needs_district
    )
    states = [h for h in how if h.isdigit()]

    logger.info(
        f"Geography hierarchy: District={needs_district}, State={True if 'state' in how else False}, National={True if 'national' in how else False}"
    )

    # Most granular = District level
    if needs_district:
        synth_households = []

        # Get all districts we need to create
        district_geos = age_data_subset[
            age_data_subset["GEO_ID"].str.startswith("5001800US")
        ]

        for _, row in district_geos.iterrows():
            geo_id = row["GEO_ID"]
            state_fips_code = geo_id[9:11]
            district_code = geo_id[11:13]

            # Skip if we're filtering by specific states
            if len(states) > 0 and state_fips_code not in states:
                continue

            # Sample from households in this state
            pool = data_by_household[
                data_by_household["state_fips"] == int(state_fips_code)
            ]

            sample_ids = pool.sample(
                min(sample_per_district, len(pool)), replace=True
            ).index

            synth_households.append(
                pd.DataFrame(
                    {
                        "household_id": sample_ids,
                        "district": district_code,
                        "state_fips": state_fips_code,
                        "weight": pool.loc[
                            sample_ids, "household_weight"
                        ].values,
                    }
                )
            )

        synth_households = pd.concat(synth_households, ignore_index=True)

    # Most granular = State level (no districts)
    elif needs_state:
        synth_households = []

        logger.warning(f"states: {states}")

        # Determine which states we need
        if len(states) == 0:
            # We select all states in the data
            state_geos = age_data_subset[
                age_data_subset["GEO_ID"].str.startswith("0400000US")
            ]
            states = [geo_id[9:11] for geo_id in state_geos["GEO_ID"]]

            logger.warning(
                f"states after theoretically appending all fips: {states}"
            )

        for state_fips in states:
            pool = data_by_household[
                data_by_household["state_fips"] == int(state_fips)
            ]

            # For state-level, we can use all households in the state
            synth_households.append(
                pd.DataFrame(
                    {
                        "household_id": pool.index,
                        "district": "-1",  # No district assignment
                        "state_fips": state_fips,
                        "weight": pool["household_weight"].values,
                    }
                )
            )

        synth_households = pd.concat(synth_households, ignore_index=True)

    # Only national level
    elif needs_national:
        # All households contribute directly
        fips = data_by_household["state_fips"].values
        fips = str(fips) if len(fips) > 1 else "0" + str(fips)
        synth_households = pd.DataFrame(
            {
                "household_id": np.arange(len(data_by_household)),
                "district": "-1",
                "state_fips": fips,
                "weight": data_by_household["household_weight"].values,
            }
        )

    logger.info(
        f"Created {len(synth_households)} synthetic household assignments"
    )
    if needs_state:
        logger.info(
            f"Created households for states: {synth_households['state_fips'].nunique()}"
        )
    if needs_district:
        logger.info(
            f"Created {sample_per_district} households per district for {len(synth_households['state_fips'].unique())} states"
        )

    return synth_households, sim_calculations, sim


def create_metric_matrix(
    sim: Microsimulation,
    sim_calculations: dict,
    ages: pd.DataFrame,
    soi_targets: pd.DataFrame,
    households: pd.DataFrame,
):
    """
    Create metric matrix where each synthetic household contributes to all
    appropriate geography levels in the hierarchy that are being targeted.
    """
    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    # Pre-calculate age bands for actual households
    age = sim.calculate("age").values

    matrix = pd.DataFrame()

    for i, age_range in enumerate(age_ranges):
        if age_range != "85+":
            lower_age, upper_age = age_range.split("-")
            in_age_band = (age >= int(lower_age)) & (age <= int(upper_age))
        else:
            in_age_band = age >= 85

        # Map to household level
        in_age_band = sim.map_result(
            in_age_band, "person", "household", how="sum"
        )

        unique_geo_ids = ages["GEO_ID"].unique()

        for _, geo_id in enumerate(unique_geo_ids):
            if geo_id.startswith("0100000US"):
                level_prefix = "national"
                # ALL synthetic households contribute to national
                geo_mask = np.ones(len(households), dtype=bool)

            elif geo_id.startswith("0400000US"):
                state_fips_code = geo_id[9:11]
                level_prefix = (
                    f"state_{get_state_abbr_from_fips(state_fips_code)}"
                )
                # All households in this state contribute (whether assigned to districts or not)
                geo_mask = households["state_fips"] == state_fips_code

            elif geo_id.startswith("5001800US"):
                district_code = geo_id[11:13]
                state_fips_code = geo_id[9:11]
                level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
                # Only households assigned to this specific district
                geo_mask = (households["district"] == district_code) & (
                    households["state_fips"] == state_fips_code
                )
            else:
                continue

            # Get values from actual households for the synthetic households selected
            synthetic_in_age_band = in_age_band[
                households["household_id"].values
            ]

            # Apply geographic mask
            combined_mask = synthetic_in_age_band * geo_mask.astype(float)
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

        # Create AGI mask at tax unit level
        agi_values = sim_calculations["adjusted_gross_income"]
        mask = (agi_values > lower) & (agi_values <= upper)

        # Same hierarchical geographic logic
        if geo_id.startswith("0100000US"):
            geo_mask = np.ones(len(households), dtype=bool)
            level_prefix = "national"

        elif geo_id.startswith("0400000US"):
            state_fips_code = geo_id[9:11]
            geo_mask = households["state_fips"] == state_fips_code
            level_prefix = f"state_{get_state_abbr_from_fips(state_fips_code)}"

        elif geo_id.startswith("5001800US"):
            district_code = geo_id[11:13]
            state_fips_code = geo_id[9:11]
            geo_mask = (households["district"] == district_code) & (
                households["state_fips"] == state_fips_code
            )
            level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
        else:
            continue

        # Map tax unit values to synthetic households
        # First map from tax_unit to household
        if is_count:
            tax_unit_metric = mask * (var_values > 0).astype(float)
        else:
            tax_unit_metric = var_values * mask

        household_metric = sim.map_result(
            tax_unit_metric, "tax_unit", "household"
        )

        # Then map from actual household data to the selected synthetic households
        synthetic_metric = household_metric[households["household_id"].values]

        # Apply geographic mask
        final_metric = synthetic_metric * geo_mask.astype(float)

        if is_count:
            col = f"soi/{level_prefix}/{var}/count/{band}"
        else:
            col = f"soi/{level_prefix}/{var}/amount/{band}"

        matrix[col] = final_metric

    return matrix


def subsample_targets(how: list[str]) -> str:
    """Subsample targets to reduce data size for calibration.

    Args:
        age: DataFrame with age targets for all geographic levels.
        soi_targets: DataFrame with SOI targets for all geographic levels.
        how: List of strings specifying the level and or sampling method of targets for calibration.
            Valid options are:
            - "national": Use national targets only.
            - "state": Use state-level targets.
            - "district": Use district-level targets.
            - a combination of the above, e.g. ["national", "state"].
            - a list of specific state FIPS codes to use, e.g. ["06", "12"], this will calibrate the state and district targets corresponding to the fips codes.
    Returns:
        str: combined regex pattern for filtering targets.
    """
    for h in how:
        if (h not in ["national", "state", "district"]) and (
            int(h) < 0 or int(h) > 56
        ):
            logger.error(
                f"Invalid 'how' argument value: {h}. Expected 'national', 'state', 'district', or a list of state FIPS codes."
            )
    geo_code_patterns = []

    if "national" in how:
        geo_code_patterns.append("0100000US")
    if "state" in how:
        geo_code_patterns.append("0400000US")
    if "district" in how:
        geo_code_patterns.append("5001800US")
    for h in how:
        if h not in ["national", "state", "district"]:
            geo_code_patterns.append(f"0400000US{h.zfill(2)}")
            geo_code_patterns.append(f"5001800US{h.zfill(2)}")

    return f"^({'|'.join(geo_code_patterns)})"


def calibrate(
    how: Optional[Union[list[str], str]] = ["national"],
    initial_weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Calibrate US national, state, and district-level targets using microcalibrate for age and soi variables.

    Args:
        how: str or list[str] specifying the level and or sampling method of targets for calibration. Default is "national".
            Valid options are:
            - "national": Use national targets only.
            - "state": Use state-level targets.
            - "district": Use district-level targets.
            - a combination of the above, e.g. ["national", "state"].
            - a list of specific state FIPS codes to use, e.g. ["06", "12"], this will calibrate the state and district targets corresponding to the fips codes.

    Returns:
        pd.DataFrame: Performance DataFrame from the calibration process.
    """
    if not isinstance(how, list) and not isinstance(how, str):
        logger.error(
            f"Invalid 'how' argument type: {type(how)}. Expected str or list[str]."
        )
    if isinstance(how, str):
        how = [how]

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

    # Focus on specific geography levels or fips codes to reduce data size
    geo_code_pattern = subsample_targets(how)

    logger.info(f"Filtering targets in {how} mode")

    age_data_subset = age_data_all_levels[
        age_data_all_levels["GEO_ID"].str.match(geo_code_pattern)
    ].reset_index(drop=True)

    agi_data_subset = agi_data_all_levels[
        agi_data_all_levels["GEO_ID"].str.match(geo_code_pattern)
    ].reset_index(drop=True)

    logger.info(
        f"Filtered to {len(age_data_subset)} age rows, {len(agi_data_subset)} SOI rows"
    )

    # Keep district-level data for household creation logic
    age_data_by_district = age_data_subset.loc[
        lambda df: df["GEO_ID"].str.startswith("5001800US")
    ].reset_index(drop=True)

    logger.info("Creating target matrix...")
    targets = create_target_matrix(age_data_subset, agi_data_subset)
    target_names = list(targets.columns)

    logger.info(f"{len(target_names)} targets were created")

    logger.info(f"Creating households and simulation data with {how} mode...")
    synth_households, sim_calculations, sim = create_households(
        sample_per_district=500,
        age_data_subset=age_data_subset,
        target_names=target_names,
        how=how,
        dataset="cps_2023",
        time_period=2023,
    )

    logger.info("Creating metric matrix with household filtering...")
    data_by_household = create_metric_matrix(
        sim=sim,
        sim_calculations=sim_calculations,
        ages=age_data_subset,
        soi_targets=agi_data_subset,
        households=synth_households,
    )

    logger.info(f"Metric matrix created with shape: {data_by_household.shape}")

    target_set = set(target_names)
    metric_set = set(data_by_household.columns)
    missing = target_set - metric_set
    if missing:
        logger.warning(f"Missing columns in metric matrix: {missing}")

    if initial_weights is not None:
        weights = initial_weights
    else:
        weights = np.ones(len(synth_households))

    device = "mps:0" if torch.backends.mps.is_available() else "cpu"

    data_by_household_tensor = torch.tensor(
        data_by_household.astype(float).values,
        dtype=torch.float32,
        device=device,
    )
    targets = targets.values.flatten()

    def estimate_targets(weights: torch.Tensor) -> torch.Tensor:
        """
        Estimate targets based on the weights for multi-level calibration.

        Args:
            weights: Shape [N] - one weight per synthetic household

        Returns:
            Shape [count_targets] - flattened estimated targets for all geographic levels
        """
        # No need to extract indices - data_by_household_tensor is already aligned
        # with synthetic households

        # Apply weights: multiply each household's metrics by its weight
        weighted_household_data = (
            weights.unsqueeze(1) * data_by_household_tensor
        )

        # Sum weighted household data across all households
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
        normalization_factor=(
            normalization_factor
            if (len(how) > 1 and not any(h.isdigit() for h in how))
            else None
        ),
    )

    calibration.calibrate()
    calibration.performance_df.to_csv("calibration_log.csv", index=False)

    return calibration.performance_df, calibration.weights


if __name__ == "__main__":
    calibrate()
