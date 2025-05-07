import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import h5py
from policyengine_us import Microsimulation

from us_congressional_districts.utils import get_data_directory


def create_district_metric_matrix(
    dataset: str = "cps_2022",
    ages: pd.DataFrame = pd.DataFrame(),
    time_period: int = 2022
):
    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period

    age = sim.calculate("age").values

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

    return matrix


def create_target_matrix(ages):
    """
    Create an aggregate target matrix for the appropriate geographic area

    Args:
        ages: a data frame containing GEO_ID and NAME as the first two columns,
          with target variables afterwards
    """
    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    y = pd.DataFrame()
    for age_range in age_ranges:
        y[f"age/{age_range}"] = ages[age_range]

    return y


def create_state_mask(
    dataset: str = "cps_2022",
    districts: pd.Series = pd.Series(['5001800US5600']),
    time_period: int = 2022
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
        get_data_directory() / "input" / "demographics" / "age-district.csv"
    ).GEO_ID

    states = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age-state.csv"
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
        raise ValueError("No valid district-to-state mappings found. Check the ID formats.")
    
    # Convert to tensors
    indices = torch.tensor(indices, dtype=torch.long).t()
    values = torch.ones(len(indices[0]), dtype=torch.float)
    
    # Create sparse tensor
    mapping_matrix = torch.sparse.FloatTensor(
        indices, values, 
        torch.Size([num_states, num_districts])
    )
    
    return mapping_matrix 


def calibrate(
    epochs: int = 128,
    overwrite_ecps = True
):

    # Target data sets (there's probably a better way to do this)
    ages_district = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age-district.csv"
    )

    ages_state = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age-state.csv"
    )

    ages_national = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age-national.csv"
    )

    # the metrics matrix
    matrix_ = create_district_metric_matrix("cps_2022", ages_district, 2022)
    state_mask = create_state_mask("cps_2022", ages_district.GEO_ID, 2022)
    
    y_= create_target_matrix(ages_district)
    y_national_= create_target_matrix(ages_national)
    y_state_= create_target_matrix(ages_state)

    sim = Microsimulation(dataset = "cps_2022")
    sim.default_calculation_period = 2022

    COUNT_DISTRICTS = 435 

    original_weights = np.log(
        sim.calculate("household_weight", 2022).values / COUNT_DISTRICTS
    )

    weights = torch.tensor(
        np.ones((COUNT_DISTRICTS, len(original_weights)))
        * original_weights,
        dtype=torch.float32,
        requires_grad=True,
    )

    # PyTorch metrics matrix
    metrics = torch.tensor(matrix_.values, dtype=torch.float32)

    # PyTorch targets for different geographic aggregations
    y = torch.tensor(y_.values, dtype=torch.float32)
    y_national = torch.tensor(y_national_.values, dtype=torch.float32)
    y_state = torch.tensor(y_state_.values, dtype=torch.float32)

    r = torch.tensor(state_mask, dtype=torch.float32)

    district_to_state_matrix = create_district_to_state_matrix()

    def loss(w):
        pred = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        mse = torch.mean(((pred - y) / y) ** 2)

        pred_n = (w.sum(axis=0) * metrics.T).sum(axis=1)
        mse_n = torch.mean(((pred_n - y_national) / y_national) ** 2)

        pred_s = torch.sparse.mm(district_to_state_matrix, pred)
        mse_s = torch.mean(((pred_s - y_state) / y_state) ** 2)

        return mse + mse_n + mse_s

    optimizer = torch.optim.Adam([weights], lr=0.15)

    desc = range(32) if os.environ.get("DATA_LITE") else range(epochs)
    final_weights = (torch.exp(weights) * r).detach().numpy()

    for epoch in desc:
        optimizer.zero_grad()
        weights_ = torch.exp(weights) * r
        l = loss(weights_)
        l.backward()
        optimizer.step()

        if epoch % 1 == 0:
            print(f"Loss: {l.item()}, Epoch: {epoch}")
        if epoch % 10 == 0:
            final_weights = (torch.exp(weights) * r).detach().numpy()

            with h5py.File(
                get_data_directory()
                / "output"
                / "congressional_district_weights.h5", "w"
            ) as f:
                f.create_dataset("2022", data=final_weights)

            # TODO: figure out strategy for ensuring that the file to overwrite
            # is here and ready to be read from
            #
            #if overwrite_ecps:
            #    with h5py.File(
            #        get_data_directory() / "output" / "enhanced_cps_2022.h5",
            #        "r+"
            #    ) as f:
            #        if "household_weight/2022" in f:
            #            del f["household_weight/2022"]
            #        f.create_dataset(
            #            "household_weight/2022", data=final_weights.sum(axis=0)
            #        )

            #        if "district_weight/2022" in f:
            #            del f["district_weight/2022"]
            #        f.create_dataset(
            #            "district_weight/2022", data=final_weights.sum(axis=1)
            #        )

            #        if "state_weight/2022" in f:
            #            del f["state_weight/2022"]
            #        f.create_dataset(
            #            "state_weight/2022",
            #            data=(
            #                district_to_state_matrix.to_dense().numpy()
            #                @ final_weights.sum(axis=1)
            #            )
            #        )

    return final_weights


if __name__ == "__main__":
    calibrate()
