import os
#import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
# from tqdm import tqdm
import h5py
from policyengine_us import Microsimulation

from us_congressional_districts.utils import get_data_directory


def create_district_target_matrix(
    dataset: str = "cps_2022",
    time_period: int = 2022,
    reform=None,
):
    ages = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age-district.csv"
    )

    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period

    matrix = pd.DataFrame()
    y = pd.DataFrame()

    age = sim.calculate("age").values
    for age_range in age_ranges:
        if age_range != "85+":
            lower_age, upper_age = age_range.split("-")
            in_age_band = (age >= int(lower_age)) & (age < int(upper_age))
        else:
            in_age_band = age >= 85 

        matrix[f"age/{age_range}"] = sim.map_result(
            in_age_band, "person", "household"
        )

        y[f"age/{age_range}"] = ages[age_range]

    # Mask so that weights cannot be shared among households of different states
    state_mask = create_state_mask(
        household_states = sim.calculate("state_fips").values,
        districts=ages.GEO_ID
    )

    return matrix, y, state_mask


def create_state_mask(
    household_states: np.ndarray, districts: pd.Series
) -> np.ndarray:
    # Create a matrix R to accompany the loss matrix M s.t. (W x M) x R = Y_
    # where Y_ is the target matrix s.t. no target is constructed
    # from weights from a different state.

    district_states = districts.str[9:11].astype(np.int32) 
    r = np.zeros((len(districts), len(household_states)))

    for i in range(len(districts)):
        r[i] = household_states == district_states[i]

    return r


# TODO bring this into loss.py to stay consistent with *-uk-data
def calibrate(
    epochs: int = 128,
):
    matrix_, y_, state_mask = create_district_target_matrix(
        "cps_2022", 2022
    )

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

    metrics = torch.tensor(matrix_.values, dtype=torch.float32)
    y = torch.tensor(y_.values, dtype=torch.float32)
    r = torch.tensor(state_mask, dtype=torch.float32)

    def loss(w):
        pred = (w.unsqueeze(-1) * metrics.unsqueeze(0)).sum(dim=1)
        mse = torch.mean(((pred - y) / y) ** 2)
        return mse

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

    return final_weights


if __name__ == "__main__":
    calibrate()
