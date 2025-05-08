from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from policyengine_core.data import Dataset
from policyengine_us import Microsimulation

from us_congressional_districts.utils import get_data_directory
from us_congressional_districts.calibrate import calibrate

# This is the setup
calibrate()

weights_h5 = h5py.File(Path(get_data_directory() / 'output' / 'congressional_district_weights.h5'))
cps = Dataset.from_file(Path(get_data_directory() / 'input' / 'cps' / 'cps_2023.h5'), 2023)
ages = pd.read_csv(Path(get_data_directory() / 'input' / 'demographics' / 'age_district.csv'))

districts = list(ages.GEO_ID)

# TODO: You should go ahead and get true GEOs too, and make sure the districts line up

# Test that the estimated number of 10, 11, 12, and 13 year olds in
# district '5001800US0101' is close to the true value
assert(districts[0] == '5001800US0101')

district_target = ages.iloc[0]['10-14']

# Ballpark district target from online sources
assert(district_target > 30000)
assert(district_target < 60000)

cps_h5 = cps.load()
cps_ages = cps_h5['age'][:]

sim = Microsimulation(dataset = cps)
sim.default_calculation_period = 2023

# Estimate the number of 10, 11, 12, and 13 year olds in district '5001800US0101'
person_age = sim.calculate('age')

# Assert that the simulation calculation did not alter ordering of CPS ages
assert(all(person_age.values == cps_ages))

in_age_band_person = (person_age >= 10) & (person_age < 14)
in_age_band_hh = sim.map_result(in_age_band_person, "person", "household")

# Assert that age band counts are non-negative integers of less than 10
assert np.all((in_age_band_hh >= 0) & (in_age_band_hh == np.floor(in_age_band_hh))), "Values must be non-negative integers"
assert np.max(in_age_band_hh) < 10, "Max value must be less than 20"

# Find a household with 2 members in the age band, and find the people individually
hh_with_2_in_range_pos = np.where(in_age_band_hh == 2)[0][0]
hh_with_2_in_range_id = cps_h5['household_id'][hh_with_2_in_range_pos]

persons_in_hh_with_2_in_range_pos = np.where(cps_h5['person_household_id'] == hh_with_2_in_range_id)

hh_ages = cps_ages[persons_in_hh_with_2_in_range_pos]
assert(np.sum([a in [10, 11, 12, 13] for a in hh_ages]) == 2)

# Test that the shape of the weights corresponds to households
w = f['2023'][:]
assert(w.shape[0] == 435)
assert(in_age_band_hh.shape[0] == w.shape[1])

district_weights = w[0, :]
district_estimate = np.dot(district_weights, in_age_band_hh)

assert(np.abs(district_estimate - district_target) / district_target < .01)

# TODO: test that a household with positive weights in this district has
# zero weight in every other
