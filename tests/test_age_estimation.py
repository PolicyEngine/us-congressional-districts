from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from policyengine_core.data import Dataset
from policyengine_us import Microsimulation

from us_congressional_districts.utils import get_data_directory
from us_congressional_districts.calibrate import calibrate


def test_sanity():
    # This is the setup
    calibrate()
    
    weights_h5 = h5py.File(Path(get_data_directory() / 'output' / 'congressional_district_weights.h5'))
    cps = Dataset.from_file(Path(get_data_directory() / 'input' / 'cps' / 'cps_2023.h5'), 2023)
    ages = pd.read_csv(Path(get_data_directory() / 'input' / 'demographics' / 'age_district.csv'))
    districts_official = pd.read_csv(Path(get_data_directory() / 'input' / 'geographies' / 'districts.csv'))
    
    # Compare districts with age to the districts from geographies
    districts_from_ages = list(ages.GEO_ID)
    
    assert len([d for d in districts_from_ages if d not in list(districts_official.GEO_ID)]) == 0
    assert len([d for d in districts_official.GEO_ID if d not in list(districts_from_ages)]) == 0
    
    # Test that the estimated number of 10, 11, 12, and 13 year olds in
    # district '5001800US0101' is close to the true value
    assert districts_from_ages[0] == '5001800US0101'
    assert list(districts_official.GEO_ID)[0] == '5001800US0101'
    
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
    assert np.all((in_age_band_hh >= 0) & (in_age_band_hh == np.floor(in_age_band_hh)))
    assert np.max(in_age_band_hh) < 10
    
    # Find a household with 2 members in the age band, and find the people individually
    hh_with_2_in_range_pos = np.where(in_age_band_hh == 2)[0][0]
    hh_with_2_in_range_id = cps_h5['household_id'][hh_with_2_in_range_pos]
    
    persons_in_hh_with_2_in_range_pos = np.where(cps_h5['person_household_id'] == hh_with_2_in_range_id)
    
    hh_ages = cps_ages[persons_in_hh_with_2_in_range_pos]
    assert np.sum([a in [10, 11, 12, 13] for a in hh_ages]) == 2
    
    # Test that the shape of the weights corresponds to households
    w = weights_h5['2023'][:]
    assert w.shape[0] == 435
    assert in_age_band_hh.shape[0] == w.shape[1]
    
    district_weights = w[0, :]
    district_estimate = np.dot(district_weights, in_age_band_hh)
    
    assert np.abs(district_estimate - district_target) / district_target < .01
    
    # test that a household with positive weights in this district has
    # zero weight in every other
    district_pos = np.where(w[:, hh_with_2_in_range_pos] > 0)[0]
    districts = districts_official.iloc[district_pos, 0].values
    states_from_districts = [s[9:11] for s in districts]

    # all districts should belong to the same state
    assert len(np.unique(states_from_districts)) == 1
    
    # This state should have exactly as many positive weight districts for the hh
    assert len([d for d in districts_from_ages if d[9:11] == states_from_districts[0]]) == len(districts)
    
    # Within a district, the weights should not all be clustered on a few households 
    hh_single_state_pos = np.where(sim.calculate('state_fips').values == int(states_from_districts[0]))[0]
    hh_single_state_weights = w[np.ix_(district_pos, hh_single_state_pos)].flatten()

    # We'll say (for now) that the weights are sufficiently spread out when the
    # Coefficient of Variation is less than 150%
    assert np.std(hh_single_state_weights) / np.mean(hh_single_state_weights) < 1.5
