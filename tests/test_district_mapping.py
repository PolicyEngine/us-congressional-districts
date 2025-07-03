from pathlib import Path

import numpy as np
import pandas as pd

from us_congressional_districts.utils import get_data_directory
from us_congressional_districts.district_mapping import (
    get_district_mapping_matrix,
)


def test_mapping_matrix():
    mapping_matrix = get_district_mapping_matrix()
    assert mapping_matrix.shape[0] == mapping_matrix.shape[1] == 436

    total_elements = 436 * 436
    diag_avg = np.trace(mapping_matrix) / 436
    total_avg = mapping_matrix.sum() / total_elements
    offdiag_avg = (mapping_matrix.sum() - np.trace(mapping_matrix)) / (
        total_elements - 436
    )
    assert (
        diag_avg > offdiag_avg
    ), "Diagonal average is not greater than off-diagonal average"


def test_mapping_matrix_precursor():
    mapping_path = Path(
        get_data_directory(), "input", "geographies", "district_mapping.csv"
    )
    mapping_df = pd.read_csv(mapping_path)

    # Alaska has remained a single district state
    AK_rows = mapping_df.loc[mapping_df.code_old == "5001800US0200"]
    assert AK_rows.shape[0] == 1
    assert AK_rows[["proportion"]].values[0] == 1.0

    # North Carolina added a 14th district
    NC_rows = mapping_df.loc[mapping_df.code_new == "5001800US3714"]
    assert all(NC_rows[["proportion"]].values[0] < 1.0)

    # Check that the position of one number is identical:
    mapping_matrix = get_district_mapping_matrix()

    sorted_old = sorted(mapping_df.code_old.unique())
    sorted_new = sorted(mapping_df.code_new.unique())

    # Ensure 5001800US3712,5001800US3714,0.12120554112861055 was encoded well
    old_i = sorted_old.index("5001800US3712")
    new_j = sorted_new.index("5001800US3714")
    matrix_value_ij = mapping_matrix[old_i, new_j]
    np.testing.assert_almost_equal(matrix_value_ij, 0.1212055411, decimal=6)


def test_redistricting_transformation():
    matrix_path = Path(
        get_data_directory(), "input", "geographies", "district_mapping.csv"
    )
    mapping_df = pd.read_csv(matrix_path)

    old_codes = sorted(mapping_df.code_old.unique())
    new_codes = sorted(mapping_df.code_new.unique())

    # Start with a vector of ones
    v_old = np.ones((436, 1))

    # Identify NC-related rows using FIPS code '37'
    nc_old_indexes = np.where([s[-4:-2] == "37" for s in old_codes])[0]
    nc_new_indexes = np.where([s[-4:-2] == "37" for s in new_codes])[0]

    assert len(nc_old_indexes) == 13
    assert len(nc_new_indexes) == 14

    # Assign higher weight to NC rows in the old vector
    v_old[nc_old_indexes] = 500

    # Apply transformation (e.g., redistribution matrix)
    mapping_matrix = get_district_mapping_matrix()
    v_new = mapping_matrix.T @ v_old

    # Confirm the affected rows match expectations
    assert np.array_equal(np.where(v_new > 50)[0], nc_new_indexes)

    # Reset to a vector of ones
    v_old = np.ones((436, 1))

    # Identify DC-related rows using FIPS code '11'
    dc_old_indexes = np.where([s[-4:-2] == "11" for s in old_codes])[0]
    dc_new_indexes = np.where([s[-4:-2] == "11" for s in new_codes])[0]

    assert len(dc_old_indexes) == 1
    assert len(dc_new_indexes) == 1

    v_old[dc_old_indexes] = 500
    v_new = mapping_matrix.T @ v_old

    assert np.array_equal(np.where(v_new > 50)[0], dc_new_indexes)
