from itertools import islice

import numpy as np
from isingchat.ising import (
    compatible_projections,
    energy_thermo_limit,
    energy_thermo_limit_dense,
    energy_thermo_limit_fast,
    make_spin_proj_table,
    norm_sparse_log_transfer_matrix,
    norm_sparse_log_transfer_matrix_fast,
    spin_projections,
)


def test_energy_data():
    """"""
    temp = 0.6
    mag_field = 1.5
    int_power = 2
    interactions = np.array([1, 0.5]) ** int_power
    num_neighbors = len(interactions)
    spin_proj_table = make_spin_proj_table(num_neighbors)
    data = energy_thermo_limit_dense(
        temp,
        mag_field,
        interactions=interactions,
        spin_proj_table=spin_proj_table,
    )
    print(data)


def test_gen_projections():
    """Verify that the compatible projections are correct."""
    num_neighbors = 4
    proj_table = make_spin_proj_table(num_neighbors)
    compat_projections = compatible_projections(num_neighbors)
    num_rows = 2 ** num_neighbors
    ref_proj_gen = islice(compat_projections, 0, 2 * num_rows, 2)
    for idx, proj_set_pair in enumerate(ref_proj_gen):
        ref_idx, _ = proj_set_pair
        ref_proj = spin_projections(ref_idx, num_neighbors)
        proj_idx = proj_table[idx]
        assert np.all(ref_proj == proj_idx)


def test_csr_log_transfer_matrix_fast():
    """Verify that the transfer matrix is built correctly.

    In this test, we check that the fast algorithm and the conventional
    give the same results.
    """
    temperature = 1
    num_neighbors = 3
    mag_field = 0
    proj_table = make_spin_proj_table(num_neighbors)
    interactions = 1 / np.arange(1, num_neighbors + 1) ** 2
    matrix_normal = norm_sparse_log_transfer_matrix(
        temperature, mag_field, interactions, proj_table
    )
    matrix_fast = norm_sparse_log_transfer_matrix_fast(
        temperature, mag_field, interactions, num_neighbors
    )
    assert matrix_normal.nnz == matrix_fast.nnz


def test_energy_thermo_limit_fast():
    """Verify that the energy in the thermodynamic limit is correct."""
    temperature = 1
    num_neighbors = 10
    mag_field = 0
    proj_table = make_spin_proj_table(num_neighbors)
    interactions = 1 / np.arange(1, num_neighbors + 1) ** 2
    energy_normal = energy_thermo_limit(
        temperature, mag_field, interactions, proj_table
    )
    energy_fast = energy_thermo_limit_fast(
        temperature, mag_field, interactions, num_neighbors
    )
    assert np.allclose(energy_fast, energy_normal)
