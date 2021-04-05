import numpy as np

from isingchat.ising import energy_thermo_limit_dense, make_spin_proj_table


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
