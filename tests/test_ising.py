import numpy as np
from spinchains_manybody.ising import (
    energy_thermo_limit_dense, make_spin_proj_table
)


def test_energy_data():
    """"""
    temp = 0.6
    mag_field = 1.5
    int_power = 2
    hop_params_list = np.array([1, 0.5]) ** int_power
    num_neighbors = len(hop_params_list)
    spin_proj_table = make_spin_proj_table(num_neighbors)
    data = energy_thermo_limit_dense(temp, mag_field,
                                     hop_params_list=hop_params_list,
                                     spin_proj_table=spin_proj_table)
    print(data)
