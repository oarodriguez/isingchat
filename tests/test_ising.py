from spinchains_manybody.ising import get_energy_data, make_spin_proj_table


def test_energy_data():
    """"""
    temp = 0.6
    mag_field = 1.5
    num_spins = 5
    int_power = 2
    hop_params_list = [1, 0.5 ** int_power]
    num_neighbors = len(hop_params_list)
    spin_proj_table = make_spin_proj_table(num_neighbors)
    data = get_energy_data(temp, mag_field, num_spins,
                           hop_params_list=hop_params_list,
                           spin_proj_table=spin_proj_table)
    print(data)
