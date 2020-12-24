import typing as t
from dataclasses import dataclass
from math import log

import numpy as np
from scipy.linalg import eigvals

from .utils import bin_digits


def make_spin_proj_table(num_neighbors: int):
    """Creates the table of spin projections."""
    table = np.empty((2 ** num_neighbors, num_neighbors))
    for idx in range(2 ** num_neighbors):
        projections = [-2 * int(v) + 1 for v in bin_digits(idx, num_neighbors)]
        table[idx, :] = projections
    return table


@dataclass
class EnergyData:
    """"""
    helm_free_erg: float
    helm_free_erg_tl: float


def get_energy_data(temp: float,
                    mag_field: float,
                    num_spins: int,
                    hop_params_list: t.Sequence,
                    spin_proj_table: np.ndarray):
    """Calculate the Helmholtz free energy of the system."""
    num_rows, num_neighbors = spin_proj_table.shape
    w_log_matrix = np.zeros((num_rows, num_rows), dtype="f8")
    w_log_matrix[0, 0] = 0.

    # Start loop.
    for idx in range(num_rows):
        for jdx in range(num_rows):
            is_nonzero = True
            for sigma in range(1, num_neighbors):
                proj_one = spin_proj_table[idx, sigma]
                proj_two = spin_proj_table[jdx, sigma - 1]
                is_nonzero = is_nonzero and (proj_one == proj_two)

            # Cycle to the next index.
            if not is_nonzero:
                w_log_matrix[idx, jdx] = -np.inf
                continue

            proj_one = spin_proj_table[idx, 0]
            w_elem = (mag_field * proj_one / temp)
            for edx in range(num_neighbors):
                hop_param = hop_params_list[edx]
                proj_two = spin_proj_table[jdx, edx]
                w_elem += (hop_param * proj_one * proj_two / temp)

            # Update matrix element.
            w_log_matrix[idx, jdx] = w_elem

    # Normalize matrix elements.
    max_w_log_elem = np.max(w_log_matrix)
    w_log_matrix -= max_w_log_elem
    w_norm_eigvals = eigvals(np.exp(w_log_matrix)).real
    max_eigvals = np.max(w_norm_eigvals)
    z_aux = np.sum(w_norm_eigvals ** num_spins)
    helm_free_erg = -temp * (log(z_aux) / num_spins + max_w_log_elem)
    helm_free_erg_tl = -temp * (log(max_eigvals) + max_w_log_elem)
    return EnergyData(helm_free_erg, helm_free_erg_tl)
