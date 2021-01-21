import typing as t
from dataclasses import dataclass
from functools import partial
from math import log

import numpy as np
from dask import bag
from isingchat.exec_ import ParamsGrid
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

from .utils import bin_digits


def make_spin_proj_table(num_neighbors: int):
    """Creates the table of spin projections."""
    table = np.empty((2 ** num_neighbors, num_neighbors))
    for idx in range(2 ** num_neighbors):
        projections = [-2 * v + 1 for v in bin_digits(idx, num_neighbors)]
        table[idx, :] = projections
    return table


@dataclass
class EnergyData:
    """"""
    helm_free_erg: float
    helm_free_erg_tl: float


@njit(cache=True)
def dense_log_transfer_matrix(temp: float,
                              mag_field: float,
                              hop_params_list: np.ndarray,
                              spin_proj_table: np.ndarray):
    """Calculate the (dense) transfer matrix of the system.

    We use numba to accelerate the calculations.
    """
    num_rows, num_neighbors = spin_proj_table.shape
    w_log_matrix = np.zeros((num_rows, num_rows), dtype=np.float64)

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

    return w_log_matrix


@njit(cache=True)
def _csr_log_transfer_matrix_parts(temp: float,
                                   mag_field: float,
                                   hop_params_list: np.ndarray,
                                   spin_proj_table: np.ndarray):
    """Calculate the parts of the sparse transfer matrix.

    We use numba to accelerate the calculations.
    """
    num_rows, num_neighbors = spin_proj_table.shape
    # Use lists, since we do not know a priori how many nonzero elements
    # the transfer matrix has.
    _nnz_elems = []
    _nnz_rows = []
    _nnz_cols = []

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
                continue

            proj_one = spin_proj_table[idx, 0]
            w_elem = (mag_field * proj_one / temp)
            for edx in range(num_neighbors):
                hop_param = hop_params_list[edx]
                proj_two = spin_proj_table[jdx, edx]
                w_elem += (hop_param * proj_one * proj_two / temp)

            # Store the matrix element.
            _nnz_elems.append(w_elem)
            _nnz_rows.append(idx)
            _nnz_cols.append(jdx)

    nnz_elems = np.asarray(_nnz_elems, dtype=np.float64)
    nnz_rows = np.asarray(_nnz_rows, dtype=np.int32)
    nnz_cols = np.asarray(_nnz_cols, dtype=np.int32)
    return nnz_elems, nnz_rows, nnz_cols


def norm_sparse_log_transfer_matrix(temp: float,
                                    mag_field: float,
                                    hop_params_list: np.ndarray,
                                    spin_proj_table: np.ndarray):
    """Calculate the (sparse) normalized transfer matrix."""
    num_rows, _ = spin_proj_table.shape
    nnz_elems, nnz_rows, nnz_cols = \
        _csr_log_transfer_matrix_parts(temp,
                                       mag_field,
                                       hop_params_list,
                                       spin_proj_table)

    # Normalize matrix elements.
    max_w_log_elem = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem
    w_shape = (num_rows, num_rows)
    return csr_matrix((nnz_elems, (nnz_rows, nnz_cols)),
                      shape=w_shape)


def energy_thermo_limit_dense(temp: float,
                              mag_field: float,
                              hop_params_list: np.ndarray,
                              spin_proj_table: np.ndarray):
    """Calculate the Helmholtz free energy of the system."""
    w_log_matrix = dense_log_transfer_matrix(temp,
                                             mag_field,
                                             hop_params_list,
                                             spin_proj_table)

    # Normalize matrix elements.
    max_w_log_elem = np.max(w_log_matrix)
    w_log_matrix -= max_w_log_elem
    w_norm_eigvals, _ = sparse_eigs(np.exp(w_log_matrix), k=1, which="LM")
    max_eigvals = np.max(w_norm_eigvals.real)
    helm_free_erg_tl = -temp * (log(max_eigvals) + max_w_log_elem)
    return helm_free_erg_tl


def energy_thermo_limit(temp: float,
                        mag_field: float,
                        hop_params_list: np.ndarray,
                        spin_proj_table: np.ndarray):
    """Calculate the Helmholtz free energy of the system."""
    num_rows, _ = spin_proj_table.shape
    nnz_elems, nnz_rows, nnz_cols = \
        _csr_log_transfer_matrix_parts(temp,
                                       mag_field,
                                       hop_params_list,
                                       spin_proj_table)

    # Normalize nonzero matrix elements.
    max_w_log_elem = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem
    norm_nnz_elems = np.exp(nnz_elems)
    # Construct the sparse matrix.
    w_shape = (num_rows, num_rows)
    w_matrix = csr_matrix((norm_nnz_elems, (nnz_rows, nnz_cols)),
                          shape=w_shape)
    # Evaluate the largest eigenvalue, since it defines the free energy in
    # the thermodynamic limit.
    # noinspection PyTypeChecker
    w_norm_eigvals, _ = sparse_eigs(w_matrix, k=1, which="LM")
    max_eigvals = w_norm_eigvals.real[0]
    helm_free_erg_tl = -temp * (log(max_eigvals) + max_w_log_elem)
    return helm_free_erg_tl


def grid_func_base(params: t.Tuple[float, float],
                   hop_params: np.ndarray):
    """"""
    temperature, magnetic_field = params
    num_neighbors = len(hop_params)
    spin_proj_table = make_spin_proj_table(num_neighbors)
    return energy_thermo_limit(temperature,
                               magnetic_field,
                               hop_params_list=hop_params,
                               spin_proj_table=spin_proj_table)


def eval_energy(params_grid: ParamsGrid,
                hop_params: np.ndarray):
    """"""
    grid_func = partial(grid_func_base,
                        hop_params=hop_params)
    # Evaluate the grid using a multidimensional iterator. This
    # way we do not allocate memory for all the combinations of
    # parameter values that form the grid.
    params_bag = bag.from_sequence(params_grid)
    chi_square_data = params_bag.map(grid_func).compute()
    return chi_square_data
