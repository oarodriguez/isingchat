"""Collection of routines used to study the Ising chain."""
import math
import typing as t
from dataclasses import dataclass
from functools import partial

from math import log
from pprint import pprint

import numpy as np
import scipy
from dask import bag
from numba import njit
from scipy.linalg import eigvals
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as sparse_eigs
from numpy.linalg import matrix_power

from .exec_ import ParamsGrid
from .utils import (
    bin_digits,
    clear_bit,
    convert_bin_to_decimal,
    spin_projections,
    get_bit_list
)


def make_spin_proj_table(num_neighbors: int):
    """Create the table of spin projections."""
    table = np.empty((2 ** num_neighbors, num_neighbors), dtype="i8")
    for idx in range(2 ** num_neighbors):
        projections = [-2 * v + 1 for v in bin_digits(idx, num_neighbors)]
        table[idx, :] = projections
    return table


# See https://realpython.com/python-bitwise-operators/#bitmasks.


@njit
def compatible_projections(num_neighbors: int):
    """Find the compatible projections that form the transfer matrix."""
    num_projections = 2 ** num_neighbors
    for ref_index in range(num_projections):
        # The first compatible projection index is obtained by clearing the
        # leftmost bit from the reference index, and shifting the result one
        # position to the left.
        proj_a_index = clear_bit(ref_index, num_neighbors - 1) << 1
        # Yield the first pair.
        yield ref_index, proj_a_index

        # The second compatible projection index is obtained by summing one
        # to the first compatible projection.
        proj_b_index = proj_a_index + 1
        # Yield the second pair.
        yield ref_index, proj_b_index


@dataclass
class EnergyData:
    """Collect the spin chain energies."""

    helm_free_erg: float
    helm_free_erg_tl: float


@njit(cache=True)
def dense_log_transfer_matrix(
    temp: float,
    mag_field: float,
    hop_params_list: np.ndarray,
    spin_proj_table: np.ndarray,
):
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
            w_elem = mag_field * proj_one / temp
            for edx in range(num_neighbors):
                hop_param = hop_params_list[edx]
                proj_two = spin_proj_table[jdx, edx]
                w_elem += hop_param * proj_one * proj_two / temp

            # Update matrix element.
            w_log_matrix[idx, jdx] = w_elem

    return w_log_matrix


@njit(cache=True)
def _csr_log_transfer_matrix_parts(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    spin_proj_table: np.ndarray,
):
    """Calculate the parts of the sparse transfer matrix.

    We use numba to accelerate the calculations.
    """
    num_rows, num_neighbors = spin_proj_table.shape
    # Use lists, since we do not know a priori how many nonzero elements
    # the transfer matrix has.
    _nnz_elements = []
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
            w_elem = mag_field * proj_one / temp
            for edx in range(num_neighbors):
                hop_param = interactions[edx]
                proj_two = spin_proj_table[jdx, edx]
                w_elem += hop_param * proj_one * proj_two / temp

            # Store the matrix element.
            _nnz_elements.append(w_elem)
            _nnz_rows.append(idx)
            _nnz_cols.append(jdx)

    nnz_elements = np.asarray(_nnz_elements, dtype=np.float64)
    nnz_rows = np.asarray(_nnz_rows, dtype=np.int32)
    nnz_cols = np.asarray(_nnz_cols, dtype=np.int32)
    return nnz_elements, nnz_rows, nnz_cols


@njit(cache=True)
def _csr_log_transfer_matrix_parts_fast(
    temp: float, mag_field: float, interactions: np.ndarray, num_neighbors: int
):
    """Calculate the parts of the sparse transfer matrix.

    We use numba to accelerate the calculations.
    """
    # Use lists, since we do not know a priori how many nonzero elements
    # the transfer matrix has.
    _nnz_elements = []
    _nnz_rows = []
    _nnz_cols = []

    # NOTE: Watch out with memory leaks with generators that yield
    #  numpy arrays or more complex data types.
    for proj_sets_pair in compatible_projections(num_neighbors):
        ref_index, compat_index = proj_sets_pair
        ref_proj = spin_projections(ref_index, num_neighbors)
        compat_proj = spin_projections(compat_index, num_neighbors)

        proj_one = ref_proj[0]
        w_elem = mag_field * proj_one / temp
        for edx in range(num_neighbors):
            hop_param = interactions[edx]
            proj_two = compat_proj[edx]
            w_elem += hop_param * proj_one * proj_two / temp

        _nnz_elements.append(w_elem)
        _nnz_rows.append(ref_index)
        _nnz_cols.append(compat_index)

    nnz_elements = np.asarray(_nnz_elements, dtype=np.float64)
    nnz_rows = np.asarray(_nnz_rows, dtype=np.int32)
    nnz_cols = np.asarray(_nnz_cols, dtype=np.int32)
    return nnz_elements, nnz_rows, nnz_cols


@njit(cache=True)
def _centrosym_log_transfer_matrix_parts_fast(
    temp: float, interactions: np.ndarray, num_neighbors: int
):
    """Calculate the parts of the A+JC sparse transfer matrix using the
    centrosymmetric property. Note this method is used just for h=0

    We use numba to accelerate the calculations.
    """
    matrix_size = 2 ** (
        num_neighbors - 1)  # for centrosymmetric matrix just need a half
    _nnz_elements = []
    # for centrosymmetric matrix we need to calculate just an a half or matrix
    _nnz_rows = []
    _nnz_cols = []
    for index in range(int(matrix_size / 2)):
        # from top to bottom
        # top_bin = bin_digits(index, num_neighbors-1)
        top_bin = get_bit_list(index, num_neighbors - 1)
        top_w_elem = 0
        for ind_2, bin_dig in enumerate(top_bin):
            if bin_dig == 0:
                top_w_elem += interactions[ind_2]
            else:
                top_w_elem += -interactions[ind_2]
        # first
        _nnz_elements.append((top_w_elem + interactions[-1]) / temp)
        _nnz_rows.append(index)
        _nnz_cols.append(2 * index)
        # second
        _nnz_elements.append((top_w_elem - interactions[-1]) / temp)
        _nnz_rows.append(index)
        _nnz_cols.append(2 * index + 1)
        # from bottom to top
        # bottom_bin = bin_digits(matrix_size-1-index, num_neighbors - 1)
        bottom_bin = get_bit_list(matrix_size - 1 - index, num_neighbors - 1)
        bottom_w_elem = 0
        for ind_2, bin_dig in enumerate(bottom_bin):
            if bin_dig == 0:
                bottom_w_elem += interactions[ind_2]
            else:
                bottom_w_elem += -interactions[ind_2]
        _nnz_elements.append((bottom_w_elem - interactions[-1]) / temp)
        _nnz_rows.append(matrix_size - 1 - index)
        _nnz_cols.append(2 * index)
        # second column
        _nnz_elements.append((bottom_w_elem + interactions[-1]) / temp)
        _nnz_rows.append(matrix_size - 1 - index)
        _nnz_cols.append(2 * index + 1)

    nnz_elements = np.asarray(_nnz_elements, dtype=np.float64)
    nnz_rows = np.asarray(_nnz_rows, dtype=np.int32)
    nnz_cols = np.asarray(_nnz_cols, dtype=np.int32)
    return nnz_elements, nnz_rows, nnz_cols


def _csr_finite_log_transfer_matrix_parts_fast(
    temp: float, mag_field: float, interactions: np.ndarray, num_neighbors: int
):
    """Calculate the parts of the sparse transfer matrix.

    We use numba to accelerate the calculations.
    """
    _nnz_elements = []
    _nnz_rows = list(range(2 ** num_neighbors))
    _nnz_cols = []
    for row in _nnz_rows:
        aux_bin = bin_digits(row, num_neighbors)
        first_element = aux_bin.pop(0)
        aux_bin.append(first_element)
        col = convert_bin_to_decimal(aux_bin)
        _nnz_cols.append(col)

        ref_proj = spin_projections(row, num_neighbors)
        proj_one = ref_proj[0]
        w_elem = mag_field * proj_one / temp
        for index in range(len(ref_proj)):
            hop_param = interactions[index]
            if index < len(ref_proj) - 1:
                proj_two = ref_proj[index + 1]
            else:
                proj_two = ref_proj[0]
            w_elem += hop_param * proj_one * proj_two / temp
        _nnz_elements.append(w_elem)

    nnz_elements = np.asarray(_nnz_elements, dtype=np.float64)
    nnz_rows = np.asarray(_nnz_rows, dtype=np.int32)
    nnz_cols = np.asarray(_nnz_cols, dtype=np.int32)
    return nnz_elements, nnz_rows, nnz_cols


def norm_sparse_log_transfer_matrix(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    spin_proj_table: np.ndarray,
):
    """Calculate the (sparse) normalized transfer matrix."""
    num_rows, _ = spin_proj_table.shape
    nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts(
        temp, mag_field, interactions, spin_proj_table
    )

    # Normalize matrix elements.
    max_w_log_elem = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem
    w_shape = (num_rows, num_rows)
    return csr_matrix((nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape)


def norm_sparse_log_transfer_matrix_fast(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    num_neighbors: int,
):
    """Calculate the (sparse) normalized transfer matrix."""
    nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions, num_neighbors
    )

    # Normalize matrix elements.
    max_w_log_elem = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    return csr_matrix((nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape)


def energy_finite_chain_fast(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    num_neighbors: int,
    num_tm_eigvals: int = None,
):
    """Calculate the Helmholtz free energy for a finite chain."""
    nnz_elements, nnz_rows, nnz_cols = _csr_finite_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Strictly, we should calculate all the eigvals and calculate the
    # Free energy according to F. A, Kassan-ogly (2001),
    #   https://www.tandfonline.com/doi/abs/10.1080/0141159010822758.
    w_norm_eigvals: np.ndarray = eigvals(w_matrix.todense())
    eigvals_norms: np.ndarray = np.abs(w_norm_eigvals)
    max_eigenvalue_norm_idx = eigvals_norms.argmax()
    max_eigenvalue_norm = eigvals_norms[max_eigenvalue_norm_idx]
    reduced_eigvals = w_norm_eigvals / max_eigenvalue_norm
    reduced_eigvals_contrib = np.sum(reduced_eigvals ** (num_neighbors))
    # print(
    #     "reduced_eigvals: {}".format(
    #         log(reduced_eigvals_contrib.real) / num_neighbors
    #     )
    # )
    # print("\n")
    helm_free_erg = -temp * (
        max_w_log_elem
        + log(max_eigenvalue_norm)
        + log(reduced_eigvals_contrib.real) / num_neighbors
    )
    return helm_free_erg


def energy_imperfect_finite_chain_fast(
    temp: float,
    mag_field: float,
    interactions_1: np.ndarray,
    interactions_2: np.ndarray,
    num_neighbors: int,
    num_tm_eigvals: int = None,
):
    """Calculate Helmholtz free energy for a chain with two transfer matrix.
    This two transfer matrix represent one unit-cell of two spins.
    Args:
        :param temp (float): Temperature of the system (T)
        :param mag_field (float): External magnetic field (h)
        :param interactions_1 (np.ndarray): List of interaction for the first
        spin in the unit-cell
        :param interactions_2 (np.ndarray): List of interaction for the second
        spin in the unit-cell
        :param num_neighbors (int): Number of nearest neighboirs of interaction (nv)
        :param num_tm_eigvals (int): Number of eigenvals for transfer matrix
        to use for calculate the Free Helmholtz energy

    Return:
        :return helm_free_erg (float): free helmholtz energy of the chain
    """
    # First matrix
    nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions_1, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem_1 = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem_1
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix_1 = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Second matrix
    nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions_2, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem_2 = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem_2
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix_2 = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    w_matrix = w_matrix_1 * w_matrix_2
    # Strictly, we should calculate all the eigvals and calculate the
    # Free energy according to F. A, Kassan-ogly (2001),
    #   https://www.tandfonline.com/doi/abs/10.1080/0141159010822758.
    # However, in practice, the contribution of the second largest and
    # subsequent eigvals to the partition function decreases fast, so it
    # is sufficient to calculate only a few of the largest eigvals.
    if num_tm_eigvals is None:
        num_eigvals = min(num_neighbors ** 2, num_rows - 2)
    else:
        num_eigvals = min(num_tm_eigvals, num_rows - 2)
    # For three or two interactions we take all eigvals
    if len(interactions_1) <= 3:
        w_matrix_dense = w_matrix.todense()
        w_all_norm_eigvals: np.ndarray = scipy.linalg.eig(w_matrix_dense)
        w_norm_eigvals = w_all_norm_eigvals[0]
    else:
        w_norm_eigvals: np.ndarray = sparse_eigs(
            w_matrix, k=num_eigvals, which="LM", return_eigenvectors=False
        )
    # sparse_eigs()
    eigvals_norms: np.ndarray = np.abs(w_norm_eigvals)
    max_eigenvalue_norm_idx = eigvals_norms.argmax()
    max_eigenvalue_norm = eigvals_norms[max_eigenvalue_norm_idx]
    reduced_eigvals = w_norm_eigvals / max_eigenvalue_norm
    reduced_eigvals_contrib = np.sum(reduced_eigvals ** (num_neighbors))
    cell_unit = 2
    helm_free_erg = -(temp / cell_unit) * (
        max_w_log_elem_1
        + max_w_log_elem_2
        + log(max_eigenvalue_norm)
        + log(reduced_eigvals_contrib.real) / num_neighbors
    )
    return helm_free_erg


def energy_thermo_limit_dense(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    spin_proj_table: np.ndarray,
):
    """Calculate the Helmholtz free energy of the system."""
    w_log_matrix = dense_log_transfer_matrix(
        temp, mag_field, interactions, spin_proj_table
    )

    # Normalize matrix elements.
    max_w_log_elem = np.max(w_log_matrix)
    w_log_matrix -= max_w_log_elem
    w_norm_eigvals, _ = sparse_eigs(np.exp(w_log_matrix), k=1, which="LM")
    max_eigvals = np.max(w_norm_eigvals.real)
    helm_free_erg_tl = -temp * (log(max_eigvals) + max_w_log_elem)
    return helm_free_erg_tl


def energy_thermo_limit(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    spin_proj_table: np.ndarray,
):
    """Calculate the Helmholtz free energy of the system."""
    num_rows, _ = spin_proj_table.shape
    nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts(
        temp, mag_field, interactions, spin_proj_table
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    w_shape = (num_rows, num_rows)
    w_matrix = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Evaluate the largest eigenvalue, since it defines the free energy in
    # the thermodynamic limit.
    # noinspection PyTypeChecker
    w_norm_eigvals, _ = sparse_eigs(w_matrix, k=1, which="LM")
    max_eigvals = w_norm_eigvals.real[0]
    helm_free_erg_tl = -temp * (log(max_eigvals) + max_w_log_elem)
    return helm_free_erg_tl


def energy_thermo_limit_fast(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    num_neighbors: int,
):
    """Calculate the Helmholtz free energy of the system."""
    nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Evaluate the largest eigenvalue only.
    num_eigvals = 1
    w_norm_eigvals: np.ndarray
    # noinspection PyTypeChecker
    w_norm_eigvals = sparse_eigs(
        w_matrix, k=num_eigvals, which="LM", return_eigenvectors=False
    )
    # print('max_w_log_elem: {}'.format(max_w_log_elem))
    # print('temp: {}'.format(temp))
    # print('eigvals: ')
    # print(w_norm_eigvals)
    # print('matrix: ')
    # print(w_matrix)
    # max_eigvals = w_norm_eigvals.real[0]
    eigvals_norms: np.ndarray = np.abs(w_norm_eigvals)
    max_eigenvalue_norm_idx = eigvals_norms.argmax()
    max_eigenvalue_norm = eigvals_norms[max_eigenvalue_norm_idx]
    # reduced_eigvals = w_norm_eigvals / max_eigenvalue_norm
    # In the thermodynamic limit, the number of spins is infinity.
    # Accordingly, only the largest reduced eigenvalue contributes.
    reduced_eigvals_contrib = 1.0
    helm_free_erg_tl = -temp * (
        max_w_log_elem
        + log(max_eigenvalue_norm)
        + log(reduced_eigvals_contrib.real)
    )
    return helm_free_erg_tl


def energy_thermo_limit_fast_centro(
    temp: float,
    interactions: np.ndarray,
    num_neighbors: int,
):
    """Calculate the Helmholtz free energy of the system using the centrosymmetric
    property. Note that this is true just for h=0"""
    nnz_elements, nnz_rows, nnz_cols = _centrosym_log_transfer_matrix_parts_fast(
        temp, interactions, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    num_rows = 2 ** (num_neighbors - 1)
    w_shape = (num_rows, num_rows)
    w_matrix = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Evaluate the largest eigenvalue only.
    num_eigvals = 1
    w_norm_eigvals: np.ndarray
    # noinspection PyTypeChecker
    w_norm_eigvals = sparse_eigs(
        w_matrix, k=num_eigvals, which="LM", return_eigenvectors=False
    )
    # print('max_w_log_elem: {}'.format(max_w_log_elem))
    # print('temp: {}'.format(temp))
    # print('eigvals: ')
    # print(w_norm_eigvals)
    # print('matrix: ')
    # print(w_matrix)
    eigvals_norms: np.ndarray = np.abs(w_norm_eigvals)
    max_eigenvalue_norm_idx = eigvals_norms.argmax()
    max_eigenvalue_norm = eigvals_norms[max_eigenvalue_norm_idx]
    # reduced_eigvals = w_norm_eigvals / max_eigenvalue_norm
    # In the thermodynamic limit, the number of spins is infinity.
    # Accordingly, only the largest reduced eigenvalue contributes.
    reduced_eigvals_contrib = 1.0
    helm_free_erg_tl = -temp * (
        max_w_log_elem
        + log(max_eigenvalue_norm)
        + log(reduced_eigvals_contrib.real)
    )
    return helm_free_erg_tl


def energy_imperfect_thermo_limit_fast(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    interactions_2: np.ndarray,
    num_neighbors: int,
):
    """Calculate the Helmholtz free energy of the system."""
    nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem_1 = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem_1
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix_1 = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Second matrix
    nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions_2, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem_2 = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem_2
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix_2 = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    w_matrix = w_matrix_1 * w_matrix_2
    # Evaluate the largest eigenvalue only.
    num_eigvals = 1
    w_norm_eigvals: np.ndarray
    # noinspection PyTypeChecker
    w_norm_eigvals = sparse_eigs(
        w_matrix, k=num_eigvals, which="LM", return_eigenvectors=False
    )
    max_eigenvalue = w_norm_eigvals.real[0]
    # In the thermodynamic limit, the number of spins is infinity.
    # Accordingly, only the largest reduced eigenvalue contributes.
    cell_unit = 2
    helm_free_erg_tl = -(temp / cell_unit) * (
        max_w_log_elem_1 + max_w_log_elem_2 + log(max_eigenvalue)
    )
    return helm_free_erg_tl


def free_energy_fast(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    num_neighbors: int,
):
    """Calculate the Helmholtz free energy of the system."""
    nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem = np.max(nnz_elements)
    nnz_elements -= max_w_log_elem
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Evaluate the largest eigenvalue, since it defines the free energy in
    # the thermodynamic limit.
    # noinspection PyTypeChecker
    w_norm_eigvals, _ = sparse_eigs(w_matrix, k=1, which="LM")
    max_eigvals = w_norm_eigvals.real[0]
    helm_free_erg_tl = -temp * (log(max_eigvals) + max_w_log_elem)
    return helm_free_erg_tl


def grid_func_base(
    params: t.Tuple[float, float],
    interactions: np.ndarray,
    interactions_2: np.ndarray = None,
    finite_chain: bool = False,
    num_tm_eigvals: int = None,
    is_centrosymmetric: bool = False,
    get_eigenvalues: bool = False
):
    """Calculate the spin chain properties over a parameter grid."""
    temperature, magnetic_field = params
    num_neighbors = len(interactions)
    if is_centrosymmetric:
        return energy_thermo_limit_fast_centro(
            temperature,
            interactions=interactions,
            num_neighbors=num_neighbors,
        )
    if interactions_2 is not None:
        if finite_chain:
            return energy_imperfect_finite_chain_fast(
                temperature,
                magnetic_field,
                interactions_1=interactions,
                interactions_2=interactions_2,
                num_neighbors=num_neighbors,
                num_tm_eigvals=num_tm_eigvals,
            )
        else:
            return energy_imperfect_thermo_limit_fast(
                temperature,
                magnetic_field,
                interactions=interactions,
                interactions_2=interactions_2,
                num_neighbors=num_neighbors,
            )
    if finite_chain:
        return energy_finite_chain_fast(
            temperature,
            magnetic_field,
            interactions=interactions,
            num_neighbors=num_neighbors,
            num_tm_eigvals=num_tm_eigvals,
        )
    return energy_thermo_limit_fast(
        temperature,
        magnetic_field,
        interactions=interactions,
        num_neighbors=num_neighbors,
    )


def grid_func_base_eigenvalues(
    params: t.Tuple[float, float],
    interactions: np.ndarray,
    interactions_2: np.ndarray = None,
    finite_chain: bool = False,
    num_tm_eigvals: int = None,
    is_centrosymmetric: bool = False,
):
    """
    Calculate the transfer matrix eigenvalues of a spin chain
    over a parameter grid.
    """
    temperature, magnetic_field = params
    # if interactions_2 is not None:
    #     # TODO put here the eigenvalues calculations for imperfect chains
    #     if finite_chain:
    #         return eigens_tm_imperfect_finite_chain_fast(
    #             temperature,
    #             magnetic_field,
    #             interactions_1=interactions,
    #             interactions_2=interactions_2,
    #             num_neighbors=num_neighbors,
    #             num_tm_eigvals=num_tm_eigvals,
    #         )
    #     else:
    #         return eigens_tm_imperfect_thermo_limit_fast(
    #             temperature,
    #             magnetic_field,
    #             interactions=interactions,
    #             interactions_2=interactions_2,
    #             num_neighbors=num_neighbors,
    #         )
    # if finite_chain:
    #     # TODO put here the transfer matrix eigenvalues for finite chains
    #     return eigens_tm_finite_chain_fast(
    #         temperature,
    #         magnetic_field,
    #         interactions=interactions,
    #         num_neighbors=num_neighbors,
    #         num_tm_eigvals=num_tm_eigvals,
    #     )
    return eigens_tm_tl(
        temperature,
        magnetic_field,
        interactions=interactions,
        num_tm_eigvals=num_tm_eigvals,
        is_centrosymmetric=is_centrosymmetric
    )


def grid_func_base_cor_length_tl(
    params: t.Tuple[float, float],
    interactions: np.ndarray,
    interactions_2: np.ndarray = None,
    finite_chain: bool = False,
    num_tm_eigvals: int = None,
    is_centrosymmetric: bool = False,
    is_inv_temp: bool = False,
):
    """
    Calculate the transfer matrix eigenvalues of a spin chain
    over a parameter grid.
    """
    temperature, magnetic_field = params
    # if is_centrosymmetric:
    #     # TODO put here the eigenvalues calculations using centrosymmetric
    #     # property
    #     return centrosym_eigens_tm_tl(
    #         temperature,
    #         interactions=interactions,
    #         num_neighbors=num_neighbors,
    #     )
    # if interactions_2 is not None:
    #     # TODO put here the eigenvalues calculations for imperfect chains
    #     if finite_chain:
    #         return eigens_tm_imperfect_finite_chain_fast(
    #             temperature,
    #             magnetic_field,
    #             interactions_1=interactions,
    #             interactions_2=interactions_2,
    #             num_neighbors=num_neighbors,
    #             num_tm_eigvals=num_tm_eigvals,
    #         )
    #     else:
    #         return eigens_tm_imperfect_thermo_limit_fast(
    #             temperature,
    #             magnetic_field,
    #             interactions=interactions,
    #             interactions_2=interactions_2,
    #             num_neighbors=num_neighbors,
    #         )
    # if finite_chain:
    #     # TODO put here the transfer matrix eigenvalues for finite chains
    #     return eigens_tm_finite_chain_fast(
    #         temperature,
    #         magnetic_field,
    #         interactions=interactions,
    #         num_neighbors=num_neighbors,
    #         num_tm_eigvals=num_tm_eigvals,
    #     )
    if is_inv_temp:
        return correlation_length_limit(
            1 / temperature,
            magnetic_field,
            interactions=interactions,
            is_centrosymmetric=is_centrosymmetric
        )

    return correlation_length_limit(
        temperature,
        magnetic_field,
        interactions=interactions,
        is_centrosymmetric=is_centrosymmetric
    )


def grid_func_base_cor_func(
    params: t.Tuple[float, float, float],
    interactions: np.ndarray,
    interactions_2: np.ndarray = None,
    finite_chain: bool = False,
    num_tm_eigvals: int = None,
    is_centrosymmetric: bool = False,
):
    """
    Calculate the transfer matrix eigenvalues of a spin chain
    over a parameter grid.
    """
    spin_spin_dist, temperature, magnetic_field = params
    # if is_centrosymmetric:
    #     # TODO put here the eigenvalues calculations using centrosymmetric
    #     # property
    #     return centrosym_eigens_tm_tl(
    #         temperature,
    #         interactions=interactions,
    #         num_neighbors=num_neighbors,
    #     )
    # if interactions_2 is not None:
    #     # TODO put here the eigenvalues calculations for imperfect chains
    #     if finite_chain:
    #         return eigens_tm_imperfect_finite_chain_fast(
    #             temperature,
    #             magnetic_field,
    #             interactions_1=interactions,
    #             interactions_2=interactions_2,
    #             num_neighbors=num_neighbors,
    #             num_tm_eigvals=num_tm_eigvals,
    #         )
    #     else:
    #         return eigens_tm_imperfect_thermo_limit_fast(
    #             temperature,
    #             magnetic_field,
    #             interactions=interactions,
    #             interactions_2=interactions_2,
    #             num_neighbors=num_neighbors,
    #         )
    # if finite_chain:
    #     # TODO put here the transfer matrix eigenvalues for finite chains
    #     return eigens_tm_finite_chain_fast(
    #         temperature,
    #         magnetic_field,
    #         interactions=interactions,
    #         num_neighbors=num_neighbors,
    #         num_tm_eigvals=num_tm_eigvals,
    #     )
    return correlation_function_tl(
        spin_spin_dist,
        temperature,
        magnetic_field,
        interactions,
        num_tm_eigvals
    )


def eval_energy(
    params_grid: ParamsGrid,
    interactions: np.ndarray,
    interactions_2: np.ndarray = None,
    finite_chain: bool = False,
    num_tm_eigvals: int = None,
    num_workers: int = None,
    is_centrosymmetric: bool = False
):
    """Calculate the energy over a parameter grid."""
    grid_func = partial(
        grid_func_base,
        interactions=interactions,
        interactions_2=interactions_2,
        finite_chain=finite_chain,
        num_tm_eigvals=num_tm_eigvals,
        is_centrosymmetric=is_centrosymmetric
    )
    # Evaluate the grid using a multidimensional iterator. This
    # way we do not allocate memory for all the combinations of
    # parameter values that form the grid.
    params_bag = bag.from_sequence(params_grid)
    compute_kwargs = {}
    if num_workers is not None:
        compute_kwargs["num_workers"] = num_workers
    chi_square_data = params_bag.map(grid_func).compute(**compute_kwargs)
    return chi_square_data


# -----------------------------------------------------------------------------
# Spin-spin correlation function and correlation length calculus
# -----------------------------------------------------------------------------
def z_projection_gen(
    n: int,
    side: str = 'right'
):
    """
    Return Right or left tensor product of Pauli sigma_z matrix with identity n times
    :param n (int) Required.
        Number of times for the tensor product
    :param side (str) Optional. Default 'right'. Could be 'right' or 'left'
        Right or left tensor product extension of Pauli sigma_z matrix
    :return: sigma_z (scipy.sparse.csr.csr_matrix)
    """
    sigma_z: scipy.sparse.csr.csr_matrix
    sigma_z = csr_matrix(
        [[1, 0],
         [0, -1]]
    )
    _identity = scipy.sparse.identity(2)

    if side == 'right':
        for i in range(n):
            sigma_z = scipy.sparse.kron(sigma_z, _identity)
    elif side == 'left':
        for i in range(n):
            sigma_z = scipy.sparse.kron(_identity, sigma_z)
    else:
        print('side {} is not support. Please use "right" or "left" instead')

    return sigma_z


# TODO: make the function to multiply matrix and obtain the correlation function
def correlation_function_finite_chain_matrix_mult(
    r: int,
    temp: float,
    mag_field: float,
    interactions: np.ndarray
):
    num_neighbors = len(interactions)
    num_spins = num_neighbors  # size of the chain is the same of the int range

    nnz_elements, nnz_rows, nnz_cols = _csr_finite_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions, num_neighbors
    )

    # Normalize nonzero matrix elements.
    norm_nnz_elements = np.exp(nnz_elements)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix = csr_matrix(
        (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
    )
    w_norm_eigvals, _ = sparse_eigs(w_matrix, k=1, which="LM")
    z_function = np.sum(w_norm_eigvals)
    pauli_z_gen = z_projection_gen(num_neighbors - 1)
    # Make the matrix multiplication
    result = pauli_z_gen.dot(matrix_power(w_matrix.todense(), r)).dot(
        pauli_z_gen.dot(matrix_power(w_matrix.todense(), num_spins - r)))

    # Return correlation function
    return result.trace().max() / z_function


def eigens_tm_tl(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    num_tm_eigvals: int = None,
    is_centrosymmetric: bool = False
):
    """
        Return max_w_log_elem and eigenvectors of the transfer matrix of the infinite chain
        In case of len(interactions) > 3, the default num eigvals returned is min(num_neighbors ** 2, num_rows - 2)
    """
    num_neighbors = len(interactions)

    if is_centrosymmetric:
        nnz_elements, nnz_rows, nnz_cols = \
            _centrosym_log_transfer_matrix_parts_fast(
                temp, interactions, num_neighbors
            )
        num_rows = 2 ** (num_neighbors - 1)
        # Construct the sparse matrix.
        w_shape = (num_rows, num_rows)

        # Normalize nonzero matrix elements.
        max_w_log_elem = np.max(nnz_elements)
        nnz_elements -= max_w_log_elem
        # For A+JC matrix
        norm_nnz_elements = np.exp(nnz_elements)
        w_matrix_plus = csr_matrix(
            (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
        )
        # For A-JC matrix
        norm_nnx_elements_minus = np.concatenate(
            (norm_nnz_elements[:int(len(norm_nnz_elements) / 2)],
             -norm_nnz_elements[int(len(norm_nnz_elements) / 2):])
        )
        w_matrix_minus = csr_matrix(
            (norm_nnx_elements_minus, (nnz_rows, nnz_cols)), shape=w_shape
        )
        # Check num_tm_eigvals in order to use eigs sparse routine or dense
        # sparse routine
        if num_tm_eigvals is None:
            num_eigvals = int(min(num_neighbors ** 2,
                              num_rows - 2))  # Default num_eigvals
        else:
            num_eigvals = int(num_tm_eigvals / 2)

        if num_eigvals > num_rows - 2:
            w_matrix_dense_plus = w_matrix_plus.todense()
            w_norm_eigvals_plus, w_norm_eigvect_plus = scipy.linalg.eig(
                w_matrix_dense_plus)

            w_matrix_dense_minus = w_matrix_minus.todense()
            w_norm_eigvals_minus, w_norm_eigvect_minus = scipy.linalg.eig(
                w_matrix_dense_minus)

            if num_eigvals > num_rows:
                # return all eigvals
                return \
                    max_w_log_elem, \
                    np.concatenate(
                        (w_norm_eigvals_plus,
                         w_norm_eigvals_minus)
                    ), \
                    np.concatenate(
                        (w_norm_eigvect_plus, w_norm_eigvect_minus), axis=0)
            else:
                return \
                    max_w_log_elem, \
                    np.concatenate(
                        (w_norm_eigvals_plus[:num_eigvals],
                         w_norm_eigvals_minus[:num_eigvals])
                    ),\
                    np.concatenate(
                        (w_norm_eigvect_plus[:, :num_eigvals],
                         w_norm_eigvect_minus[:, :num_eigvals]),axis=0
                    )
        else:
            w_norm_eigvals_plus, w_norm_eigvect_plus = sparse_eigs(
                w_matrix_plus, k=num_eigvals, which="LM"
            )
            w_norm_eigvals_minus, w_norm_eigvect_minus = sparse_eigs(
                w_matrix_minus, k=num_eigvals, which="LM"
            )

            return \
                max_w_log_elem, \
                np.concatenate(
                    (w_norm_eigvals_plus, w_norm_eigvals_minus)
                ), \
                np.concatenate(
                    (w_norm_eigvect_plus, w_norm_eigvect_minus),axis=0)
    else:
        nnz_elements, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
            temp, mag_field, interactions, num_neighbors
        )
        num_rows = 2 ** num_neighbors

        # Normalize nonzero matrix elements.
        max_w_log_elem = np.max(nnz_elements)
        nnz_elements -= max_w_log_elem
        norm_nnz_elements = np.exp(nnz_elements)
        # Construct the sparse matrix.
        w_shape = (num_rows, num_rows)
        w_matrix = csr_matrix(
            (norm_nnz_elements, (nnz_rows, nnz_cols)), shape=w_shape
        )
        # Check num_tm_eigvals in order to use eigs sparse routine or dense
        # sparse routine
        if num_tm_eigvals is None:
            num_eigvals = min(num_neighbors ** 2,
                              num_rows - 2)  # Default num_eigvals
        else:
            num_eigvals = num_tm_eigvals
        if num_eigvals > num_rows - 2:
            w_matrix_dense = w_matrix.todense()
            # w_all_norm_eigvals, w_all_norm_eigvect = scipy.linalg.eig(w_matrix_dense)
            # w_norm_eigvals = w_all_norm_eigvals[0]
            w_norm_eigvals, w_norm_eigvect = scipy.linalg.eig(w_matrix_dense)
            if num_eigvals > num_rows:
                print(
                    'Warning: The number of eigvals can not be greater than '
                    'num of the rows. We calculate all eigvals')

                # return all eigvals
                return max_w_log_elem, w_norm_eigvals, w_norm_eigvect
            else:
                return max_w_log_elem, \
                       w_norm_eigvals[:num_eigvals], \
                       w_norm_eigvect[:, :num_eigvals]
        else:
            w_norm_eigvals, w_norm_eigvect = sparse_eigs(
                w_matrix, k=num_eigvals, which="LM"
            )

            return max_w_log_elem, w_norm_eigvals, w_norm_eigvect


def correlation_function_tl(
    r: int,
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    num_tm_eigvals: int = None
):
    num_neighbors = len(interactions)
    if num_tm_eigvals is None:
        num_tm_eigvals = 2 ** num_neighbors  # all eigvals

    _, w_norm_eigvals, w_norm_eigvects = eigens_tm_tl(temp, mag_field,
                                                      interactions,
                                                      num_tm_eigvals)
    eigvals_norms: np.ndarray = np.abs(w_norm_eigvals)
    max_eigenvalue_norm_idx = eigvals_norms.argmax()
    max_eigenvalue_norm = eigvals_norms[max_eigenvalue_norm_idx]
    max_eigvect_norm = w_norm_eigvects[:, max_eigenvalue_norm_idx]
    reduced_eigvals = (w_norm_eigvals / max_eigenvalue_norm) ** r
    sigma_z = z_projection_gen(num_neighbors - 1)  # matrix extension
    corr_function = 0
    for ind_eig in range(len(reduced_eigvals)):
        if ind_eig == max_eigenvalue_norm_idx:
            pass
        else:
            corr_function += (reduced_eigvals[ind_eig]) * (
                np.abs(max_eigvect_norm.dot(
                    sigma_z.dot(w_norm_eigvects[:, ind_eig]))) ** 2
            )

    return corr_function


def correlation_length_limit(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    is_centrosymmetric: bool = False
):
    _, w_norm_eigvals, _ = eigens_tm_tl(temp,
                                        mag_field,
                                        interactions,
                                        2,
                                        is_centrosymmetric)
    eigvals_norms: np.ndarray = np.abs(w_norm_eigvals)
    eigvals_norms[::-1].sort()

    return 1 / abs(math.log(eigvals_norms[0] / eigvals_norms[1]))


@njit(cache=True)
def _finite_transfer_matrix_parts_fast(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    num_neighbors: int
):
    """ Calculate the parts of the sparse transfer matrix
    property. Note this method is used just for h=0

    We use numba to accelerate the calculations.
    """
    matrix_size = 2 ** num_neighbors
    _nnz_elements = []
    # we need to calculate just two values per row
    _nnz_rows = []
    _nnz_cols = []
    for index in range(int(matrix_size / 2)):
        # from top to bottom
        # top_bin = bin_digits(index, num_neighbors-1)
        top_bin = get_bit_list(index, num_neighbors)
        top_w_elem = mag_field
        for ind_2, bin_dig in enumerate(top_bin):
            if bin_dig == 0:
                top_w_elem += interactions[ind_2]
            else:
                top_w_elem += -interactions[ind_2]
        # first
        _nnz_elements.append((top_w_elem + interactions[-1]) / temp)
        _nnz_rows.append(index)
        _nnz_cols.append(2 * index)
        # second
        _nnz_elements.append((top_w_elem - interactions[-1]) / temp)
        _nnz_rows.append(index)
        _nnz_cols.append(2 * index + 1)
        # from bottom to top
        # bottom_bin = bin_digits(matrix_size-1-index, num_neighbors - 1)
        bottom_bin = get_bit_list(matrix_size - 1 - index, num_neighbors - 1)
        bottom_w_elem = 0
        for ind_2, bin_dig in enumerate(bottom_bin):
            if bin_dig == 0:
                bottom_w_elem += interactions[ind_2]
            else:
                bottom_w_elem += -interactions[ind_2]
        _nnz_elements.append((bottom_w_elem - interactions[-1]) / temp)
        _nnz_rows.append(matrix_size - 1 - index)
        _nnz_cols.append(2 * index)
        # second column
        _nnz_elements.append((bottom_w_elem + interactions[-1]) / temp)
        _nnz_rows.append(matrix_size - 1 - index)
        _nnz_cols.append(2 * index + 1)

    nnz_elements = np.asarray(_nnz_elements, dtype=np.float64)
    nnz_rows = np.asarray(_nnz_rows, dtype=np.int32)
    nnz_cols = np.asarray(_nnz_cols, dtype=np.int32)
    return nnz_elements, nnz_rows, nnz_cols
