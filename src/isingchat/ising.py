import typing as t
from dataclasses import dataclass
from functools import partial
from math import log

import numpy as np
from dask import bag
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

from .exec_ import ParamsGrid
from .utils import bin_digits


def make_spin_proj_table(num_neighbors: int):
    """Creates the table of spin projections."""
    table = np.empty((2 ** num_neighbors, num_neighbors), dtype="i8")
    for idx in range(2 ** num_neighbors):
        projections = [-2 * v + 1 for v in bin_digits(idx, num_neighbors)]
        table[idx, :] = projections
    return table


# See https://realpython.com/python-bitwise-operators/#bitmasks.
@njit
def set_bit(value: int, bit_index: int):
    """Set the bit in ``value`` at the position ``bit_index``."""
    return value | (1 << bit_index)


@njit
def clear_bit(value: int, bit_index: int):
    """Clear the bit in ``value`` at the position ``bit_index``."""
    return value & ~(1 << bit_index)


@njit
def get_bit(value: int, bit_index: int):
    """Get the bit in ``value`` at the position ``bit_index``."""
    return (value & (1 << bit_index)) >> bit_index


@njit
def get_bit_list(value: int, num_bits: int):
    """Return a list with the binary digits of a number.

    The returned list has ``num_bits`` elements. All of the list's elements
    whose position is larger than the most significant bit position
    are filled with zeros. The list is returned in reverse order.
    """
    bit_list = []
    for idx in range(num_bits):
        bit_at_idx = (value >> idx) & 1
        bit_list.append(bit_at_idx)
    return bit_list[::-1]


@njit
def spin_projections(number: int, num_neighbors: int):
    """Find the spin projections associated with a given integer."""
    bit_list = get_bit_list(number, num_neighbors)
    return np.array([-2 * bit_value + 1 for bit_value in bit_list])


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
    """"""

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
            w_elem = mag_field * proj_one / temp
            for edx in range(num_neighbors):
                hop_param = interactions[edx]
                proj_two = spin_proj_table[jdx, edx]
                w_elem += hop_param * proj_one * proj_two / temp

            # Store the matrix element.
            _nnz_elems.append(w_elem)
            _nnz_rows.append(idx)
            _nnz_cols.append(jdx)

    nnz_elems = np.asarray(_nnz_elems, dtype=np.float64)
    nnz_rows = np.asarray(_nnz_rows, dtype=np.int32)
    nnz_cols = np.asarray(_nnz_cols, dtype=np.int32)
    return nnz_elems, nnz_rows, nnz_cols


@njit(cache=True)
def _csr_log_transfer_matrix_parts_fast(
    temp: float, mag_field: float, interactions: np.ndarray, num_neighbors: int
):
    """Calculate the parts of the sparse transfer matrix.

    We use numba to accelerate the calculations.
    """
    # Use lists, since we do not know a priori how many nonzero elements
    # the transfer matrix has.
    _nnz_elems = []
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

        _nnz_elems.append(w_elem)
        _nnz_rows.append(ref_index)
        _nnz_cols.append(compat_index)

    nnz_elems = np.asarray(_nnz_elems, dtype=np.float64)
    nnz_rows = np.asarray(_nnz_rows, dtype=np.int32)
    nnz_cols = np.asarray(_nnz_cols, dtype=np.int32)
    return nnz_elems, nnz_rows, nnz_cols


def norm_sparse_log_transfer_matrix(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    spin_proj_table: np.ndarray,
):
    """Calculate the (sparse) normalized transfer matrix."""
    num_rows, _ = spin_proj_table.shape
    nnz_elems, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts(
        temp, mag_field, interactions, spin_proj_table
    )

    # Normalize matrix elements.
    max_w_log_elem = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem
    w_shape = (num_rows, num_rows)
    return csr_matrix((nnz_elems, (nnz_rows, nnz_cols)), shape=w_shape)


def norm_sparse_log_transfer_matrix_fast(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    num_neighbors: int,
):
    """Calculate the (sparse) normalized transfer matrix."""
    nnz_elems, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions, num_neighbors
    )

    # Normalize matrix elements.
    max_w_log_elem = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    return csr_matrix((nnz_elems, (nnz_rows, nnz_cols)), shape=w_shape)


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
    nnz_elems, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts(
        temp, mag_field, interactions, spin_proj_table
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem
    norm_nnz_elems = np.exp(nnz_elems)
    # Construct the sparse matrix.
    w_shape = (num_rows, num_rows)
    w_matrix = csr_matrix(
        (norm_nnz_elems, (nnz_rows, nnz_cols)), shape=w_shape
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
    nnz_elems, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem
    norm_nnz_elems = np.exp(nnz_elems)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix = csr_matrix(
        (norm_nnz_elems, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Evaluate the largest eigenvalue, since it defines the free energy in
    # the thermodynamic limit.
    # noinspection PyTypeChecker
    w_norm_eigvals, _ = sparse_eigs(w_matrix, k=1, which="LM")
    max_eigvals = w_norm_eigvals.real[0]
    helm_free_erg_tl = -temp * (log(max_eigvals) + max_w_log_elem)
    return helm_free_erg_tl


def grid_func_base(params: t.Tuple[float, float], interactions: np.ndarray):
    """"""
    temperature, magnetic_field = params
    num_neighbors = len(interactions)
    return energy_thermo_limit_fast(
        temperature,
        magnetic_field,
        interactions=interactions,
        num_neighbors=num_neighbors,
    )


def eval_energy(
    params_grid: ParamsGrid, interactions: np.ndarray, num_workers: int = None
):
    """"""
    grid_func = partial(grid_func_base, interactions=interactions)
    # Evaluate the grid using a multidimensional iterator. This
    # way we do not allocate memory for all the combinations of
    # parameter values that form the grid.
    params_bag = bag.from_sequence(params_grid)
    compute_kwargs = {}
    if num_workers is not None:
        compute_kwargs["num_workers"] = num_workers
    chi_square_data = params_bag.map(grid_func).compute(**compute_kwargs)
    return chi_square_data
