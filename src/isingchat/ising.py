"""Collection of routines used to study the Ising chain."""

import typing as t
from dataclasses import dataclass
from functools import partial
from math import log

import numpy as np
import scipy
from dask import bag
from numba import njit
from scipy.linalg import eigvals
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

from .exec_ import ParamsGrid
from .utils import (
    bin_digits,
    clear_bit,
    convert_bin_to_decimal,
    spin_projections,
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


def _csr_finite_log_transfer_matrix_parts_fast(
    temp: float, mag_field: float, interactions: np.ndarray, num_neighbors: int
):
    """Calculate the parts of the sparse transfer matrix.

    We use numba to accelerate the calculations.
    """
    _nnz_elems = []
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
        _nnz_elems.append(w_elem)

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


def energy_finite_chain_fast(
    temp: float,
    mag_field: float,
    interactions: np.ndarray,
    num_neighbors: int,
    num_tm_eigvals: int = None,
):
    """Calculate the Helmholtz free energy for a finite chain."""
    nnz_elems, nnz_rows, nnz_cols = _csr_finite_log_transfer_matrix_parts_fast(
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
    # Strictly, we should calculate all the eigenvalues and calculate the
    # Free energy according to F. A, Kassan-ogly (2001),
    #   https://www.tandfonline.com/doi/abs/10.1080/0141159010822758.
    w_norm_eigvals: np.ndarray = eigvals(w_matrix.todense())
    eigvals_norms: np.ndarray = np.abs(w_norm_eigvals)
    max_eigval_norm_idx = eigvals_norms.argmax()
    max_eigval_norm = eigvals_norms[max_eigval_norm_idx]
    reduced_eigvals = w_norm_eigvals / max_eigval_norm
    reduced_eigvals_contrib = np.sum(reduced_eigvals ** (num_neighbors))
    # print(
    #     "reduced_eigvals: {}".format(
    #         np.log(reduced_eigvals_contrib.real) / num_neighbors
    #     )
    # )
    # print("\n")
    helm_free_erg = -temp * (
        max_w_log_elem
        + np.log(max_eigval_norm)
        + np.log(reduced_eigvals_contrib.real) / num_neighbors
    )
    return helm_free_erg


# TODO: check that this function reduce to the regular one
def energy_imperfect_finite_chain_fast(
    temp: float,
    mag_field: float,
    interactions_1: np.ndarray,
    interactions_2: np.ndarray,
    num_neighbors: int,
    num_tm_eigvals: int = None,
):
    """Calculate Helmholtz free energy for a chain with two imperfections."""
    # First matrix
    nnz_elems, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions_1, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem_1 = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem_1
    norm_nnz_elems = np.exp(nnz_elems)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix_1 = csr_matrix(
        (norm_nnz_elems, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Second matrix
    nnz_elems, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions_2, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem_2 = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem_2
    norm_nnz_elems = np.exp(nnz_elems)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix_2 = csr_matrix(
        (norm_nnz_elems, (nnz_rows, nnz_cols)), shape=w_shape
    )
    w_matrix = w_matrix_1 * w_matrix_2
    # Strictly, we should calculate all the eigenvalues and calculate the
    # Free energy according to F. A, Kassan-ogly (2001),
    #   https://www.tandfonline.com/doi/abs/10.1080/0141159010822758.
    # However, in practice, the contribution of the second largest and
    # subsequent eigenvalues to the partition function decreases fast, so it
    # is sufficient to calculate only a few of the largest eigenvalues.
    if num_tm_eigvals is None:
        num_eigvals = min(num_neighbors ** 2, num_rows - 2)
    else:
        num_eigvals = min(num_tm_eigvals, num_rows - 2)
    # For three or two interactions we take all eigenvalues
    if len(interactions_1) <= 3:
        w_matrix_dense = w_matrix.todense()
        w_all_norm_eigvals: np.ndarray = scipy.linalg.eig(w_matrix_dense)
        w_norm_eigvals = w_all_norm_eigvals[0]
    else:
        w_norm_eigvals: np.ndarray = sparse_eigs(
            w_matrix, k=num_eigvals, which="LM", return_eigenvectors=False
        )
    eigvals_norms: np.ndarray = np.abs(w_norm_eigvals)
    max_eigval_norm_idx = eigvals_norms.argmax()
    max_eigval_norm = eigvals_norms[max_eigval_norm_idx]
    reduced_eigvals = w_norm_eigvals / max_eigval_norm
    reduced_eigvals_contrib = np.sum(reduced_eigvals ** (num_neighbors))
    cellunit = 2
    helm_free_erg = -(temp / cellunit) * (
        max_w_log_elem_1
        + max_w_log_elem_2
        + np.log(max_eigval_norm)
        + np.log(reduced_eigvals_contrib.real) / num_neighbors
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
    # Evaluate the largest eigenvalue only.
    num_eigvals = 1
    w_norm_eigvals: np.ndarray
    # noinspection PyTypeChecker
    w_norm_eigvals, _ = sparse_eigs(
        w_matrix, k=num_eigvals, which="LM", return_eigenvectors=True
    )
    # max_eigvals = w_norm_eigvals.real[0]
    eigvals_norms: np.ndarray = np.abs(w_norm_eigvals)
    max_eigval_norm_idx = eigvals_norms.argmax()
    max_eigval_norm = eigvals_norms[max_eigval_norm_idx]
    # reduced_eigvals = w_norm_eigvals / max_eigval_norm
    # In the thermodynamic limit, the number of spins is infinity.
    # Accordingly, only the largest reduced eigenvalue contributes.
    reduced_eigvals_contrib = 1.0
    helm_free_erg_tl = -temp * (
        max_w_log_elem
        + log(max_eigval_norm)
        + np.log(reduced_eigvals_contrib.real)
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
    nnz_elems, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem_1 = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem_1
    norm_nnz_elems = np.exp(nnz_elems)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix_1 = csr_matrix(
        (norm_nnz_elems, (nnz_rows, nnz_cols)), shape=w_shape
    )
    # Second matrix
    nnz_elems, nnz_rows, nnz_cols = _csr_log_transfer_matrix_parts_fast(
        temp, mag_field, interactions_2, num_neighbors
    )

    # Normalize nonzero matrix elements.
    max_w_log_elem_2 = np.max(nnz_elems)
    nnz_elems -= max_w_log_elem_2
    norm_nnz_elems = np.exp(nnz_elems)
    # Construct the sparse matrix.
    num_rows = 2 ** num_neighbors
    w_shape = (num_rows, num_rows)
    w_matrix_2 = csr_matrix(
        (norm_nnz_elems, (nnz_rows, nnz_cols)), shape=w_shape
    )
    w_matrix = w_matrix_1 * w_matrix_2
    # Evaluate the largest eigenvalue only.
    num_eigvals = 1
    w_norm_eigvals: np.ndarray
    # noinspection PyTypeChecker
    w_norm_eigvals = sparse_eigs(
        w_matrix, k=num_eigvals, which="LM", return_eigenvectors=False
    )
    max_eigval = w_norm_eigvals.real[0]
    # In the thermodynamic limit, the number of spins is infinity.
    # Accordingly, only the largest reduced eigenvalue contributes.
    cellunit = 2
    helm_free_erg_tl = -(temp / cellunit) * (
        max_w_log_elem_1 + max_w_log_elem_2 + log(max_eigval)
    )
    return helm_free_erg_tl


def free_energy_fast(
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
    print(w_norm_eigvals)
    helm_free_erg_tl = -temp * (log(max_eigvals) + max_w_log_elem)
    return helm_free_erg_tl


def grid_func_base(
    params: t.Tuple[float, float],
    interactions: np.ndarray,
    interactions_2: np.ndarray = None,
    finite_chain: bool = False,
    num_tm_eigvals: int = None,
):
    """Calculate the spin chain properties over a parameter grid."""
    temperature, magnetic_field = params
    num_neighbors = len(interactions)
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


def eval_energy(
    params_grid: ParamsGrid,
    interactions: np.ndarray,
    interactions_2: np.ndarray = None,
    finite_chain: bool = False,
    num_tm_eigvals: int = None,
    num_workers: int = None,
):
    """Calculate the energy over a parameter grid."""
    grid_func = partial(
        grid_func_base,
        interactions=interactions,
        interactions_2=interactions_2,
        finite_chain=finite_chain,
        num_tm_eigvals=num_tm_eigvals,
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
