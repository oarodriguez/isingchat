import numpy as np
from numba import njit


def bin_digits(value: int, length: int = None):
    """Get a list with the binary digits of an integer.

    Fill the list with zeros to the left to get a list of length ``length``.
    If ``length`` is ``None``, return only the value digits.
    """
    length = length or 0
    fmt_str = f"{{0:0{length}b}}"
    return [int(v) for v in fmt_str.format(value)]


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


def convert_bin_to_decimal(bin_digits: list):
    """Convert a list of binary digits ``value`` to decimal int."""
    value = 0
    long_bin = len(bin_digits) - 1
    for i, bin in enumerate(bin_digits):
        value += bin * 2 ** (long_bin - i)
    return value
