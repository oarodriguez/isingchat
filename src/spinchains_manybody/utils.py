def bin_digits(value: int, length: int = None):
    """Get a list with the binary digits of an integer.

    Fill the list with zeros to the left to get a list of length ``length``.
    If ``length`` is ``None``, return only the value digits.
    """
    length = length or 0
    fmt_str = f"{{0:0{length}b}}"
    return [int(v) for v in fmt_str.format(value)]
