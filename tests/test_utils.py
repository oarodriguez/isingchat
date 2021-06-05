from isingchat.utils import bin_digits


def test_bin_digits():
    """"""
    assert bin_digits(6) == list(map(int, "110"))
    assert bin_digits(7) == list(map(int, "111"))
    assert bin_digits(7, length=0) == list(map(int, "111"))
    assert bin_digits(7, length=1) == list(map(int, "111"))
    assert bin_digits(7, length=2) == list(map(int, "111"))
    assert bin_digits(7, length=3) == list(map(int, "111"))
    assert bin_digits(7, length=4) == list(map(int, "0111"))
    assert bin_digits(7, length=5) == list(map(int, "00111"))
