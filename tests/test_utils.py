from isingchat.utils import bin_digits


def test_bin_digits():
    """"""
    assert bin_digits(6) == "110"
    assert bin_digits(7) == "111"
    assert bin_digits(7, length=0) == "111"
    assert bin_digits(7, length=1) == "111"
    assert bin_digits(7, length=2) == "111"
    assert bin_digits(7, length=3) == "111"
    assert bin_digits(7, length=4) == "0111"
    assert bin_digits(7, length=5) == "00111"
