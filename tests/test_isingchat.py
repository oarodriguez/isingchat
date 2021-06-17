import isingchat


def test_version():
    """Verify we have updated the package version."""
    assert isingchat.__version__ == "0.7.0.dev0"
