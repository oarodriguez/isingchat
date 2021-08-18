"""Verify the library top-level functionality."""

import isingchat


def test_version():
    """Verify we have updated the package version."""
    assert isingchat.__version__ == "2021.1.0"
