"""isingchat - A library to calculate the properties of a 1D Ising spin chain.

Copyright © 2021, Omar Abel Rodríguez-López.
"""
try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

from ._environ import Environ

metadata = importlib_metadata.metadata("isingchat")  # type: ignore

# Export package information.
__version__ = metadata["version"]
__author__ = metadata["author"]
__description__ = metadata["description"]
__license__ = metadata["license"]

__all__ = [
    "__author__",
    "__description__",
    "__license__",
    "__version__",
    "metadata",
]
environ = Environ.from_environ()
