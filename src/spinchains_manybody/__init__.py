try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

from ._environ import Environ

__version__ = importlib_metadata.version("spinchains_manybody")

# A variable that holds several useful environment variables for the project.
environ = Environ.from_environ()
