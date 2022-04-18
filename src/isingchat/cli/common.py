from pathlib import Path

import attr


@attr.s(auto_attribs=True)
class Paths:
    """Paths to the files where results are stored."""

    config: Path
    free_energy: Path
    eigenvalues: Path
    cor_length: Path
    cor_function: Path

    @classmethod
    def from_path(cls, path: Path):
        """Define the file paths based on an existing path."""
        config_path = path
        energy_path = path.with_suffix(".free-energy.h5")
        eigenvalues_path = path.with_suffix(".eigenvalues.h5")
        cor_length = path.with_suffix(".cor-length.h5")
        cor_function = path.with_suffix(".cor-function.h5")
        return cls(config=config_path,
                   free_energy=energy_path,
                   eigenvalues=eigenvalues_path,
                   cor_length=cor_length,
                   cor_function=cor_function)
