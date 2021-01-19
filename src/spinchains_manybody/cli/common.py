from pathlib import Path

import attr


@attr.s(auto_attribs=True)
class Paths:
    """Paths to the files where results are stored."""
    config: Path
    energy: Path

    @classmethod
    def from_path(cls, path: Path):
        """Define the file paths based on an existing path."""
        if path.is_dir():
            config_path = path / "ising.yml"
            energy_path = config_path.with_name("energy.h5")
        else:
            config_path = path
            energy_path = path.with_suffix(".energy.h5")
        return cls(config=config_path,
                   energy=energy_path)
