import pathlib

import click
from spinchains_manybody.exceptions import CLIError


@click.command()
@click.argument("config-path", type=click.Path(exists=True,
                                               dir_okay=True,
                                               resolve_path=True))
def ising(config_path: str):
    """Properties of the 1D Ising chain with long-range interactions."""
    config_path_ = pathlib.Path(config_path)
    if config_path_.is_dir():
        config_path_ = config_path_ / "config.yml"
        if not config_path_.exists():
            raise CLIError
