import pathlib
from functools import partial

import click
from spinchains_manybody.io import read_ising_config
from spinchains_manybody.ising import (
    ParamsGrid, eval_energy, grid_func_base
)

from .exceptions import CLIError
from .utils import (
    DaskProgressBar, RichProgressBar, columns, console
)


@click.command()
@click.argument("config-path", type=click.Path(exists=True,
                                               dir_okay=True,
                                               resolve_path=True))
def ising(config_path: str):
    """Properties of the 1D Ising chain with long-range interactions."""
    _config_path = pathlib.Path(config_path)
    if _config_path.is_dir():
        _config_path = _config_path / "config.yml"
        if not _config_path.exists():
            raise CLIError

    # Read the config data.
    config_data = read_ising_config(_config_path)
    system_data = config_data["system"]
    temperature = system_data["temperature"]
    magnetic_field = system_data["magnetic_field"]
    hop_params = system_data["hop_params"]
    exec_config = config_data["exec"]
    parallel = exec_config["parallel"]

    # Evaluate the grid.
    params_grid = ParamsGrid(temperature, magnetic_field)
    if not parallel:
        progress_bar = RichProgressBar(*columns,
                                       console=console,
                                       auto_refresh=False)
        grid_task = progress_bar.add_task("[red]Progress",
                                          total=params_grid.size)
        with progress_bar:
            energy_data = []
            grid_func = partial(grid_func_base,
                                hop_params=hop_params)
            grid_map = map(grid_func, params_grid)
            for energy_value in grid_map:
                energy_data.append(energy_value)
                progress_bar.update(grid_task, advance=1)
                progress_bar.refresh()
    else:
        with DaskProgressBar():
            energy_data = eval_energy(params_grid, hop_params)
