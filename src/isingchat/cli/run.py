import json
import pathlib
from functools import partial

import click
import h5py
import numpy as np
from isingchat.io import read_ising_config, save_energy_data
from isingchat.ising import (
    ParamsGrid, eval_energy, grid_func_base
)
from rich import box
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text
from ruamel.yaml import YAML

from .common import Paths
from .exceptions import CLIError
from .utils import (
    DaskProgressBar, RichProgressBar, columns, console
)

yaml = YAML()
yaml.indent = 2
yaml.default_flow_style = False


@click.command()
@click.argument("config-path",
                type=click.Path(exists=True,
                                dir_okay=False,
                                resolve_path=True))
@click.option("-f", "--force",
              is_flag=True,
              help="If given, any result stored in the output directory will "
                   "be overwritten.")
def run(config_path: str,
        force: bool):
    """Properties of the 1D Ising chain with long-range interactions.

    CONFIG_PATH: The path to the configuration file.
    """
    # Paths to the files used for saving the results data.
    _config_path = pathlib.Path(config_path)
    paths = Paths.from_path(_config_path)

    # Handle flow if the force flag is enabled.
    _config_path = paths.config
    _energy_data_path = paths.energy
    if not force:
        if _energy_data_path.exists():
            raise CLIError(f"the file '{_energy_data_path.name}' with results "
                           f"exists in the output directory")

    with _config_path.open("r") as cfp:
        config_info = yaml.load(cfp)

    # Read the config data.
    config_data = read_ising_config(config_info)
    system_data = config_data["system"]
    temperature = system_data["temperature"]
    magnetic_field = system_data["magnetic_field"]
    hop_params = system_data["hop_params"]
    exec_config = config_data["exec"]
    exec_parallel = exec_config["parallel"]

    # CLI title message.
    title_text = Text("Ising chain with beyond nearest-neighbor interactions",
                      justify="center", style="bold red")
    title_panel = Panel(title_text, box=box.DOUBLE_EDGE)

    # The grid used to arrange the configuration info.
    config_grid = Table.grid(expand=True, padding=(0, 0), pad_edge=True)
    config_grid.add_column(ratio=1)
    config_grid.add_column(ratio=4)

    # Show the configuration summary.
    config_grid.add_row("Config path", str(config_path))
    config_grid.add_row("Output path", str(_config_path.parent))
    data_str = json.dumps(config_info)
    pretty_config = Pretty(json.loads(data_str),
                           highlighter=console.highlighter,
                           justify="left")
    config_grid.add_row("Config Preview", pretty_config)
    config_panel = Panel(config_grid,
                         title="[yellow]Execution Summary",
                         title_align="left")

    # Display the title and the configuration summary.
    console.print(title_panel)
    console.print(config_panel)

    # Evaluate the energy over the parameters grid.
    params_grid = ParamsGrid(temperature, magnetic_field)
    if not exec_parallel:
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

    grid_shape = params_grid.shape
    energy_array: np.ndarray = np.asarray(energy_data).reshape(grid_shape)

    # Export the data.
    with h5py.File(_energy_data_path, "w") as h5_file:
        save_energy_data(energy_array, h5_file)
    console.print(Padding(f"Results data file '{_energy_data_path}' saved",
                          pad=(0, 1)))

    # Display a nice "Completed" message.
    completed_text = Padding("🎉 [green bold]Execution completed 🎉",
                             pad=1)
    completed_panel = Panel(completed_text, box=box.DOUBLE_EDGE, expand=False)
    console.print(completed_panel, justify="center")
