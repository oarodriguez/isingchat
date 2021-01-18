import json
import pathlib
from functools import partial

import click
from rich import box
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text
from ruamel.yaml import YAML
from spinchains_manybody.io import read_ising_config
from spinchains_manybody.ising import (
    ParamsGrid, eval_energy, grid_func_base
)

from .exceptions import CLIError
from .utils import (
    DaskProgressBar, RichProgressBar, columns, console
)

yaml = YAML()
yaml.indent = 4
yaml.default_flow_style = False


@click.command()
@click.argument("config-path", type=click.Path(exists=True,
                                               dir_okay=True,
                                               resolve_path=True))
def ising(config_path: str):
    """Properties of the 1D Ising chain with long-range interactions."""
    _config_path = pathlib.Path(config_path)
    if _config_path.is_dir():
        _config_path = _config_path / "config.yml"
        config_path = _config_path / "config.yml"
        if _config_path.is_dir():
            raise CLIError(f"{config_path} is not a file")

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

    # The grid used to arrange the configuration info.
    config_panel = Panel(config_grid,
                         title="[yellow]Execution Summary",
                         title_align="left")
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

    # Export the data.
    # TODO.

    console.print(Padding("🎉 [green bold]Execution completed 🎉",
                          pad=(1, 1)), justify="center")
