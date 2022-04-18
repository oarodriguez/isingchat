from . import *

@click.command()
@click.argument(
    "config-path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="If given, any result stored in the output directory will "
    "be overwritten.",
)
def eigens(config_path: str, force: bool):
    """
    Eigenvalues and eigenvectors of the 1D Ising chain with long-range interactions.

    CONFIG_PATH: The path to the configuration file.
    """
    import time
    import psutil

    start = time.perf_counter()
    # Paths to the files used for saving the results data.
    _config_path = pathlib.Path(config_path)
    paths = Paths.from_path(_config_path)

    # Handle flow if the force flag is enabled.
    _config_path = paths.config
    _eigens_data_path = paths.eigenvalues
    if not force:
        if _eigens_data_path.exists():
            raise CLIError(
                f"the file '{_eigens_data_path.name}' with results "
                f"exists in the output directory"
            )

    with _config_path.open("r") as cfp:
        config_info = yaml.load(cfp)

    # Read the config data.
    config_data = read_ising_config(config_info)
    system_data = config_data["system"]
    temperature = system_data["temperature"]
    magnetic_field = system_data["magnetic_field"]
    interactions = system_data["interactions"]
    interactions_2 = system_data.get("interactions_2",None)
    finite_chain = system_data["finite"]
    num_tm_eigvals = system_data["num_tm_eigvals"]
    exec_config = config_data["exec"]
    exec_parallel = exec_config["parallel"]
    num_workers = exec_config.get("num_workers")
    use_centrosymmetric = config_data["use_centrosymmetric"]

    # CLI title message.
    title_text = Text(
        "Ising chain with beyond nearest-neighbor interactions",
        justify="center",
        style="bold red",
    )
    title_panel = Panel(title_text, box=box.DOUBLE_EDGE)

    # The grid used to arrange the configuration info.
    config_grid = Table.grid(expand=True, padding=(0, 0), pad_edge=True)
    config_grid.add_column(ratio=1)
    config_grid.add_column(ratio=4)

    # Show the configuration summary.
    config_grid.add_row("Config path", str(config_path))
    config_grid.add_row("Output path", str(_config_path.parent))
    data_str = json.dumps(config_info)
    pretty_config = Pretty(
        json.loads(data_str), highlighter=console.highlighter, justify="left"
    )
    config_grid.add_row("Config Preview", pretty_config)
    config_panel = Panel(
        config_grid, title="[yellow]Execution Summary", title_align="left"
    )

    # Display the title and the configuration summary.
    console.print(title_panel)
    console.print(config_panel)

    # Evaluate the energy over the parameters grid.
    params_grid = ParamsGrid(temperature, magnetic_field)
    if not exec_parallel:
        progress_bar = RichProgressBar(
            *columns, console=console, auto_refresh=False
        )
        grid_task = progress_bar.add_task(
            "[red]Progress", total=params_grid.size
        )
        with progress_bar:
            eigens_data = []
            grid_func = partial(
                grid_func_base_eigenvalues,
                interactions=interactions,
                interactions_2=interactions_2,
                finite_chain=finite_chain,
                num_tm_eigvals=num_tm_eigvals,
                is_centrosymmetric=use_centrosymmetric
            )
            grid_map = map(grid_func, params_grid)
            for value in grid_map:
                eigens_data.append(value)
                progress_bar.update(grid_task, advance=1)
                progress_bar.refresh()
                # break
    else:
        print('Parallel process is not available')
        return 0

    # Export the data.
    with h5py.File(_eigens_data_path, "w") as h5_file:
        save_eigen_data(eigens_data,params_grid, h5_file)
    console.print(
        Padding(f"Results data file '{_eigens_data_path}' saved", pad=(0, 1))
    )

    # Display a nice "Completed" message.
    completed_text = Padding("🎉 [green bold]Execution completed 🎉", pad=1)
    completed_panel = Panel(completed_text, box=box.DOUBLE_EDGE, expand=False)
    console.print(completed_panel, justify="center")
    end = time.perf_counter()
    elapsed = end - start
    print('total time used: {}'.format(elapsed))
    print('Memory details: {}'.format(psutil.virtual_memory()))
