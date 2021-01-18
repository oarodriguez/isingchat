import typing as t
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

yaml = YAML()
yaml.indent = 4
yaml.default_flow_style = False


def read_param(param_config: t.Union[float, dict, t.Iterable],
               dtype: np.dtype = np.float64):
    """"""
    # A single value becomes a one-element array.
    if isinstance(param_config, (float, int)):
        return np.array([param_config], dtype=dtype)
    # Read data from a dictionary.
    if isinstance(param_config, dict):
        param_min = param_config["min"]
        param_max = param_config["max"]
        num_samples = param_config["num_samples"]
        return np.linspace(param_min, param_max, num=num_samples,
                           dtype=dtype)
    # Try to convert directly to an array.
    return np.asarray(param_config, dtype=dtype)


def read_ising_config(config_path: Path):
    """Read a config file suitable for modeling the 1D-Ising model."""
    if config_path.is_dir():
        config_path = config_path / "config.yml"
        if config_path.is_dir():
            raise ValueError(f"{config_path} is not a file")
        if not config_path.exists():
            raise FileNotFoundError

    with config_path.open("r") as cfp:
        config_data = yaml.load(cfp)

    system_info = config_data["system"]
    hamiltonian_params = system_info["hamiltonian"]
    # Read the hopping terms.
    hop_params_config = hamiltonian_params["hop_params_list"]
    hop_params = read_param(hop_params_config)
    # Read the temperature.
    temp_config = hamiltonian_params["temperature"]
    temperature = read_param(temp_config)
    # Read the magnetic field.
    magnetic_field_config = hamiltonian_params["magnetic_field"]
    magnetic_field = read_param(magnetic_field_config)
    # Execution data
    exec_config = config_data["exec"]
    # Metadata
    metadata = config_data.get("metadata")
    return {
        "system": {
            "hop_params": hop_params,
            "temperature": temperature,
            "magnetic_field": magnetic_field
        },
        "exec": exec_config,
        "metadata": metadata
    }
