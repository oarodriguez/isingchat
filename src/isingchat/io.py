import typing as t

import h5py
import numpy as np
from ruamel.yaml import YAML

yaml = YAML()
yaml.indent = 4
yaml.default_flow_style = False


def read_param(
    param_config: t.Union[float, dict, t.Iterable],
    dtype: np.dtype = np.float64,
):
    """"""
    # A single value becomes a one-element array.
    if isinstance(param_config, (float, int)):
        return np.array([param_config], dtype=dtype)
    # Read data from a dictionary.
    if isinstance(param_config, dict):
        param_min = param_config["min"]
        param_max = param_config["max"]
        num_samples = param_config.get("num_samples", None)
        step_size = param_config.get("step_size", None)
        if num_samples is None and step_size is None:
            raise ValueError(
                "a value for 'step_size' or 'num_samples' is " "required"
            )
        if num_samples and step_size:
            raise ValueError(
                "'step_size' and 'num_samples' are not "
                "compatible; give only one of them"
            )
        elif num_samples:
            return np.linspace(
                param_min, param_max, num=num_samples, dtype=dtype
            )
        else:
            return np.arange(param_min, param_max, step_size)
    # Try to convert directly to an array.
    return np.asarray(param_config, dtype=dtype)


def read_ising_config(config_data: dict):
    """Read a config file suitable for modeling the 1D-Ising model."""
    system_info = config_data["system"]
    finite_system = system_info.get("finite", False)
    num_tm_eigvals = system_info.get("num_tm_eigvals", None)
    hamiltonian_params = system_info["params"]
    # Read the hopping terms.
    interactions_config = hamiltonian_params["interactions"]
    interactions = read_param(interactions_config)
    # Read others hopping terms
    interactions_2_config = hamiltonian_params.get("interactions_2", None)
    if interactions_2_config is None:
        interactions_2 = None
    else:
        interactions_2 = read_param(interactions_2_config)
    # Read the temperature.
    temp_config = hamiltonian_params["temperature"]
    temperature = read_param(temp_config)
    # Read the magnetic field.
    magnetic_field_config = hamiltonian_params["magnetic_field"]
    magnetic_field = read_param(magnetic_field_config)
    # Execution data
    exec_config = config_data.get("exec", None)
    if exec_config is None:
        exec_config = {"parallel": False}
    # Metadata
    metadata = config_data.get("metadata")
    return {
        "system": {
            "interactions": interactions,
            "interactions_2": interactions_2,
            "temperature": temperature,
            "magnetic_field": magnetic_field,
            "finite": finite_system,
            "num_tm_eigvals": num_tm_eigvals,
        },
        "exec": exec_config,
        "metadata": metadata,
    }


def save_free_energy_data(energy: np.ndarray, h5_file: h5py.File):
    """Save the energy data into an HDF5 file."""
    h5_file.create_dataset("free-energy", data=energy)
