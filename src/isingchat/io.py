import typing as t

import h5py
import numpy as np
from isingchat.exec_ import ParamsGrid
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
    if interactions_2_config is not None:
        interactions_2 = read_param(interactions_2_config)
    # Read the temperature.
    inv_temp_config = hamiltonian_params.get('inv_temperature', None)
    if inv_temp_config is not None:
        inv_temp = read_param(inv_temp_config)
    else:
        inv_temp = None
    temp_config = hamiltonian_params.get("temperature",None)
    if temp_config is not None:
        temperature = read_param(temp_config)
    else:
        temperature = None
    # Read the magnetic field.
    magnetic_field_config = hamiltonian_params["magnetic_field"]
    magnetic_field = read_param(magnetic_field_config)
    # Read the spin_spin_dist
    if hamiltonian_params.get('spin_spin_dist') is not None:
        spin_spin_dist_config = hamiltonian_params["spin_spin_dist"]
        spin_spin_dist = read_param(spin_spin_dist_config)
    else:
        spin_spin_dist = None
    # Execution data
    exec_config = config_data.get("exec", None)
    if exec_config is None:
        exec_config = {"parallel": False}
    # Metadata
    metadata = config_data.get("metadata")
    # Use centroymmetric property
    use_centrosymmetric = config_data.get("use_centrosymmetric",None)
    # Use inv_data_temp
    is_inv_temp = config_data.get("is_inv_temp", None)
    return {
        "system": {
            "interactions": interactions,
            "temperature": temperature,
            "magnetic_field": magnetic_field,
            "finite": finite_system,
            "num_tm_eigvals": num_tm_eigvals,
            "spin_spin_dist": spin_spin_dist,
            "inv_temperature": inv_temp
        },
        "exec": exec_config,
        "metadata": metadata,
        "use_centrosymmetric": use_centrosymmetric,
        "is_inv_temp": is_inv_temp
    }


def save_free_energy_data(energy: np.ndarray, h5_file: h5py.File):
    """Save the energy data into an HDF5 file."""
    h5_file.create_dataset("free-energy", data=energy)

def save_eigen_data(data: list,params_grid: ParamsGrid, h5_file: h5py.File):
    """Save the energy data into an HDF5 file."""
    data_array = np.asarray(data, dtype=np.ndarray)
    grid_shape = params_grid.shape
    max_w_matrix_data = np.asarray(data_array[:,0],dtype=np.float64).reshape(grid_shape)
    h5_group = h5_file.create_group("eigens")
    h5_group.create_dataset("max-w-matrix", data=max_w_matrix_data)
    eigenvalues_data = np.asarray(data_array[:,1],dtype=np.ndarray).reshape(grid_shape)
    h5_group.create_dataset("eigenvalues",data=eigenvalues_data.tolist())
    eigenvectors_data = np.asarray(data_array[:,2],dtype=np.ndarray).reshape(grid_shape)
    h5_group.create_dataset("eigenvectors", data=eigenvectors_data.tolist())

def save_cor_length_data(cor_length: np.ndarray, h5_file: h5py.File):
    """Save the energy data into an HDF5 file."""
    h5_file.create_dataset("correlation-length", data=cor_length)

def save_cor_function_data(cor_function: np.ndarray, h5_file: h5py.File):
    """Save the energy data into an HDF5 file."""
    h5_file.create_dataset("correlation-function", data=cor_function)
