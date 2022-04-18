import json
import pathlib
from functools import partial

import click
import h5py
import numpy as np
from isingchat.exec_ import ParamsGrid, ParamsGrid3D
from isingchat.io import read_ising_config, save_free_energy_data, \
    save_eigen_data, save_cor_length_data, save_cor_function_data
from isingchat.ising import eval_energy, grid_func_base, \
    grid_func_base_cor_func, grid_func_base_eigenvalues, \
    grid_func_base_cor_length_tl
from rich import box
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text
from ruamel.yaml import YAML

from isingchat.cli.common import Paths
from isingchat.cli.exceptions import CLIError
from isingchat.cli.utils import DaskProgressBar, RichProgressBar, columns, \
    console

yaml = YAML()
yaml.indent = 2
yaml.default_flow_style = False
