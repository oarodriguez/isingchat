from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ParamsGrid(Iterable):
    """"""

    temperature: np.ndarray
    magnetic_field: np.ndarray

    @property
    def shape(self):
        return self.temperature.shape + self.magnetic_field.shape

    @property
    def size(self):
        """The grid total size."""
        return int(np.prod(self.shape))

    def __iter__(self):
        """"""
        grid_shape = self.shape
        for ndindex in np.ndindex(*grid_shape):
            temp_idx, mag_field_idx = ndindex
            yield (
                self.temperature[temp_idx],
                self.magnetic_field[mag_field_idx],
            )


@dataclass
class ParamsGrid3D(Iterable):
    """"""

    spin_spin_dist: np.ndarray
    temperature: np.ndarray
    magnetic_field: np.ndarray

    @property
    def shape(self):
        return self.spin_spin_dist.shape + self.temperature.shape + self.magnetic_field.shape

    @property
    def size(self):
        """The grid total size."""
        return int(np.prod(self.shape))

    def __iter__(self):
        """"""
        grid_shape = self.shape
        for ndindex in np.ndindex(*grid_shape):
            spin_spin_dist_idx, temp_idx, mag_field_idx = ndindex
            yield (
                self.spin_spin_dist[spin_spin_dist_idx],
                self.temperature[temp_idx],
                self.magnetic_field[mag_field_idx],
            )
