# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Union, Sequence, List
import numpy as np
import os
from alef.utils.utils import row_wise_compare, row_wise_unique, check1Dlist
from alef.pools.base_pool import BasePool
import matplotlib.pyplot as plt
import logging
from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)

class PoolMultioutputFromData(BasePool):
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        data_is_noisy: bool=True,
        observation_noise: float = None,
        seed:int=123,
        set_seed:bool=False
    ):
        super().__init__()
        self.set_data( x_data, y_data )
        self.set_data_is_noisy(data_is_noisy)
        self.observation_noise = observation_noise
        if set_seed:
            np.random.seed(seed)
    
    def set_data(self, x_data: np.ndarray, y_data: np.ndarray):
        r"""
        set x and y manually
        """
        assert x_data.ndim == 2
        assert y_data.ndim == 2
        assert y_data.shape[1] > 1

        self.__x = x_data.copy()
        self.__y = y_data.copy()
        
        self.__dimension = self.__x.shape[1]
        self.__output_dimension = self.__y.shape[1]

        assert self.__x.shape[0] == self.__y.shape[0]

    def set_data_is_noisy(self, noisy:bool):
        self.__data_is_noisy = noisy

    def query(self, x, noisy: bool=True):
        if np.shape(np.atleast_2d(x))[0] > 1:
            raise ValueError("please query only 1 point")
        
        idx = row_wise_compare(self.__x, x)
        idx = np.where(idx)[0] # np.where return a tuple, idx here is an array
        
        if len(idx) < 1 and not self.get_query_non_exist():
            raise ValueError('queried point does not exist')
        if len(idx) < 1 and self.get_query_non_exist():
            raise NotImplementedError('queried point does not exist')
        elif len(idx) > 1:
            idx = np.random.choice(idx)
        else:
            idx = idx[0]

        y = self.__y[idx, :]
        
        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return y
        else:
            return y

    def batch_query(self, X: np.ndarray, noisy: bool):
        if self.__data_is_noisy and not noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            noisy = True
        Y = [self.query(X[..., i,:], noisy).reshape(1, -1) for i in range(X.shape[-2])]
        return np.concatenate(Y, axis=-2)

    def get_grid_data(self, n_per_dim: int, noisy: bool):
        """
        input: n_per_dim noisy
        """
        raise NotImplementedError
    
    def get_data_from_idx(self, idx: Sequence[int], noisy: bool):

        x = self.__x[idx].copy()
        y = self.__y[idx].copy()

        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return x, y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return x, y
        else:
            return x, y

    def get_random_data(self, n: int, noisy: bool):
        N_pool = self.__x.shape[0]
        
        idx = np.random.choice(N_pool, n, replace = self._with_replacement)

        x = self.__x[idx].copy()
        y = self.__y[idx].copy()

        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return x, y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return x, y
        else:
            return x, y
    
    def get_random_data_in_box(
        self, n: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        noisy: bool=False
    ):
        a = np.array(check1Dlist(a, self.get_dimension()))
        box_width = np.array(check1Dlist(box_width, self.get_dimension()))
        b = a+box_width
        
        mask = np.all(self.__x >= a, axis=1) * np.all(self.__x <= b, axis=1)
        if mask.sum() < n:
            raise ValueError("does not have enough data within the specified box")
        
        idx = np.where(mask)[0]
        idx = np.random.choice(idx, n, replace= self._with_replacement)

        x = self.__x[idx].copy()
        y = self.__y[idx].copy()

        if not self._with_replacement:
            self.__x = np.delete(self.__x, idx, axis=0)
            self.__y = np.delete(self.__y, idx, axis=0)
        
        if noisy and not self.__data_is_noisy:
            return x, y + self.generate_gaussian_noise(y.shape)
        elif not noisy and self.__data_is_noisy:
            logger.warning("cannot return noise-free data when the data is noisy")
            logger.warning("return noisy data instead")
            return x, y
        else:
            return x, y

    def get_random_constrained_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_random_constrained_data_in_box(self, *args, **kwargs):
        raise NotImplementedError

    def get_box_bounds(self):
        a = self.__x.min(axis=0)
        b = self.__x.max(axis=0)
        return a, b
    
    def get_dimension(self, *args, **kwargs):
        return self.__dimension

    def generate_gaussian_noise(self, size: Sequence[int]):
        return np.random.normal(0, self.observation_noise, size=size) 

    def get_full_data(self):
        return self.__x.copy(), self.__y.copy()

    def possible_queries(self):
        return self.__x.copy()

if __name__ == '__main__':
    import pandas as pd
    import os
    from alef.data_sets.pytest_set import PytestMOSet
    
    dataset = PytestMOSet()
    dataset.load_data_set()

    X, Y = dataset.get_complete_dataset()

    pool = PoolMultioutputFromData(X, Y)
    
    xx, yy = pool.get_full_data()
    print(xx.shape)
    print(yy.shape)