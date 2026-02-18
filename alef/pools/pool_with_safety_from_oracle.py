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

from typing import Union, Sequence, Optional
import numpy as np
from alef.utils.plotter import Plotter
from alef.utils.utils import row_wise_compare, row_wise_unique
from alef.oracles.base_oracle import BaseOracle, StandardOracle
from alef.oracles.base_constrained_oracle import StandardConstrainedOracle
from alef.oracles.helpers.constrained_sampler import ConstrainedSampler
from alef.pools.base_pool_with_safety import BasePoolWithSafety
import matplotlib.pyplot as plt


class PoolWithSafetyFromOracle(BasePoolWithSafety):
    def __init__(
        self,
        oracle: StandardOracle,
        safety_oracle: Union[BaseOracle, Sequence[BaseOracle]]=None,
        seed: int=123,
        set_seed: bool=False
    ):
        assert isinstance(oracle, StandardOracle), NotImplementedError
        self.oracle = oracle
        if isinstance(safety_oracle, list):
            assert all(isinstance(so, StandardOracle) for so in safety_oracle), NotImplementedError
            self.safety_oracle = safety_oracle
        else:
            assert isinstance(safety_oracle, StandardOracle), NotImplementedError
            self.safety_oracle = [safety_oracle]
        super().__init__(safety_dimension=len(self.safety_oracle))

        self.__ConstrainedHelper = ConstrainedSampler()
        self.__x = None
        if set_seed:
            np.random.seed(seed)
    
    def get_max(self):
        if hasattr(self.oracle, 'get_max'):
            return self.oracle.get_max()
        else:
            D = self.get_dimension()
            if D>3:
                print(f'dimension is {D}, this might take a while')
            X, Y, Z = self.get_random_data(int(100**D), noisy=False)
            return max(Y)
    
    def get_constrained_max(
        self, 
        constraint_lower: float =-np.inf,
        constraint_upper: float = np.inf
    ):
        D = self.get_variable_dimension()
        if D>3:
            print(f'dimension is {D}, this might take a while')
        X, Y, Z = self.get_random_constrained_data(int(100**D), noisy=False, constraint_lower=constraint_lower, constraint_upper=constraint_upper)
        return max(Y)

    def query(self, x, noisy: bool=True):
        if np.shape(np.atleast_2d(x))[0] > 1:
            raise ValueError("please query only 1 point")

        idx = row_wise_compare(self.__x, x)
        idx = np.where(idx)[0]
        if len(idx) < 1 and not self.get_query_non_exist():
            raise ValueError('queried point does not exist')
        
        y = self.oracle.query(x, noisy)
        z = []
        for oracle in self.safety_oracle:
            z.append(oracle.query(x, noisy))
        z = np.array(z)

        if not self._with_replacement:
            self.__x = np.delete(self.__x,(idx),axis=0)
        
        return y, z
    
    def get_grid_data(self, n_per_dim:int, noisy:bool=True):
        """
        input: n_per_dim noisy
        """
        if not hasattr(self.oracle, 'get_grid_data'):
            raise NotImplementedError(f'{self.oracle.__class__.__name__} does not have \'get_grid_data\' method')
        if np.any([not hasattr(so, 'get_grid_data') for so in self.safety_oracle]):
            raise NotImplementedError('At least one safety_oracle does not have \'get_grid_data\' method')
        
        X, Y = self.oracle.get_grid_data(n_per_dim, noisy)
        Z = np.empty([X.shape[0], len(self.safety_oracle)])
        for i in range(X.shape[0]):
            for j in range(len(self.safety_oracle)):
                Z[i,j] = self.safety_oracle[j].query(X[i,:], noisy)
        return X, Y, Z

    def get_random_data(self, n, noisy=True):
        X, Y = self.oracle.get_random_data(n, noisy)
        Z = np.empty([X.shape[0], len(self.safety_oracle)])
        for i in range(X.shape[0]):
            for j in range(len(self.safety_oracle)):
                Z[i,j] = self.safety_oracle[j].query(X[i,:], noisy)
        return X, Y, Z
    
    def get_random_data_in_box(
        self, n: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        noisy:bool=True
    ):
        X, Y = self.oracle.get_random_data_in_box(n, a, box_width, noisy)
        Z = np.empty([X.shape[0], len(self.safety_oracle)])
        for i in range(X.shape[0]):
            for j in range(len(self.safety_oracle)):
                Z[i,j] = self.safety_oracle[j].query(X[i,:], noisy)
        return X, Y, Z

    def get_random_constrained_data(
        self, n : int,
        noisy : bool=True,
        constraint_lower: Union[float, Sequence[float]] =-np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf
    ):
        return self.__ConstrainedHelper.get_random_constrained_data(
            StandardConstrainedOracle(self.oracle, self.safety_oracle),
            n,
            noisy,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper
        )

    def get_random_constrained_data_in_box(
        self, n: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        noisy : bool=True,
        constraint_lower: Union[float, Sequence[float]] =-np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf
    ):
        return self.__ConstrainedHelper.get_random_constrained_data_in_box(
            StandardConstrainedOracle(self.oracle, self.safety_oracle),
            n,
            a,
            box_width,
            noisy,
            constraint_lower=constraint_lower,
            constraint_upper=constraint_upper
        )

    def get_box_bounds(self, *args, **kwargs):
        return self.oracle.get_box_bounds(*args, **kwargs)
    def get_dimension(self, *args, **kwargs):
        return self.oracle.get_dimension(*args, **kwargs)

    def set_data(self, x_data: np.ndarray):
        r"""
        set x manually
        """
        self.__x = x_data.copy()

    def discretize_random(self,n : int):
        r"""
        set x randomly from the space defined in the oracle (get discretized input space from the oracle)
        """
        self.__x = np.random.uniform(*self.get_box_bounds(), size=(n, self.get_dimension()))

    def possible_queries(self):
        return self.__x.copy()

    def get_context_status(self, *args, **kwargs):
        return self.oracle.get_context_status(*args, **kwargs)

