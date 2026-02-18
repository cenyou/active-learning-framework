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

import numpy as np
import os
import abc
from copy import deepcopy
from typing import Tuple, Union, Sequence, List
from matplotlib import pyplot as plt
from alef.utils.utils import create_grid, create_grid_multi_bounds, check1Dlist
from alef.oracles.helpers.context_interface import ContextSupport
from alef.oracles.base_oracle import BaseOracle, StandardOracle

class StandardConstrainedOracle(BaseOracle, ContextSupport):
    def __init__(
        self,
        oracle: StandardOracle,
        constraint: Union[StandardOracle, List[StandardOracle]],
        #constraint_value: Union[float, List[float]] = 0.0,
        #constraint_is_lower_bound: Union[bool, List[bool]] = False,
    ):
        """
        :param oracle: the main oracle
        :param constraint: the constraint oracle(s)
        """
        self.set_oracle(oracle, constraint)

    def set_oracle(
        self,
        oracle: StandardOracle,
        constraint: Union[StandardOracle, List[StandardOracle]],
    ):
        """
        :param oracle: the main oracle
        :param constraint: the constraint oracle(s)
        """
        assert isinstance(oracle, StandardOracle), NotImplementedError
        self.oracle = oracle
        if isinstance(constraint, list):
            assert all(isinstance(c, StandardOracle) for c in constraint)
            self.constraint_oracle = constraint
        else:
            assert isinstance(constraint, StandardOracle)
            self.constraint_oracle = [constraint]
        assert all(c.get_dimension() == self.oracle.get_dimension() for c in self.constraint_oracle)
        assert all(np.allclose(c.get_box_bounds(), self.oracle.get_box_bounds()) for c in self.constraint_oracle)


    def query(self, x: np.ndarray, noisy: bool = True) -> Tuple[float, np.ndarray]:
        """
        main method for querying the oracle and constraint

        :param x: the input
        :param noisy: whether to add noise or not

        :return: (float, np.ndarray[float]) - 
            the function value and the constraint value(s)
            the constraint(s) are concatenated into a [n_constraint,] array
        """
        y = self.oracle.query(x, noisy=noisy)
        z = np.array([c.query(x, noisy=noisy) for c in self.constraint_oracle])
        return y, z

    def get_max(self):
        raise NotImplementedError

    def batch_query(self, X: np.ndarray, noisy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        batch query method for querying the oracle and constraint

        :param X: [N, D] array - the inputs
        :param noisy: bool - whether to add noise or not

        :return:
            [N,] np.ndarray[float]
            [N, n_constraints] np.ndarray[float]
        """
        Y = self.oracle.batch_query(X, noisy=noisy)
        Z = np.hstack([
            c.batch_query(X, noisy=noisy).reshape(-1, 1) for c in self.constraint_oracle
        ])
        return Y, Z

    def get_grid_data(self, n_per_dim: int, noisy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        get the grid data for the oracle and constraint

        :param n_per_dim: int - the number of points per dimension
        :param noisy: bool - whether to add noise or not

        :return: ([N, D] np.ndarray, [N, 1] np.ndarray, [N, n_constraint] np.ndarray) - 
            the grid data for the oracle and the constraint
            (X, Y, Z)
        """
        a, b = self.get_box_bounds()
        X = create_grid(a, b, n_per_dim, self.get_variable_dimension())
        X = self._decorate_variable_with_context(X)
        Y, Z = self.batch_query(X, noisy=noisy)
        return X, Y.reshape(-1, 1), Z

    def get_random_data(self, n: int, noisy: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        get random data for the oracle and constraint

        :param n: int - the number of points
        :param noisy: bool - whether to add noise or not

        :return: ([N, D] np.ndarray, [N, 1] np.ndarray, [N, n_constraint] np.ndarray) - 
            the random data for the oracle and the constraint
            (X, Y, Z)
        """
        a, b = self.get_box_bounds()
        X = np.random.uniform(low=a, high=b, size=(n, self.get_variable_dimension()))
        X = self._decorate_variable_with_context(X)
        Y, Z = self.batch_query(X, noisy=noisy)
        return X, Y.reshape(-1, 1), Z

    def get_random_data_in_box(
        self,
        n: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        noisy: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates random n uniform random points in the box [a, a + box_width]

        :param n: int - the number of points
        :param a: float or [D,] array - the lower bound of the box
        :param box_width: float or [D,] array - the width of the box
        :param noisy: bool - whether to add noise or not

        :return: ([N, D] np.ndarray, [N, 1] np.ndarray, [N, n_constraint] np.ndarray) - 
            the random data for the oracle and the constraint
            (X, Y, Z)
        """
        aa = np.array(check1Dlist(a, self.get_dimension()))
        bw = np.array(check1Dlist(box_width, self.get_dimension()))
        bb = aa + bw
        use_context, context_idx, context_values = self.get_context_status(return_idx=True, return_values=True)
        if use_context:
            aa[context_idx] = context_values
            bb[context_idx] = context_values

        l, u = self.get_box_bounds()
        assert np.all(aa <= u)
        assert np.all(bb >= l)
        assert np.any(aa < u)
        assert np.any(bb > l)

        if hasattr(l, '__len__'):
            aa[aa < l] = l[aa < l]
            bb[bb > u] = u[bb > u]
        else:
            aa[aa < l] = l
            bb[bb > u] = u

        X = np.random.uniform(low=aa, high=bb, size=(n, self.get_dimension()))
        Y, Z = self.batch_query(X, noisy=noisy)
        return X, Y.reshape(-1, 1), Z

    def get_box_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        get the box bounds for the oracle

        :return: (np.ndarray, np.ndarray) - the lower and upper bounds
        """
        return self.oracle.get_box_bounds()

    def set_box_bounds(self, a: float, b: float):
        """
        set the box bounds for the oracle & constraint

        :param a: float - the lower bound
        :param b: float - the upper bound
        """
        self.oracle.set_box_bounds(a, b)
        for c in self.constraint_oracle:
            c.set_box_bounds(a, b)

    def get_dimension(self) -> int:
        """
        get the dimension of the oracle

        :return: int - the dimension
        """
        return self.oracle.get_dimension()








