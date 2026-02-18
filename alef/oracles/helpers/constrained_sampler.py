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
import logging
from alef.utils.custom_logging import getLogger
from alef.utils.utils import row_wise_compare, row_wise_unique
from alef.oracles.base_oracle import BaseOracle, StandardOracle
from alef.oracles.base_constrained_oracle import StandardConstrainedOracle

logger = getLogger(__name__)

class ConstrainedSampler:
    """
    this is a helper class.
    Generate samples under output constraints from an oracle.
    """
    def get_random_constrained_data(
        self,
        oracle: Union[StandardOracle, StandardConstrainedOracle],
        n : int,
        noisy : bool=True,
        constraint_lower: Union[float, Sequence[float]] =-np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf
    ):
        logger.info('sampling random constrained data...')
        if isinstance(oracle, StandardOracle):
            bound_low, bound_upp = self._decorate_len(constraint_lower, constraint_upper, 1)
            n_constr = 0
            X, Y = None, None

            while n_constr < n:
                X_rd, Y_rd = oracle.get_random_data(max(n - n_constr, 2), noisy=noisy)
                X, Y = self._constrain_data_on_y(X, Y, X_rd, Y_rd, n, bound_low[0], bound_upp[0])
                n_constr = X.shape[0]
                logger.info(f'get {n_constr} samples')

            return X, Y

        elif isinstance(oracle, StandardConstrainedOracle):
            bound_low, bound_upp = self._decorate_len(constraint_lower, constraint_upper, len(oracle.constraint_oracle))
            n_constr = 0
            X, Y, Z = None, None, None

            while n_constr < n:
                X_rd, Y_rd, Z_rd = oracle.get_random_data(max(n - n_constr, 2), noisy)
                X, Y, Z = self._constrain_data_on_z(X, Y, Z, X_rd, Y_rd, Z_rd, n, bound_low, bound_upp)
                n_constr = X.shape[0]
                logger.info(f'get {n_constr} samples')

            return X, Y, Z

        else:
            raise NotImplementedError

    def get_random_constrained_data_in_box(
        self,
        oracle: Union[StandardOracle, StandardConstrainedOracle],
        n: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        noisy : bool=True,
        constraint_lower: Union[float, Sequence[float]] =-np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf
    ):
        logger.info('sampling random constrained data in box...')
        if isinstance(oracle, StandardOracle):
            bound_low, bound_upp = self._decorate_len(constraint_lower, constraint_upper, 1)
            n_constr = 0
            X, Y = None, None

            while n_constr < n:
                X_rd, Y_rd = oracle.get_random_data_in_box(max(n - n_constr, 2), a, box_width, noisy=noisy)
                X, Y = self._constrain_data_on_y(X, Y, X_rd, Y_rd, n, bound_low[0], bound_upp[0])
                n_constr = X.shape[0]
                logger.info(f'get {n_constr} samples')

            return X, Y

        elif isinstance(oracle, StandardConstrainedOracle):
            bound_low, bound_upp = self._decorate_len(constraint_lower, constraint_upper, len(oracle.constraint_oracle))
            n_constr = 0
            X, Y, Z = None, None, None

            while n_constr < n:
                X_rd, Y_rd, Z_rd = oracle.get_random_data_in_box(max(n - n_constr, 2), a, box_width, noisy=noisy)
                X, Y, Z = self._constrain_data_on_z(X, Y, Z, X_rd, Y_rd, Z_rd, n, bound_low, bound_upp)
                n_constr = X.shape[0]
                logger.info(f'get {n_constr} samples')

            return X, Y, Z
        else:
            raise NotImplementedError

    def _decorate_len(
        self,
        constraint_lower: Union[float, Sequence[float]],
        constraint_upper: Union[float, Sequence[float]],
        target_len: int
    ):
        """
        format check,
        return constraint_lower, constraint_upper, which are lists of length target_len
        """
        if not hasattr(constraint_lower, '__len__'):
            bound_low = [constraint_lower] * target_len
        else:
            bound_low = constraint_lower
            assert len(bound_low) == target_len
        
        if not hasattr(constraint_upper, '__len__'):
            bound_upp = [constraint_upper] * target_len
        else:
            bound_upp = constraint_upper
            assert len(bound_upp) == target_len
        
        return bound_low, bound_upp

    def _constrain_data_on_y(
        self,
        X_origin, Y_origin,
        X_new, Y_new,
        n_target : int,
        constraint_lower: float,
        constraint_upper: float
    ):
        mask = np.logical_and(Y_new >= constraint_lower, Y_new <= constraint_upper)[:,0]
        
        if X_origin is None:
            X = X_new[mask]
            Y = Y_new[mask]
        else:
            X = np.vstack((X_origin, X_new[mask]))
            Y = np.vstack((Y_origin, Y_new[mask]))
        
        X, unique_idx = row_wise_unique(X)
        Y = Y[unique_idx]
        
        n_constr = X.shape[0]
        idx = np.random.choice(n_constr, size=min(n_target, n_constr), replace=False)

        return X[idx], Y[idx]

    def _constrain_data_on_z(
        self,
        X_origin, Y_origin, Z_origin,
        X_new, Y_new, Z_new,
        n_target : int,
        constraint_lower: Union[float, Sequence[float]],
        constraint_upper: Union[float, Sequence[float]]
    ):
        mask = np.logical_and(Z_new >= constraint_lower, Z_new <= constraint_upper)
        mask = np.all(mask, axis=-1)
        
        if X_origin is None:
            X = X_new[mask]
            Y = Y_new[mask]
            Z = Z_new[mask]
        else:
            X = np.vstack((X_origin, X_new[mask]))
            Y = np.vstack((Y_origin, Y_new[mask]))
            Z = np.vstack((Z_origin, Z_new[mask]))
        
        X, unique_idx = row_wise_unique(X)
        Y = Y[unique_idx]
        Z = Z[unique_idx]

        n_constr = X.shape[0]
        idx = np.random.choice(n_constr, size=min(n_target, n_constr), replace=False)
        
        return np.atleast_2d(X[idx]), np.atleast_2d(Y[idx]), np.atleast_2d(Z[idx])

