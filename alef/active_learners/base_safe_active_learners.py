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

import os
import abc
import numpy as np
import pandas as pd
from typing import List, Sequence, Union, Optional
from .base_active_learners import BaseActiveLearner
from alef.models.base_model import BaseModel
from alef.enums.active_learner_enums import ValidationType
from alef.oracles import BaseOracle, StandardOracle, StandardConstrainedOracle, ConstrainedSampler
from alef.pools import BasePool, BasePoolWithSafety, PoolFromOracle, PoolWithSafetyFromOracle

class BaseSafeActiveLearner(BaseActiveLearner):

    def __init__(self):
        super().__init__()
        self.constraint_on_y: bool = False
        self.z_data: np.ndarray = None
        self.x_test_z: np.ndarray = None
        self.z_test: np.ndarray = None
        self.infer_time = {}

    def set_model(self, model: BaseModel, safety_models: Optional[Union[BaseModel, Sequence[BaseModel]]] = None):
        """
        sets surrogate model

        Arguments:
        model - BaseModel - instance of some BaseModel child
        safety_models - BaseModel or list or BaseModel

        Currently, all the models should be producing 1D array, even if we want multiple safety constraint
            (single output or flattened multi output)
        The reason is that I also use this for transfer learning, and we already need MO for source and target,
            so it would be kind of complicated if we also make different safety constraint MO.
        """
        self.model = model
        if self.constraint_on_y:
            assert safety_models is None, \
                'Ambiguous: safety constraint should be on the main model itself, but additional \'safety_models\' are provided'
            self.safety_models = None
            print("ATTENTION - model itself is safety model")
        else:
            if isinstance(safety_models, (list, tuple)):
                self.safety_models = safety_models
            elif isinstance(safety_models, BaseModel):
                self.safety_models = [safety_models]

    def set_train_set(self, x_train: np.array, y_train: np.array, z_train: Optional[np.array] = None):
        """
        Method for setting the train set manually
        Arguments:
        x_train - np.array - array of shape [n,d] containing the inputs of n training datapoints with dimension d
        y_train - np.array - array of shape [n,1] containing the outputs of n training datapoints
        z_train - np.array - array of shape [n,q] containing the safety outputs of n training datapoints with dimension q
        """
        self.x_data = x_train
        self.y_data = y_train
        if not self.constraint_on_y:
            self.z_data = z_train
        self._initialize_data_history()

    def add_train_point(self, idx, query, new_y, new_z=None):
        """
        Method for adding a single datapoint to the training set
        Arguments:
        idx - int - index of the iteration
        query - np.array - array of shape [d] containing the input of the datapoint with dimension d
        new_y - float - the output of the datapoint
        new_z - np.array - array of shape [q] containing the q safety outputs
        """
        self.x_data = np.vstack((self.x_data, query))
        self.y_data = np.vstack((self.y_data, [new_y]))
        if not self.constraint_on_y:
            self.z_data = np.vstack((self.z_data, new_z))
            self.data_history.loc[len(self.data_history), :] = [idx, *query, new_y, *new_z]
        else:
            self.data_history.loc[len(self.data_history), :] = [idx, *query, new_y]

    def add_batch_train_points(self, idx, queries, new_y, new_z=None):
        """
        Method for adding a batch of datapoints to the training set
        Arguments:
        idx - int - index of the iteration
        queries - np.array - array of shape [n, d] containing the input of the datapoint with dimension d
        new_y - np.array - array of shape [n, 1] containing the output of the datapoint
        new_z - np.array - array of shape [n, q] containing the q safety outputs of the n datapoints
        """
        self.x_data = np.vstack((self.x_data, queries))
        self.y_data = np.vstack((self.y_data, new_y))
        if not self.constraint_on_y:
            self.z_data = np.vstack((self.z_data, new_z))
            self.data_history = pd.concat([
                self.data_history,
                pd.DataFrame(
                    np.hstack((np.array([idx]*queries.shape[0])[:, None], queries, new_y, new_z)),
                    columns=self.data_history.columns
                )
            ], ignore_index=True)
        else:
            self.data_history = pd.concat([
                self.data_history,
                pd.DataFrame(
                    np.hstack((np.array([idx]*queries.shape[0])[:, None], queries, new_y)),
                    columns=self.data_history.columns
                )
            ], ignore_index=True)

    def get_data_sets(self):
        """
        Method to get the data set currently present in the active learner
        """
        if self.constraint_on_y:
            return self.x_data, self.y_data
        else:
            return self.x_data, self.y_data, self.z_data

    def set_test_set(self, x_test: np.array, y_test: np.array):
        """
        Method for setting the test set manually
        Arguments:
        x_train - np.array - array of shape [n,d] containing the inputs of n test datapoints with dimension d
        y_train - np.array - array of shape [n,1] containing the outputs of n test datapoints
        """
        self.x_test = x_test
        self.y_test = y_test

    def set_safety_test_set(self, x_test_z: np.array, z_test: np.array):
        """
        Method for setting the safety test set manually
        We can use different test set for safety metrics
        Arguments:
        x_test_z - np.array - array of shape [n,d] containing the inputs of n test datapoints with dimension d
        z_test - np.array - array of shape [n,q] containing the safety outputs of n test datapoints with dimension q
        """
        self.x_test_z = x_test_z
        self.z_test = z_test

    def _initialize_data_history(self):
        
        columns = ['step_index']
        columns.extend( [f'x{i}' for i in range(self.x_data.shape[1])] )
        columns.extend( ['y'] )
        if not self.constraint_on_y:
            columns.extend( [f'z{i}' for i in range(self.z_data.shape[1])] )
            self.data_history = pd.DataFrame(
                np.hstack([
                    -1 * np.ones( [self.x_data.shape[0], 1]),
                    self.x_data,
                    self.y_data,
                    self.z_data
                ]),
                columns = columns
            )
        else:
            self.data_history = pd.DataFrame(
                np.hstack([
                    -1 * np.ones( [self.x_data.shape[0], 1]),
                    self.x_data,
                    self.y_data
                ]),
                columns = columns
            )


### Base class for Oracle Based Active Learning
class BaseOracleSafeActiveLearner(BaseSafeActiveLearner):
    def __init__(self):
        super().__init__()
        self.oracle: Union[StandardOracle, StandardConstrainedOracle] = None
        self.__ConstrainedHelper = ConstrainedSampler()

    def set_oracle(
        self,
        oracle: Union[StandardOracle, StandardConstrainedOracle],
        safety_oracles: Optional[Union[StandardOracle, Sequence[StandardOracle]]] = None
    ):
        """
        sets the oracle (learn from oracle functions)
        if constraint_on_y, provide only one StandardOracle as \'oracle\'
        if oracle is already StandardConstrainedOracle, \'safety_oracles\' is ignored
        """
        if self.constraint_on_y:
            assert isinstance(oracle, StandardOracle), NotImplementedError
            self.oracle = oracle
        else:
            if isinstance(oracle, StandardConstrainedOracle):
                self.oracle = oracle
            else:
                assert isinstance(oracle, StandardOracle), NotImplementedError
                self.oracle = StandardConstrainedOracle(oracle, safety_oracles)

    def sample_ground_truth(self):
        """
        samples many noise-free points from the oracle (only used for plotting)
        """
        N = 2000
        if self.constraint_on_y:
            self.gt_X, self.gt_function_values = self.oracle.get_random_data(N, noisy=False)
        else:
            self.gt_X, self.gt_function_values, self.gt_safety_values = \
                self.oracle.get_random_data(N, noisy=False)

        self.ground_truth_available = True

    def sample_constrained_ground_truth(
        self,
        constraint_lower: Union[float, Sequence[float]] =-np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf
    ):
        """
        samples many noise-free points from the oracle under safety constraints (only used for plotting)
        """
        N = 2000
        if self.constraint_on_y:
            self.gt_X, self.gt_function_values = self.__ConstrainedHelper.get_random_constrained_data(
                self.oracle,
                N,
                noisy=False,
                constraint_lower = constraint_lower,
                constraint_upper = constraint_upper
            )
        else:
            self.gt_X, self.gt_function_values, self.gt_safety_values = \
                self.__ConstrainedHelper.get_random_constrained_data(
                    self.oracle,
                    N,
                    noisy=False,
                    constraint_lower = constraint_lower,
                    constraint_upper = constraint_upper
                )

        self.ground_truth_available = True

    def sample_test_set(self, n_data: int, seed: int = 100, set_seed: bool = False):
        """
        Method for sampling the test dataset directly from the oracle object

        Arguments:
        n_data - int - number of test datapoints that should be sampled
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        """
        if set_seed:
            np.random.seed(seed)
        out_tuple = self.oracle.get_random_data(n_data, noisy=True)
        self.set_test_set(*out_tuple[:2])

    def sample_constrained_test_set(
        self,
        n_data: int,
        seed: int = 100,
        set_seed: bool = False,
        constraint_lower: Union[float, Sequence[float]] =-np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf
    ):
        """
        Method for sampling the test constrained dataset directly from the oracle object

        Arguments:
        n_data - int - number of test datapoints that should be sampled
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        constraint_lower - float or list of floats - lower bound constraint
        constraint_upper - float or list of floats - upper bound constraint
        """
        if set_seed:
            np.random.seed(seed)
        out_tuple = self.__ConstrainedHelper.get_random_constrained_data(
            self.oracle,
            n_data,
            noisy=True,
            constraint_lower = constraint_lower,
            constraint_upper = constraint_upper
        )
        self.set_test_set(*out_tuple[:2])

    def sample_safety_test_set(self, n_data: int, seed: int = 100, set_seed: bool = False):
        """
        Method for sampling the safety test dataset directly from the safety oracle object

        Arguments:
        n_data - int - number of test datapoints that should be sampled
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        """
        if set_seed:
            np.random.seed(seed)
        assert not self.constraint_on_y

        x, z = self.oracle.constraint_oracle[0].get_random_data(n_data, noisy=True)
        for i in range(1, len(self.oracle.constraint_oracle)):
            zi = [self.oracle.constraint_oracle[i].query(x[j], noisy=True) for j in range(n_data)]
            zi = np.array(zi).reshape([n_data, 1])
            z = np.hstack((z, zi))
        self.set_safety_test_set(x, z)

    def sample_constrained_safety_test_set(
        self,
        n_data: int,
        seed: int = 100,
        set_seed: bool = False,
        constraint_lower: Union[float, Sequence[float]] =-np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf
    ):
        """
        Method for sampling the safety test constrained dataset directly from the safety oracle object

        Arguments:
        n_data - int - number of test datapoints that should be sampled
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        constraint_lower - float or list of floats - lower bound constraint
        constraint_upper - float or list of floats - upper bound constraint
        """
        if set_seed:
            np.random.seed(seed)
        assert not self.constraint_on_y
        x, _, z = self.__ConstrainedHelper.get_random_constrained_data(
            self.oracle,
            n_data,
            noisy=True,
            constraint_lower = constraint_lower,
            constraint_upper = constraint_upper
        )
        self.set_safety_test_set(x, z)

    def sample_train_set(self, n_data, seed=100, set_seed=False):
        """
        Method for sampling the initial dataset directly from the oracle object

        Arguments:
        n_data - int - number of initial datapoints that should be sampled
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        """
        if set_seed:
            np.random.seed(seed)
        self.set_train_set(*self.oracle.get_random_data(n_data, noisy=True))

    def sample_train_set_in_box(self, n_data, a=0.0, box_width=1.0, seed=100, set_seed=False):
        """
        Method for sampling the initial dataset directly from the oracle object

        Arguments:
        n_data - int - number of initial datapoints that should be sampled
        a - float - lower bound of the box
        box_width - float - width of the box
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        """
        if set_seed:
            np.random.seed(seed)
        self.set_train_set(*self.oracle.get_random_data_in_box(n_data, a, box_width, noisy=True))

    def sample_constrained_train_set(
        self,
        n_data: int,
        seed: int = 100,
        set_seed: bool = False,
        constraint_lower: Union[float, Sequence[float]] =-np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf
    ):
        """
        Method for sampling the initial constrained dataset directly from the oracle object

        Arguments:
        n_data - int - number of initial datapoints that should be sampled
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        constraint_lower - float or list of floats - lower bound constraint
        constraint_upper - float or list of floats - upper bound constraint
        """
        if set_seed:
            np.random.seed(seed)
        self.set_train_set(
            *self.__ConstrainedHelper.get_random_constrained_data(
                self.oracle,
                n_data,
                noisy=True,
                constraint_lower = constraint_lower,
                constraint_upper = constraint_upper
            )
        )

    def sample_constrained_train_set_in_box(
        self,
        n_data: int,
        a: Union[float, Sequence[float]],
        box_width: Union[float, Sequence[float]],
        seed: int = 100,
        set_seed: bool = False,
        constraint_lower: Union[float, Sequence[float]] =-np.inf,
        constraint_upper: Union[float, Sequence[float]] = np.inf
    ):
        """
        Method for sampling the initial constrained dataset directly from the oracle object

        Arguments:
        n_data - int - number of initial datapoints that should be sampled
        a - float or list of floats - lower bound of the box
        box_width - float or list of floats - width of the box
        seed - int - seed for sampling if one should use one
        set_seed - bool - flag if seed should be set
        constraint_lower - float or list of floats - lower bound constraint
        constraint_upper - float or list of floats - upper bound constraint
        """
        if set_seed:
            np.random.seed(seed)
        self.set_train_set(
            *self.__ConstrainedHelper.get_random_constrained_data_in_box(
                self.oracle,
                n_data,
                a, box_width,
                noisy=True,
                constraint_lower = constraint_lower,
                constraint_upper = constraint_upper
            )
        )

### Base class for Pool Based Active Learning
class BasePoolSafeActiveLearner(BaseSafeActiveLearner):
    def __init__(self):
        super().__init__()
        self.pool: BasePool = None
        self.plot_data_available: bool = False

    def set_pool(self, pool: Union[BasePool, BasePoolWithSafety]):
        """
        Method to set the pool
        :param pool: BasePool or BasePoolWithSafety - pool object
        """
        if self.constraint_on_y:
            assert isinstance(pool, BasePool), 'Ambiguous: constraint is on y, but pool is not a BasePool'
        else:
            assert isinstance(pool, BasePoolWithSafety), 'Ambiguous: constraint is on additional measurements z, but pool is not a BasePoolWithSafety'
        self.pool = pool

    def set_ground_truth(self, gt_X: np.ndarray, gt_function_values: np.ndarray, gt_safety_values: Optional[np.ndarray] = None):
        self.ground_truth_available = True
        self.gt_X = gt_X
        self.gt_function_values = gt_function_values
        if not self.constraint_on_y:
            self.gt_safety_values = gt_safety_values

    def set_plot_data(self, plot_X: np.ndarray, plot_Y: np.ndarray, plot_Z: Optional[np.ndarray] = None):
        self.plot_data_available = True
        self.plot_X = plot_X
        self.plot_Y = plot_Y
        if not self.constraint_on_y:
            self.plot_Z = plot_Z

    def get_plot_data(self):
        if self.plot_data_available:
            if self.constraint_on_y:
                return self.plot_X, self.plot_Y
            else:
                return self.plot_X, self.plot_Y, self.plot_Z
        else:
            print('plot data is not set, sample from pool...')
            replacement = self.pool.get_replacement()
            self.pool.set_replacement(True)
            plot_X = self.pool.possible_queries()
            if self.constraint_on_y:
                plot_Y = self.pool.batch_query(plot_X, noisy=False)
            else:
                plot_Y, plot_Z = self.pool.batch_query(plot_X, noisy=False)
            self.pool.set_replacement(replacement)

            if self.constraint_on_y:
                return np.vstack((plot_X, self.x_data)), np.vstack((plot_Y.reshape(-1,1), self.y_data.reshape(-1,1)))
            else:
                return np.vstack((plot_X, self.x_data)), np.vstack((plot_Y.reshape(-1,1), self.y_data.reshape(-1,1))), np.vstack((plot_Z, self.z_data))

    def set_train_set_by_querying(self, x_data: np.ndarray, noisy: bool = True):
        """
        Method to set train data by providing only x values - observations are then queried from the pool
        """
        N = x_data.shape[0]
        if self.constraint_on_y:
            y = self.pool.batch_query(x_data, noisy=noisy)
            self.set_train_set(x_data, y.reshape([N, 1]))
        else:
            y, z = self.pool.batch_query(x_data, noisy=noisy)
            self.set_train_set(x_data, y.reshape([N, 1]), z)


if __name__ == "__main__":
    pass
