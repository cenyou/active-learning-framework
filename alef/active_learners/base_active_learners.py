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
from typing import List, Union
from alef.models.base_model import BaseModel
from alef.enums.active_learner_enums import ValidationType
from alef.oracles.base_oracle import BaseOracle
from alef.pools.base_pool import BasePool
from alef.utils.utils import calculate_multioutput_rmse

class BaseActiveLearner(abc.ABC):

    def __init__(self):
        super().__init__()
        self.validation_metrics: pd.DataFrame = None
        self.validation_type: Union[ValidationType, List[ValidationType]] = None
        self.validation_at: List[int] = None
        self.x_data: np.ndarray = None
        self.y_data: np.ndarray = None
        self.x_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.data_history: pd.DataFrame = None
        self.ground_truth_available: bool = False
        self.do_plotting: bool = False
        self.save_plots: bool = False
        self.plot_path: str = ""
        self.save_result: bool = False
        self.summary_path: str = ""

    def initialize_validation_metrics(self, metric_name_list: List[str]):
        """
        initializes the validation metrics dataframe
        """
        self.validation_metrics = pd.DataFrame(columns=['step_index', *metric_name_list])

    def initialize_multioutput_validation_metrics(
        self,
        output_dimension: int,
        multioutput_metric_name_list: List[str],
        metric_name_list: List[str]
    ):
        """
        initializes the validation metrics dataframe
        :param: output_dimension - dimension of outputs (num of columns per metric)
        :param: multioutput_metric_name_list - metrics applied to each output dim
        :param: metric_name_list - metrics applied to all output dim jointly
        """
        columns_paren = ['step_index']
        columns_child = ['']
        for metric in multioutput_metric_name_list:
            columns_paren.extend([metric] * output_dimension)
            columns_child.extend([f'output_{i}' for i in range(output_dimension)])
        columns_paren.extend(metric_name_list)
        columns_child.extend(['']*len(metric_name_list))
        self.validation_metrics = pd.DataFrame(columns=[columns_paren, columns_child])

    def _compute_validation_on_y_one_type(self, validation_type: ValidationType):
        r"""
        :return: [metric values]
        """
        if validation_type == ValidationType.RMSE:
            pred_mu, pred_sigma = self.model.predictive_dist(self.x_test)
            rmse = np.sqrt(np.mean(np.power(pred_mu - np.squeeze(self.y_test), 2.0)))
            metrics = [rmse]
        elif validation_type == ValidationType.MAE:
            pred_mu, pred_sigma = self.model.predictive_dist(self.x_test)
            mae = np.mean(np.absolute(pred_mu - np.squeeze(self.y_test) ))
            metrics = [mae]
        elif validation_type == ValidationType.NEG_LOG_LIKELI:
            log_likelis = self.model.predictive_log_likelihood(self.x_test, self.y_test)
            neg_log_likeli = np.mean(-1 * log_likelis)
            metrics = [neg_log_likeli]
        elif validation_type == ValidationType.RMSE_MULTIOUTPUT:
            pred_mu, pred_sigma = self.model.predictive_dist(self.x_test)
            rmse_array = calculate_multioutput_rmse(pred_mu, self.y_test)
            metrics = rmse_array
        else:
            raise NotImplementedError
        return metrics

    def compute_validation_on_y(self):
        r"""
        :return: [metric values]
        """
        if isinstance(self.validation_type, ValidationType):
            return self._compute_validation_on_y_one_type(self.validation_type)
        else: # self.validation_type: List[ValidationType]
            metrics = []
            for v in self.validation_type:
                assert not v==ValidationType.RMSE_MULTIOUTPUT, NotImplementedError
                metrics.append(*self._compute_validation_on_y_one_type(v))
            return metrics

    def compute_empty_validation_on_y(self):
        r"""
        :return: [metric values]
        """
        if isinstance(self.validation_type, ValidationType):
            assert not self.validation_type == ValidationType.RMSE_MULTIOUTPUT, NotImplementedError
            return [float('nan')]
        else: # self.validation_type: List[ValidationType]
            assert not any([v == ValidationType.RMSE_MULTIOUTPUT for v in self.validation_type]), NotImplementedError
            return [float('nan')]* len(self.validation_type)

    def add_validation_value(self, idx: int, metric_list: List[float]):
        """
        adds validation values to the validation metrics dataframe
        """
        self.validation_metrics.loc[len(self.validation_metrics), :] = [idx, *metric_list]

    def set_model(self, model: BaseModel):
        """
        sets the model that is used for active learning
        """
        self.model = model

    def set_do_plotting(self, do_plotting: bool):
        """
        Method for specifying if plotting should be done

        Arguments:
            do_plotting - bool : flag if plotting should be done
        """
        self.do_plotting = do_plotting

    def save_plots_to_path(self, plot_path: str):
        """
        method to specify that plots are saved to a path
        """
        self.save_plots = True
        self.plot_path = plot_path

    def save_experiment_summary_to_path(self, summary_path: str, filename="AL_result.xlsx"):
        """
        method to specify that experiment summary is saved to a path
        """
        self.save_result = True
        self.summary_path = summary_path
        self.summary_filename = filename

    def set_train_set(self, x_train: np.array, y_train: np.array):
        """
        Method for setting the train set manually
        Arguments:
        x_train - np.array - array of shape [n,d] containing the inputs of n training datapoints with dimension d
        y_train - np.array - array of shape [n,1] containing the outputs of n training datapoints
        """
        self.x_data = x_train
        self.y_data = y_train
        self._initialize_data_history()

    def add_train_point(self, idx, query, new_y):
        """
        Method for adding a single datapoint to the training set
        Arguments:
        idx - int - index of the iteration
        query - np.array - array of shape [d] containing the input of the datapoint with dimension d
        new_y - float - the output of the datapoint
        """
        self.x_data = np.vstack((self.x_data, query))
        self.y_data = np.vstack((self.y_data, [new_y]))
        self.data_history.loc[len(self.data_history), :] = [idx, *query, new_y]

    def add_batch_train_points(self, idx, queries, new_y):
        """
        Method for adding a batch of datapoints to the training set
        Arguments:
        idx - int - index of the iteration
        queries - np.array - array of shape [n, d] containing the input of the datapoint with dimension d
        new_y - np.array - array of shape [n, 1] containing the output of the datapoint
        """
        self.x_data = np.vstack((self.x_data, queries))
        self.y_data = np.vstack((self.y_data, new_y))
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
        return self.x_data, self.y_data

    def set_test_set(self, x_test: np.array, y_test: np.array):
        """
        Method for setting the test set manually
        Arguments:
        x_train - np.array - array of shape [n,d] containing the inputs of n test datapoints with dimension d
        y_train - np.array - array of shape [n,1] containing the outputs of n test datapoints
        """
        self.x_test = x_test
        self.y_test = y_test

    def _initialize_data_history(self):
        
        columns = ['step_index']
        columns.extend( [f'x{i}' for i in range(self.x_data.shape[1])] )
        columns.extend( ['y'] )

        self.data_history = pd.DataFrame(
            np.hstack([
                -1 * np.ones( [self.x_data.shape[0], 1]),
                self.x_data,
                self.y_data
            ]),
            columns = columns
        )

    @abc.abstractmethod
    def learn(self, n_steps: int):
        """
        Main Active learning loop - makes n_steps queries (calls the self.oracle object at each query) and updates the model after each query with the new x_data, y_data
        - validation is done after each query

        Arguments:
            n_steps : int - number of active learning iteration/number of queries
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        raise NotImplementedError

    def save_experiment_summary(self):
        print("Saving experiment summary.")
        result = self.validation_metrics

        with pd.ExcelWriter(
            os.path.join(self.summary_path, self.summary_filename),
            mode='w'
        ) as writer:
            self.data_history.to_excel(writer, sheet_name='data')
            self.validation_metrics.to_excel(writer, sheet_name='result')


### Base class for Oracle Based Active Learning
class BaseOracleActiveLearner(BaseActiveLearner):

    def __init__(self):
        super().__init__()
        self.oracle: BaseOracle = None

    def set_oracle(self, oracle: BaseOracle):
        """
        sets the oracle that should be learned and from which queries should be taken
        """
        self.oracle = oracle

    def sample_ground_truth(self):
        """
        samples many noise-free points from the oracle (only used for plotting)
        """
        self.gt_X, self.gt_function_values = self.oracle.get_random_data(2000, noisy=False)
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
        x, y = self.oracle.get_random_data(n_data, noisy=True)
        self.set_test_set(x, y)

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
        x, y = self.oracle.get_random_data(n_data, noisy=True)
        self.set_train_set(x, y)

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
        x, y = self.oracle.get_random_data_in_box(n_data, a, box_width, noisy=True)
        self.set_train_set(x, y)


### Base class for Pool Based Active Learning
class BasePoolActiveLearner(BaseActiveLearner):

    def __init__(self):
        super().__init__()
        self.pool: BasePool = None
        self.plot_data_available: bool = False

    def set_pool(self, pool: BasePool):
        """
        Method to set the pool
        :param pool: BasePool - pool object
        """
        self.pool = pool

    def set_ground_truth(self, gt_X: np.ndarray, gt_function_values: np.ndarray):
        self.ground_truth_available = True
        self.gt_X = gt_X
        self.gt_function_values = gt_function_values

    def set_plot_data(self, plot_X: np.ndarray, plot_Y: np.ndarray):
        self.plot_data_available = True
        self.plot_X = plot_X
        self.plot_Y = plot_Y

    def get_plot_data(self):
        if self.plot_data_available:
            return self.plot_X, self.plot_Y
        else:
            print('plot data is not set, sample from pool...')
            replacement = self.pool.get_replacement()
            self.pool.set_replacement(True)
            plot_X = self.pool.possible_queries()
            plot_Y = self.pool.batch_query(plot_X, noisy=False)
            self.pool.set_replacement(replacement)
            return np.vstack((plot_X, self.x_data)), np.vstack((plot_Y.reshape(-1,1), self.y_data.reshape(-1,1)))

    def set_train_set_by_querying(self, x_data: np.ndarray, noisy: bool = True):
        """
        Method to set train data by providing only x values - observations are then queried from the pool
        """
        N = x_data.shape[0]
        y_data = np.array([
            self.pool.query(x_data[i], noisy=noisy) for i in range(N)
        ]).reshape([N, 1])
        self.set_train_set(x_data, y_data)


if __name__ == "__main__":
    pass
