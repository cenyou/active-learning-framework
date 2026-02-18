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

import time
import numpy as np
import random
import time
from typing import List, Optional, Tuple, Union
from alef.acquisition_functions.acquisition_function_factory import AcquisitionFunctionFactory
from alef.acquisition_functions.al_acquisition_functions.base_al_acquisition_function import BaseALAcquisitionFunction
from alef.configs.acquisition_functions.al_acquisition_functions.base_al_acquisition_function_config import BaseALAcquisitionFunctionConfig
from alef.utils.plot_utils import (
    active_learning_1d_plot,
    active_learning_1d_plot_with_acquisition,
    active_learning_2d_plot,
    active_learning_nd_plot,
    plot_model_specifics,
    active_learning_1d_plot_multioutput,
)
from alef.enums.active_learner_enums import ValidationType
from alef.utils.utils import calculate_multioutput_rmse
from alef.active_learners.base_active_learners import BasePoolActiveLearner


class PoolActiveLearner(BasePoolActiveLearner):
    """
    Main class for non-batch pool-based active learning - one query at a time - inherits from BasePoolActiveLearner

    Attributes:
        acquisition_function : BaseALAcquisitionFunction
        validation_type : Union[ValidationType, List[ValidationType]] - Enum which validation metric should be used e.g. RMSE, NegLoglikeli
        model : BaseModel - Model object that is an instance of a child class of BaseModel such as e.g. GPModel
        use_smaller_acquisition_set : bool - Bool if only a sampled subset of the pool should used for query selection (saves computational budget)
        smaller_set_size : int - Number of samples from the pool used for acquisition calculation
    """

    def __init__(
        self,
        acquisition_function: BaseALAcquisitionFunction,
        validation_type: Union[ValidationType, List[ValidationType]],
        use_smaller_acquistion_set: bool = False,
        smaller_set_size: int = 200,
        validation_at: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        self.acquisition_function = acquisition_function
        assert isinstance(self.acquisition_function, BaseALAcquisitionFunction)
        self.validation_type = validation_type
        self.validation_at = validation_at
        if isinstance(self.validation_type, ValidationType):
            assert not self.validation_type == ValidationType.RMSE_MULTIOUTPUT, NotImplementedError
            _metric_names = [self.validation_type.name]
        else:
            assert not any([v == ValidationType.RMSE_MULTIOUTPUT for v in self.validation_type]), NotImplementedError
            _metric_names = [v.name for v in self.validation_type]
        validatetion_items = [
            *_metric_names,
            'inference_time',
            'acq_func_opt_time',
            'query_time', # = inference_time + acq_func_opt_time or acq_func_opt_time if random
            'iteration_without_validate_time',
            'plotting_time',
            'validate_time',
            'iteration_time'
        ]
        self.initialize_validation_metrics(metric_name_list=validatetion_items)

        self.use_smaller_acquistion_set = use_smaller_acquistion_set
        self.smaller_set_size = smaller_set_size
        self.inference_time = {}
        self.acquisition_func_opt_time = {}
        self.query_time = {}
        self.iteration_time_without_validation = {}
        self.plotting_time = {}
        
    def reduce_pool(self, grid: np.ndarray, new_pool_size: int):
        """
        Gets a grid of points and reduces the grid - is used to reduce the pool for acquisition function calculation
        """
        print("-Reduce acquisition set ")
        grid_size = grid.shape[0]
        if grid_size > new_pool_size:
            indexes = np.random.choice(list(range(0, grid_size)), new_pool_size, replace=False)
            new_pool = grid[indexes]
            return new_pool
        else:
            return grid

    def update(self, idx: int=1000, record_time: bool=True):
        """
        Main update function - infers the model on the current dataset, calculates the acquisition function and returns the query location
        """
        time_before_inference = time.perf_counter()
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        time_after_inference = time.perf_counter()
        x_pool = self.pool.possible_queries()

        if self.use_smaller_acquistion_set:
            x_pool = self.reduce_pool(x_pool, self.smaller_set_size)

        acquisition_function_scores = self.acquisition_func(x_pool)
        new_query = x_pool[np.argmax(acquisition_function_scores)]
        time_after_acquisition_opt = time.perf_counter()
        if record_time:
            inference_time = time_after_inference - time_before_inference
            acquisition_opt_time = time_after_acquisition_opt - time_after_inference
            self.inference_time[idx] = inference_time
            self.acquisition_func_opt_time[idx] = acquisition_opt_time
            if self.acquisition_function.require_fitted_model:
                self.query_time[idx] = inference_time + acquisition_opt_time
            else:
                self.query_time[idx] = acquisition_opt_time
        return new_query

    def acquisition_func(self, x_pool: np.array) -> np.array:
        """
        Wrapper around acquisition_function object
        """
        scores = self.acquisition_function.acquisition_score(x_pool, self.model)
        return scores

    def learn(self, n_steps: int, start_index=0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main Active learning loop - makes n_steps queries and updates the model after each query with the new x_data, y_data
        - validation is done after each query

        Arguments:
            n_steps : int - number of active learning iteration/number of queries
            start_index : int - important for plotting and logging - indicates that already start_index-1 AL steps where done previously
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        self.n_steps = n_steps

        if self.do_plotting:
            if not self.plot_data_available:
                self.set_plot_data(*self.get_plot_data())
        # warmup
        self.update(-1, record_time = False)
        for i in range(start_index, start_index + self.n_steps):
            time_before_iteration = time.perf_counter()
            query = self.update(idx=i)
            validate_this_iter = self.validation_at is None or len(self.validation_at)==0 or i in self.validation_at

            print(f"Iter {i}: Query")
            print(query)
            new_y = self.pool.query(query)
            time_after_query = time.perf_counter()
            self.iteration_time_without_validation[i] = time_after_query - time_before_iteration
            if self.do_plotting and validate_this_iter:
                self.plot(query, new_y, i)
            time_after_plotting = time.perf_counter()
            self.plotting_time[i] = time_after_plotting - time_after_query
            self.add_train_point(i, query, new_y)
            if validate_this_iter:
                self.validate(i)
            else:
                self.empty_validate(i)
        if self.save_result:
            self.save_experiment_summary()
        return self.validation_metrics, self.x_data

    def get_time_lists(self):
        """
        returns the time lists that were measured

        Returns:
            list - containes inference times over iterations
            list - containes acquisiton opt times over iterations
            list - contains complete iteration time over iterations
            list - containes iteration time without validation - only updata + query time
        """
        return (
            self.validation_metrics['inference_time'].to_list(),
            self.validation_metrics['acq_func_opt_time'].to_list(),
            self.validation_metrics['iteration_time'].to_list(),
            self.validation_metrics['iteration_without_validate_time'].to_list()
        )

    def empty_validate(self, idx: int):
        metrics = super().compute_empty_validation_on_y()
        self.add_validation_value(idx, [
            *metrics,
            self.inference_time[idx],
            self.acquisition_func_opt_time[idx],
            self.query_time[idx],
            self.iteration_time_without_validation[idx],
            self.plotting_time[idx],
            0.0,
            self.iteration_time_without_validation[idx] + self.plotting_time[idx]
        ])

    def validate(self, idx: int):
        """
        Validates the currently infered model on the test set - uses the validation_type to choose which metric should be used
        """
        print('Validate')

        t0 = time.perf_counter()
        metrics = super().compute_validation_on_y()
        t1 = time.perf_counter()
        self.add_validation_value(idx, [
            *metrics,
            self.inference_time[idx],
            self.acquisition_func_opt_time[idx],
            self.query_time[idx],
            self.iteration_time_without_validation[idx],
            self.plotting_time[idx],
            t1 - t0,
            self.iteration_time_without_validation[idx] + self.plotting_time[idx] + t1 - t0,
        ])

    def plot(self, query: np.ndarray, new_y: np.ndarray, step: int):
        """
        Plotting function - gets the actual query and the AL step index and produces plots depending on the input and output dimension
        if self.plots is True the plots are saved to self.plot_path (both variables are set in the parent class)
        """
        print("Plotting...")
        dimension = self.x_data.shape[1]
        output_dimension = self.y_data.shape[1]
        x_plot, y_plot = self.get_plot_data()
        if output_dimension == 1:
            if dimension == 1:
                pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
                
                plot_name = "query_" + str(step) + ".png"
                active_learning_1d_plot(
                    x_plot, pred_mu, pred_sigma, self.x_data, self.y_data, query, new_y,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
                # self.plot_1d(np.atleast_1d(query), np.atleast_1d(new_y), step, x_plot)

            elif dimension == 2:
                pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
                acquisition_function = self.model.entropy_predictive_dist(x_plot)
                plot_name = "query_" + str(step) + ".png"
                active_learning_2d_plot(
                    x_plot,
                    acquisition_function,
                    pred_mu,
                    y_plot,
                    self.x_data,
                    query,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
                plot_name = "query_" + str(step) + ".png"
                active_learning_nd_plot(self.x_data, self.y_data, self.save_plots, plot_name, self.plot_path)

            if self.save_plots:
                plot_name = "model_specific" + str(step) + ".png"
                plot_model_specifics(
                    x_plot, self.x_data, self.model, save_plot=self.save_plots, file_name=plot_name, file_path=self.plot_path
                )
            else:
                plot_model_specifics(x_plot, self.x_data, self.model)
        else:
            if dimension == 1:
                pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
                active_learning_1d_plot_multioutput(
                    x_plot, pred_mu, pred_sigma, self.x_data, self.y_data, query, new_y, save_plot=False, file_name=None, file_path=None
                )

    def plot_1d(self, query, new_y, step, x_plot):
        pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
        acquisition_on_plot = self.acquisition_func(x_plot)
        if self.save_plots:
            plot_name = "query_" + str(step) + ".png"
            if self.ground_truth_available:
                active_learning_1d_plot_with_acquisition(
                    x_plot,
                    pred_mu,
                    pred_sigma,
                    acquisition_on_plot,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    self.ground_truth_available,
                    self.gt_X,
                    self.gt_function_values,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
                active_learning_1d_plot_with_acquisition(
                    x_plot,
                    pred_mu,
                    pred_sigma,
                    acquisition_on_plot,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
        else:
            if self.ground_truth_available:
                active_learning_1d_plot_with_acquisition(
                    x_plot,
                    pred_mu,
                    pred_sigma,
                    acquisition_on_plot,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    self.ground_truth_available,
                    self.gt_X,
                    self.gt_function_values,
                )
            else:
                active_learning_1d_plot_with_acquisition(
                    x_plot, pred_mu, pred_sigma, acquisition_on_plot, self.x_data, self.y_data, query, new_y
                )
                # pred_mu,pred_cov = self.model.predictive_dist(x_plot)


if __name__ == "__main__":
    pass
