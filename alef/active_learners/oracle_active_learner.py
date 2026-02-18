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
import time
from typing import List, Optional, Union
from alef.acquisition_functions.al_acquisition_functions.acq_random import Random
from alef.acquisition_functions.al_acquisition_functions.base_al_acquisition_function import BaseALAcquisitionFunction
from alef.enums.active_learner_enums import ValidationType, OracleALAcquisitionOptimizationType
from alef.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot

from .base_active_learners import BaseOracleActiveLearner

class OracleActiveLearner(BaseOracleActiveLearner):
    
    """
    Main class for non-batch oracle-based active learning - one query at a time - collects queries by calling its oracle object

    Main Attributes:
        acquisition_function : BaseALAcquisitionFunction
        validation_type : Union[ValidationType, List[ValidationType]] - Enum which validation metric should be used e.g. RMSE, NegLoglikeli
        model : BaseModel - Model object that is an instance of a child class of BaseModel such as e.g. GPModel
        acquisiton_optimization_type : OracleALAcquisitionOptimizationType - specifies how the acquisition function should be optimized
        oracle: BaseOracle - oracle object for which a surrogate model should be learned and which gets called
    """

    def __init__(
        self,
        acquisition_function: BaseALAcquisitionFunction,
        validation_type: Union[ValidationType, List[ValidationType]],
        acquisiton_optimization_type: OracleALAcquisitionOptimizationType,
        validation_at: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        self.acquisition_function = acquisition_function
        assert isinstance(self.acquisition_function, BaseALAcquisitionFunction)
        self.acquisiton_optimization_type = acquisiton_optimization_type
        self.random_shooting_n = 5000
        self.validation_type = validation_type
        self.validation_at = validation_at
        if isinstance(self.validation_type, ValidationType):
            assert not self.validation_type == ValidationType.RMSE_MULTIOUTPUT, NotImplementedError
            _metric_names = [self.validation_type.name]
        else:
            assert not any([v == ValidationType.RMSE_MULTIOUTPUT for v in self.validation_type]), NotImplementedError
            _metric_names = [v.name for v in self.validation_type]
        self.initialize_validation_metrics(metric_name_list=[
            *_metric_names,
            'inference_time',
            'acq_func_opt_time',
            'query_time',
            'validate_time'
        ])
        self.inference_time = {}
        self.acq_func_opt_time = {}
        self.query_time = {}

    def update(self, idx=0, record_time: bool=True):
        """
        Main update function - infers the model on the current dataset, calculates the acquisition function and returns the query location -
        Acquisition function optimization is specified in self.acquisition_function_optimization_type
        TODO: Add evolutionary acquisition function optimizer from BO also here
        """
        t0 = time.perf_counter()
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        t1 = time.perf_counter()
        box_a, box_b = self.oracle.get_box_bounds()
        dimensions = self.oracle.get_dimension()

        if (
            isinstance(self.acquisition_function, Random)
            or self.acquisiton_optimization_type == OracleALAcquisitionOptimizationType.RANDOM_SHOOTING
        ):
            x_pool = np.random.uniform(low=box_a, high=box_b, size=(self.random_shooting_n, dimensions))
            acquisition_function_scores = self.acquisition_function.acquisition_score(x_pool, self.model)
            new_query = x_pool[np.argmax(acquisition_function_scores)]
        else:
            raise NotImplementedError
        t2 = time.perf_counter()
        if record_time:
            self.inference_time[idx] = t1 - t0
            self.acq_func_opt_time[idx] = t2 - t1
            if self.acquisition_function.require_fitted_model:
                self.query_time[idx] = t2 - t0
            else:
                self.query_time[idx] = t2 - t1
        return new_query

    def learn(self, n_steps: int, start_index: int =0):
        """
        Main Active learning loop - makes n_steps queries (calls the self.oracle object at each query) and updates the model after each query with the new x_data, y_data
        - validation is done after each query

        Arguments:
            n_steps : int - number of active learning iteration/number of queries
            start_index : int - starting index for the iteration count
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        if self.do_plotting:
            self.sample_ground_truth()
        self.n_steps = n_steps
        # warmup
        self.update(-1, record_time = False)
        # start
        for i in range(start_index, start_index + self.n_steps):
            query = self.update(i)
            validate_this_iter = self.validation_at is None or len(self.validation_at)==0 or i in self.validation_at

            print(f"Iter {i}: Query")
            print(query)
            new_y = self.oracle.query(query)
            if self.do_plotting and validate_this_iter:
                self.plot(query, new_y, i)
            self.add_train_point(i, query, new_y)
            if validate_this_iter:
                self.validate(i)
            else:
                self.empty_validate(i) # timer can still be recorded
        if self.save_result:
            self.save_experiment_summary()
        return self.validation_metrics, self.x_data

    def empty_validate(self, idx: int):
        metrics = super().compute_empty_validation_on_y()
        self.add_validation_value(idx, [
            *metrics,
            self.inference_time[idx],
            self.acq_func_opt_time[idx],
            self.query_time[idx],
            0.0
        ])

    def validate(self, idx: int):
        """
        Validates the currently infered model on the test set - uses the validation_type to choose which metric should be used
        """
        print('Validate')

        t_start = time.perf_counter()
        metrics = super().compute_validation_on_y()
        t_end = time.perf_counter()
        validate_time = t_end - t_start

        self.add_validation_value(idx, [
            *metrics,
            self.inference_time[idx],
            self.acq_func_opt_time[idx],
            self.query_time[idx],
            validate_time
        ])

    def plot(self, query: np.array, new_y: float, step: int):
        print("Plotting...")
        dimension = self.x_data.shape[1]
        x_plot = self.gt_X
        y_plot = self.gt_function_values
        if dimension == 1:
            pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
            
            plot_name = "query_" + str(step) + ".png"
            active_learning_1d_plot(
                x_plot, pred_mu, pred_sigma, self.x_data, self.y_data, query, new_y, True, self.gt_X, self.gt_function_values,
                save_plot=self.save_plots,
                file_name=plot_name,
                file_path=self.plot_path,
            )
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
            print("Dimension too high for plotting")

if __name__ == "__main__":
    pass
