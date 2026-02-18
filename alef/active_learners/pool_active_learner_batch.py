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
import random
import time
from typing import List, Optional, Union

from tensorflow.python.ops.gen_batch_ops import Batch
from alef.acquisition_functions.al_acquisition_functions.acq_random import Random
from alef.acquisition_functions.al_acquisition_functions.base_al_acquisition_function import BaseALAcquisitionFunction
from alef.acquisition_functions.al_acquisition_functions.base_batch_al_acquisition_function import BaseBatchALAcquisitionFunction
from alef.utils.plot_utils import (
    active_learning_1d_plot,
    active_learning_2d_plot,
    active_learning_nd_plot,
    plot_model_specifics,
    active_learning_1d_plot_multioutput,
)
from scipy.stats import norm
from typing import Tuple
from alef.enums.active_learner_enums import ValidationType, BatchAcquisitionOptimizationType
from alef.models.base_model import BaseModel
from alef.utils.utils import calculate_multioutput_rmse
from alef.active_learners.base_active_learners import BasePoolActiveLearner
from alef.models.batch_model_interface import BatchModelInterace


class BatchPoolActiveLearner(BasePoolActiveLearner):
    def __init__(
        self,
        acquisition_function: BaseALAcquisitionFunction,
        validation_type: Union[ValidationType, List[ValidationType]],
        batch_size: int,
        use_smaller_acquistion_set: bool = True,
        smaller_set_size: int = 200,
        validation_at: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        self.acquisition_function = acquisition_function
        assert isinstance(self.acquisition_function, (BaseBatchALAcquisitionFunction, Random) )
        self.validation_type = validation_type
        self.validation_at = validation_at
        self.batch_size = batch_size
        self.optimization_type = BatchAcquisitionOptimizationType.GREEDY
        self.use_smaller_acquistion_set = use_smaller_acquistion_set
        self.smaller_set_size = smaller_set_size

    def set_model(self, model: BaseModel):
        assert isinstance(model, BatchModelInterace)
        super().set_model(model)

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

    def update(self):
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        x_pool = self.pool.possible_queries()

        if self.use_smaller_acquistion_set:
            x_pool = self.reduce_pool(x_pool, self.smaller_set_size)

        if isinstance(self.acquisition_function, Random):
            indexes = np.random.choice(list(range(0, x_pool.shape[0])), self.batch_size, replace=False)
            return x_pool[indexes]

        def acquisition_func(batch):
            return self.acquisition_function.acquisition_score(batch, self.model)

        batch = self.optimize(acquisition_func, x_pool)
        return batch

    def optimize(self, acquisition_func, x_pool):
        x_pool = x_pool.copy()
        if self.optimization_type == BatchAcquisitionOptimizationType.GREEDY:
            batch = []
            for i in range(0, self.batch_size):
                print("Batch:")
                print(batch)
                acquistion_set = []
                counter = 0
                for x in x_pool:
                    possible_batch = np.array(batch + [x])
                    acquisition_func_value = acquisition_func(possible_batch)
                    acquistion_set.append(acquisition_func_value)
                    counter += 1
                    if counter % 50 == 0:
                        print(str(counter + 1) + "/" + str(len(x_pool)) + " grid points calculated")
                assert len(acquistion_set) == counter
                best_index = np.argmax(np.array(acquistion_set))
                selected_query = x_pool[best_index]
                x_pool = np.delete(x_pool, best_index, axis=0)
                print("Selected Query:")
                print(selected_query)
                batch.append(selected_query)
            return np.array(batch)

    def learn(self, n_steps: int, start_index=0) -> Tuple[np.ndarray, np.ndarray]:
        self.n_steps = n_steps

        if not isinstance(self.validation_type, ValidationType):
            assert not self.validation_type == ValidationType.RMSE_MULTIOUTPUT, NotImplementedError

        if self.validation_type == ValidationType.RMSE_MULTIOUTPUT:
            self.initialize_multioutput_validation_metrics(
                self.y_data.shape[1],
                multioutput_metric_name_list=[self.validation_type.name],
                metric_name_list=['validate_time']
            )
        else:
            _metric_names = [self.validation_type.name] if isinstance(self.validation_type, ValidationType) else [v.name for v in self.validation_type]
            self.initialize_validation_metrics(metric_name_list=[*_metric_names, 'validate_time'])

        if self.do_plotting:
            if not self.plot_data_available:
                self.set_plot_data(*self.get_plot_data())
        for i in range(start_index, start_index + self.n_steps):
            queries = self.update()
            validate_this_iter = self.validation_at is None or len(self.validation_at)==0 or i in self.validation_at

            print("Queries:")
            print(queries)
            y_queries = []
            for query in queries:
                new_y = self.pool.query(query)
                y_queries.append([new_y])
            y_queries = np.array(y_queries)
            if self.do_plotting and validate_this_iter:
                self.plot(queries, y_queries, i)
            self.add_batch_train_points(i, queries, y_queries)
            print(self.x_data)
            print(self.y_data)
            if validate_this_iter:
                self.validate(i)
            else:
                self.empty_validate(i) # timer can still be recorded
        if self.save_result:
            self.save_experiment_summary()
        return self.validation_metrics, self.x_data

    def empty_validate(self, idx: int):
        if isinstance(self.validation_type, ValidationType) and self.validation_type == ValidationType.RMSE_MULTIOUTPUT:
            self.add_validation_value(idx, [*[float('nan')]*self.y_data.shape[1], 0.0])
        else:
            metrics = super().compute_empty_validation_on_y()
            self.add_validation_value(idx, [*metrics, 0.0])

    def validate(self, idx: int):
        print('Validate')

        t_start = time.perf_counter()
        metrics = super().compute_validation_on_y()
        t_end = time.perf_counter()
        validate_time = t_end - t_start

        self.add_validation_value(idx, [*metrics, validate_time])

    def plot(self, queries: np.ndarray, new_y: np.ndarray, step: int):
        dimension = self.x_data.shape[1]
        output_dimension = self.y_data.shape[1]
        x_plot, y_plot = self.get_plot_data()
        if output_dimension == 1:
            if dimension == 1:
                pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
                if self.save_plots:
                    plot_name = "query_" + str(step) + ".png"
                    if self.ground_truth_available:
                        active_learning_1d_plot(
                            x_plot,
                            pred_mu,
                            pred_sigma,
                            self.x_data,
                            self.y_data,
                            queries,
                            new_y,
                            self.ground_truth_available,
                            self.gt_X,
                            self.gt_function_values,
                            save_plot=self.save_plots,
                            file_name=plot_name,
                            file_path=self.plot_path,
                        )
                    else:
                        active_learning_1d_plot(
                            x_plot,
                            pred_mu,
                            pred_sigma,
                            self.x_data,
                            self.y_data,
                            queries,
                            new_y,
                            save_plot=self.save_plots,
                            file_name=plot_name,
                            file_path=self.plot_path,
                        )
                else:
                    if self.ground_truth_available:
                        active_learning_1d_plot(
                            x_plot,
                            pred_mu,
                            pred_sigma,
                            self.x_data,
                            self.y_data,
                            queries,
                            new_y,
                            self.ground_truth_available,
                            self.gt_X,
                            self.gt_function_values,
                        )
                    else:
                        active_learning_1d_plot(x_plot, pred_mu, pred_sigma, self.x_data, self.y_data, queries, new_y)

            elif dimension == 2:
                pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
                if self.save_plots:
                    plot_name = "query_" + str(step) + ".png"
                    active_learning_2d_plot(
                        x_plot,
                        pred_sigma,
                        pred_mu,
                        y_plot,
                        self.x_data,
                        queries,
                        save_plot=self.save_plots,
                        file_name=plot_name,
                        file_path=self.plot_path,
                    )
                else:
                    active_learning_2d_plot(x_plot, pred_sigma, pred_mu, y_plot, self.x_data, queries)
            else:
                if self.save_plots:
                    plot_name = "query_" + str(step) + ".png"
                    active_learning_nd_plot(self.x_data, self.y_data, self.save_plots, plot_name, self.plot_path)
                else:
                    active_learning_nd_plot(self.x_data, self.y_data)

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
                    x_plot, pred_mu, pred_sigma, self.x_data, self.y_data, queries, new_y, save_plot=False, file_name=None, file_path=None
                )
                # pred_mu,pred_cov = self.model.predictive_dist(x_plot)


if __name__ == "__main__":
    pass
