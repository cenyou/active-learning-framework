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

from typing import Union, List, Sequence, Optional
import os
import sys
import numpy as np
import pandas as pd
import time
from .base_safe_active_learners import BasePoolSafeActiveLearner
from .interface_contextual_input import PoolContextualHelper
from scipy.stats import norm
from scipy.optimize import minimize, NonlinearConstraint, Bounds, differential_evolution
from alef.utils.utils import filter_nan
from alef.utils.safety_metrices import SafetyAreaMeasure
from alef.utils.metric_curve_plotter import MetricCurvePlotter
from alef.utils.plot_utils import safe_bayesian_optimization_1d_plot, safe_bayesian_optimization_2d_plot, safe_bayesian_optimization_2d_plot_without_gt_with_only_safepoints, safety_function_2d_plot, safety_histogram
from alef.enums.data_structure_enums import OutputType
from alef.enums.active_learner_enums import ValidationType
from alef.acquisition_functions.safe_acquisition_functions.base_safe_acquisition_function import BaseSafeAcquisitionFunction
from alef.utils.custom_logging import getLogger

logger = getLogger(__name__)

MAX_POOL_SIZE_PER_UPDATE = 10**5

class PoolSafeActiveLearner(BasePoolSafeActiveLearner, PoolContextualHelper):
    def __init__(
        self,
        acquisition_function: BaseSafeAcquisitionFunction,
        validation_type: Union[ValidationType, List[ValidationType]],
        constraint_on_y: bool=False,
        validation_at: Optional[List[int]] = None,
        train_with_safe_data_only: List[bool]=[False],
        update_by_gradient: bool=False,
        limit_pool_size_per_update: bool=False,
        max_pool_size_per_update: int=MAX_POOL_SIZE_PER_UPDATE,
        **kwargs
    ):
        """
        Main class for safe active learning.
        
        :param acquisition_function: BaseSafeAcquisitionFunction object - compute the constrained acquisition scores
        :param validation_type : Union[ValidationType, List[ValidationType]] - Enum which validation metric should be used
        :param constraint_on_y: bool - whether the safety is constrained directly on the main model or not
        :param validation_at: Optional[List[int]] - list of indices where the validation should be performed, if None, validation is performed every iteration
        :param train_with_safe_data_only: List[bool] - list of booleans indicating whether to train the model with safe data only for each model
        :param update_by_gradient: bool - whether to use gradient based optimization for the acquisition function
        """
        super().__init__()
        self.acquisition_function = acquisition_function
        assert isinstance(self.acquisition_function, BaseSafeAcquisitionFunction)
        self.validation_type = validation_type
        self.validation_at = validation_at
        self.constraint_on_y = constraint_on_y
        self.num_of_models = 1 + self.acquisition_function.number_of_constraints - int(self.constraint_on_y)  # the number of f & q_i, q are safety functions
        assert len(train_with_safe_data_only) in [1, self.num_of_models]
        self.train_with_safe_data_only = train_with_safe_data_only if len(train_with_safe_data_only) == self.num_of_models else train_with_safe_data_only*self.num_of_models
        self.update_by_gradient = update_by_gradient
        self.infer_time = {}
        self.acq_func_opt_time = {}
        self.query_time = {}
        if isinstance(self.validation_type, ValidationType):
            assert not self.validation_type == ValidationType.RMSE_MULTIOUTPUT, NotImplementedError
            _metric_names = [self.validation_type.name]
        else:
            assert not any([v == ValidationType.RMSE_MULTIOUTPUT for v in self.validation_type]), NotImplementedError
            _metric_names = [v.name for v in self.validation_type]
        self.initialize_validation_metrics(metric_name_list=[
            *_metric_names,
            'safe_bool',
            'unsafe_distance',
            *[f'infer_time_m{i}' for i in range(self.num_of_models)],
            'acq_func_opt_time',
            'query_time',
            'validate_time'
        ])
        self._limit_pool_size_per_update = limit_pool_size_per_update
        self._max_pool_size_per_update = max_pool_size_per_update

    def gradient_based_update(self, idx=0, record_time: bool=True):
        """
        Main update function - infers the model on the current dataset, optimizes the acquisition function and returns the query location -
        Acquisition function optimization is specified in self.acquisition_function_optimization_type
        """
        if not self.acquisition_function.support_gradient_based_optimization:
            logger.warning("Acquisition function does not support gradient based optimization, using default update instead.")
            return self.update(idx=idx)

        # access variable from pool
        x_pool = self.pool.possible_queries()
        idx_dim = self._return_variable_idx()
        D = len(idx_dim)

        # get objective & constraints
        obj, const = self.acquisition_function.get_gradient_based_objectives(
            model = self.model,
            safety_models = self.safety_models,
            x_data = self.x_data[:, idx_dim],
            y_data = self.y_data,
        )

        # get initial point & bounds
        border_right = x_pool[:, idx_dim].max(axis=0)
        border_left = x_pool[:, idx_dim].min(axis=0)
        bounds = Bounds(border_left, border_right)
        center = 0.5 * (border_right + border_left)

        # collect optimizer configs
        optimizer_args = {
            "fun": obj if obj is not None else ( lambda x: 0.0 ),
            "x0": center if obj is not None else x_pool[np.random.choice(x_pool.shape[0]), idx_dim],
            "jac": ( lambda x: np.zeros(D) ) if obj is None else None,
            "hess": ( lambda x: np.zeros((D, D)) ) if obj is None else None,
            "method": "L-BFGS-B",
            "bounds": bounds,
        }
        if const is not None:
            optimizer_args.update({"method": "trust-constr", "constraints": [const]})

        # optimize acquisition function
        t0 = time.perf_counter()

        opt_res = minimize(**optimizer_args)
        new_query = opt_res.x

        t1 = time.perf_counter()

        if record_time:
            self.acq_func_opt_time[idx] = t1 - t0

        if self.do_plotting:
            mu, std = self._update_posterior(x_pool)
            acq_score, S = self.acquisition_function.acquisition_score(
                x_pool[:, idx_dim],
                model = self.model,
                safety_models = self.safety_models,
                x_data = self.x_data[:, idx_dim],
                y_data = self.y_data,
                return_safe_set=True
            )
            if not np.any(S):
                S = np.ones_like(S, dtype=S.dtype)

            self.plotting_booklet = {
                'x_pool': x_pool,
                'safety_mask': S.astype(int),
                'acq_score': acq_score,
                'posterior': (mu, std)
            }
        return new_query

    def update(self, idx=0, record_time: bool=True):
        """
        Main update function - infers the model on the current dataset, optimizes the acquisition function and returns the query location -
        Acquisition function optimization is specified in self.acquisition_function_optimization_type
        """
        t0 = time.perf_counter()
        x_pool = self.pool.possible_queries()
        idx_dim = self._return_variable_idx()

        if not self._limit_pool_size_per_update:
            acq_score, S = self.acquisition_function.acquisition_score(
                x_pool[:, idx_dim],
                model = self.model,
                safety_models = self.safety_models,
                x_data = self.x_data[:, idx_dim],
                y_data = self.y_data,
                return_safe_set=True
            )
        else:
            N_max_iter = self._max_pool_size_per_update
            N_pool = x_pool.shape[0]
            acq_score_list = []
            S_list = []
            for i in range(0, N_pool, N_max_iter):
                x_pool_i = x_pool[i:i+N_max_iter]
                acq_score_i, S_i = self.acquisition_function.acquisition_score(
                    x_pool_i[:, idx_dim],
                    model = self.model,
                    safety_models = self.safety_models,
                    x_data = self.x_data[:, idx_dim],
                    y_data = self.y_data,
                    return_safe_set=True
                )
                acq_score_list.append(acq_score_i)
                S_list.append(S_i)
            acq_score = np.concatenate(acq_score_list)
            S = np.concatenate(S_list)

        t1 = time.perf_counter()

        if record_time:
            self.acq_func_opt_time[idx] = t1 - t0

        if not np.any(S):
            #raise StopIteration("There are no safe points to evaluate.")
            logger.warning("There are no safe points, evaluate the entire pool.")
            S = np.ones_like(S, dtype=S.dtype)

        x_safe = x_pool[S]
        new_query = x_safe[np.argmax(acq_score[S])]

        if self.do_plotting:
            mu, std = self._update_posterior(x_pool)
            self.plotting_booklet = {
                'x_pool': x_pool,
                'safety_mask': S.astype(int),
                'acq_score': acq_score,
                'posterior': (mu, std)
            }
        return new_query

    def learn(self, n_steps: int):
        """
        Main maximization loop - makes n_steps queries to oracle and returns collected validation metrics and query locations

        Arguments:
        n_steps : int - number of BO steps

        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
            int - true step we finish
        """        
        if self.constraint_on_y:
            assert self.acquisition_function.number_of_constraints == 1
        else:
            assert self.acquisition_function.number_of_constraints == len(self.safety_models)

        # warmup
        self._make_infer(idx=-1, train_with_safe_data_only=self.train_with_safe_data_only, record_time = False)
        query = self.gradient_based_update(idx=-1, record_time = False) if self.update_by_gradient else self.update(idx=-1, record_time = False)

        true_steps = 0
        for i in range(0, n_steps):
            validate_this_iter = self.validation_at is None or len(self.validation_at)==0 or i in self.validation_at
            try:
                max_iter = 100
                for j in range(max_iter):
                    print(f"Iter {i}: Query")
                    self._make_infer(idx=i, train_with_safe_data_only=self.train_with_safe_data_only)
                    query = self.gradient_based_update(idx=i) if self.update_by_gradient else self.update(idx=i)
                    print(query)
                    if self.acquisition_function.require_fitted_model and self.acquisition_function.require_fitted_safety_models:
                        self.query_time[i] = sum(self.infer_time[i]) + self.acq_func_opt_time[i]
                    elif self.acquisition_function.require_fitted_model:
                        self.query_time[i] = self.infer_time[i][0] + self.acq_func_opt_time[i]
                    elif self.acquisition_function.require_fitted_safety_models:
                        if not self.constraint_on_y:
                            self.query_time[i] = sum(self.infer_time[i][1:]) + self.acq_func_opt_time[i]
                        else:
                            self.query_time[i] = self.infer_time[i][0] + self.acq_func_opt_time[i]
                    else:
                        self.query_time[i] = self.acq_func_opt_time[i]

                    if self.constraint_on_y:
                        new_y = self.pool.query(query, noisy=True)
                        if np.isnan(new_y) and j < max_iter:
                            print('Qeury is nan, repeat')
                            continue
                        self.add_train_point(i, query, new_y)

                        z_for_validate = np.array([new_y])

                        if validate_this_iter:
                            self.validate(z_for_validate, idx=i)
                        else:
                            self.empty_validate(z_for_validate, idx=i)
                        break
                    else:
                        new_y, new_z = self.pool.query(query, noisy=True)
                        nan_check = np.isnan(
                            np.concatenate([np.reshape(new_y, -1), np.reshape(new_z, -1)])
                        )
                        if np.all(nan_check) and j < max_iter:
                            print('Qeury is nan, repeat')
                            continue
                        elif np.any(nan_check) and j < max_iter:
                            print('Qeurying output has some nan values, be careful')
                        self.add_train_point(i, query, new_y, new_z)
                        if validate_this_iter:
                            self.validate(new_z, idx=i)
                        else:
                            self.empty_validate(new_z, idx=i)
                        break

                true_steps += 1
            
            except StopIteration as e:
                print(f'Finish early: {e}')
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt

            if self.do_plotting and validate_this_iter:
                try:
                    self.plot(query, i)
                except:
                    pass

        if self.save_result:
            self.save_experiment_summary()
        return np.array(self.validation_metrics), self.x_data, true_steps

    def empty_validate(self, z, idx: int):
        metrics = super().compute_empty_validation_on_y()
        safe_bool = int(self.acquisition_function.compute_safe_data_set(z.reshape(1, -1))[0])
        unsafe_distance = []
        for j, zj in enumerate(z):
            thresh_l = self.acquisition_function.safety_thresholds_lower[j]
            thresh_u = self.acquisition_function.safety_thresholds_upper[j]
            if zj < thresh_l:
                unsafe_distance.append(thresh_l - zj)
            elif zj > thresh_u:
                unsafe_distance.append(zj - thresh_u)
            else:
                unsafe_distance.append(0.0)
        unsafe_distance = np.mean( unsafe_distance )

        self.add_validation_value(
            idx, [
                *metrics,
                safe_bool,
                unsafe_distance,
                *self.infer_time[idx],
                self.acq_func_opt_time[idx],
                self.query_time[idx],
                0.0
            ])

    def validate(self, z, idx: int=0):
        """
        validation method - calculates validation metric (self.validation_type specifies which one) and stores it to self.validation_metrics list

        Arguments:
            z: [q,] shape array, safety measurement(s) of the query
            idx: index for result dataframe, e.g. AL iteration index
        """
        idx_dim = self._return_variable_idx()

        print('Validate')

        t_start = time.perf_counter()
        metrics = super().compute_validation_on_y()
        t_end = time.perf_counter()
        validate_time = t_end - t_start

        safe_bool = int(self.acquisition_function.compute_safe_data_set(z.reshape(1, -1))[0])
        unsafe_distance = []
        for j, zj in enumerate(z):
            thresh_l = self.acquisition_function.safety_thresholds_lower[j]
            thresh_u = self.acquisition_function.safety_thresholds_upper[j]
            if zj < thresh_l:
                unsafe_distance.append(thresh_l - zj)
            elif zj > thresh_u:
                unsafe_distance.append(zj - thresh_u)
            else:
                unsafe_distance.append(0.0)
        unsafe_distance = np.mean( unsafe_distance )

        self.add_validation_value(
            idx, [
                *metrics,
                safe_bool,
                unsafe_distance,
                *self.infer_time[idx],
                self.acq_func_opt_time[idx],
                self.query_time[idx],
                validate_time
            ])

    def plot(self, query, step: int):
        print('Plot')
        if self.ground_truth_available:
            raise NotImplementedError("unfinished")
        x_pool = self.plotting_booklet['x_pool']
        S_mask = self.plotting_booklet['safety_mask']
        acq_score = self.plotting_booklet['acq_score']
        mu, std = self.plotting_booklet['posterior']
        
        var_dim = self.pool.get_variable_dimension()
        idx_dim = self._return_variable_idx()

        # len(*) - 1 because we validate once before running the experiment, which means len(*) = num_iter + 1
        if var_dim == 1:
            if self.ground_truth_available:
                raise NotImplementedError("unfinished")
            else:
                z_data = self.z_data if not self.constraint_on_y else self.y_data
                safe_bayesian_optimization_1d_plot(
                    self.pool.output_type, x_pool[..., idx_dim], acq_score,
                    mu, 2*std,
                    self.acquisition_function.safety_thresholds_lower,
                    self.acquisition_function.safety_thresholds_upper,
                    S_mask, self.x_data[...,idx_dim], self.y_data, z_data, query[...,idx_dim],
                    save_plot=self.save_plots, file_name="query_" + str(step) + ".png", file_path=self.plot_path
                )
        elif var_dim == 2:
            if self.ground_truth_available:
                raise NotImplementedError("unfinished")
                safe_bayesian_optimization_2d_plot("use this function maybe")
            else:
                pred_mu, _ = self.model.predictive_dist(x_pool[:, self._return_variable_idx()])
                safe_bayesian_optimization_2d_plot_without_gt_with_only_safepoints(
                    self.pool.output_type, x_pool[...,idx_dim], acq_score[S_mask.astype(bool)],
                    pred_mu, S_mask, self.x_data[...,idx_dim], self.y_data, query[...,idx_dim], save_plot=self.save_plots, file_name="query_" + str(step) + ".png", file_path=self.plot_path
                )
            z_data = self.z_data if not self.constraint_on_y else self.y_data
            if self.constraint_on_y:
                safety_function_2d_plot(self.pool.output_type, x_pool[..., idx_dim],
                    mu, 2*std,
                    self.acquisition_function.safety_thresholds_lower,
                    self.acquisition_function.safety_thresholds_upper,
                    self.x_data[:-1, idx_dim], z_data[:-1],
                    save_plot=self.save_plots, file_name="query_" + str(step) + "Z.png", file_path=self.plot_path)
            if not self.constraint_on_y:
                safety_function_2d_plot(self.pool.output_type, x_pool[..., idx_dim],
                mu[:, 1:], 2*std[:, 1:],
                self.acquisition_function.safety_thresholds_lower,
                self.acquisition_function.safety_thresholds_upper,
                self.x_data[:-1, idx_dim], z_data[:-1],
                save_plot=self.save_plots, file_name="query_" + str(step) + "Z.png", file_path=self.plot_path)
        else:
            print("Dimension too high for plotting", file=sys.stderr)

    def _make_infer(self, idx: int, train_with_safe_data_only: List[bool], record_time: bool=True):
        idx_dim = self._return_variable_idx()
        infer_time = []

        if any(train_with_safe_data_only):
            safety_observations = self.y_data if self.constraint_on_y else self.z_data
            safe_mask = self.acquisition_function.compute_safe_data_set(safety_observations)

        time_before_inference = time.perf_counter()
        self.model.reset_model()
        if not train_with_safe_data_only[0] or (~safe_mask).sum()==0:
            self.model.infer(*filter_nan(self.x_data[..., idx_dim], self.y_data))
        else:
            self.model.infer(*filter_nan(self.x_data[..., idx_dim][safe_mask], self.y_data[safe_mask]))
            y_pseudo, _ = self.model.predictive_dist(self.x_data[..., idx_dim][~safe_mask])
            self.model.set_model_data(*filter_nan(
                np.vstack((self.x_data[..., idx_dim][safe_mask], self.x_data[..., idx_dim][~safe_mask])),
                np.vstack((self.y_data[safe_mask], y_pseudo[..., None]))
            ))
        time_after_inference = time.perf_counter()
        t_inf = self.model.get_last_inference_time() if hasattr(self.model, 'get_last_inference_time') else time_after_inference - time_before_inference
        infer_time.append(t_inf)

        if not self.constraint_on_y:
            for i, model in enumerate(self.safety_models):
                time_before_inference = time.perf_counter()
                model.reset_model()
                if not train_with_safe_data_only[i+1] or (~safe_mask).sum()==0:
                    model.infer(*filter_nan(self.x_data[..., idx_dim], self.z_data[..., i, None]))
                else:
                    if hasattr(model, 'classification') and model.classification:
                        labels = self.acquisition_function.compute_safe_data_set_per_constraint(
                            self.z_data[..., i, None], i
                        ).reshape([-1,1]).astype(self.z_data.dtype)
                        labels = np.where(safe_mask[..., None], self.z_data[..., i, None], labels )
                        model.infer(*filter_nan(self.x_data[..., idx_dim], labels))
                    else:
                        model.infer(*filter_nan(self.x_data[..., idx_dim][safe_mask], self.z_data[safe_mask, i, None]))
                        zi_pseudo, _ = model.predictive_dist(self.x_data[..., idx_dim][~safe_mask])
                        model.set_model_data(*filter_nan(
                            np.vstack((self.x_data[..., idx_dim][safe_mask], self.x_data[..., idx_dim][~safe_mask])),
                            np.vstack((self.z_data[safe_mask, i, None], zi_pseudo[..., None]))
                        ))
                time_after_inference = time.perf_counter()
                t_inf = model.get_last_inference_time() if hasattr(model, 'get_last_inference_time') else time_after_inference - time_before_inference
                infer_time.append(t_inf)

        if record_time:
            self.infer_time[idx] = infer_time
        return True

    def _update_posterior(self, x: np.ndarray):
        mu = np.empty((x.shape[0], self.num_of_models), dtype=float)
        std = np.empty((x.shape[0], self.num_of_models), dtype=float)

        idx_dim = self._return_variable_idx()

        mu[:, 0], std[:, 0] = self.model.predictive_dist(x[:, idx_dim])
        for i in range(1, self.num_of_models):  # if self.model_is_safety_model, self.num_of_models will be 1 anyways
            mu[:, i], std[:, i] = self.safety_models[i - 1].predictive_dist(x[:, idx_dim])
        return mu, std


class PoolSafeActiveLearnerWithCCLMetric(PoolSafeActiveLearner):

    r"""
    pool safe active learning which additionally take CCL to cluster explored disconnected safe regions

    :param acquisition_function: BaseSafeAcquisitionFunction object - compute the constrained acquisition scores
    :param validation_type : ValidationType - Enum which validation metric should be used e.g. Simple Regret, Cumm. Regret,...
    :param constraint_on_y: bool - whether the safety is constrained directly on the main model or not
    :param validation_at
    """

    def __init__(
        self,
        acquisition_function: BaseSafeAcquisitionFunction,
        validation_type: ValidationType,
        query_noisy: bool=True,
        constraint_on_y: bool=False,
        run_ccl: bool=True,
        tolerance: Union[float, Sequence[float]]=0.01,
        validation_at: Optional[List[int]] = None,
        train_with_safe_data_only: List[bool]=[False],
        update_gp_hps_every_n_iter: int = 1,
        **kwargs
    ):
        super().__init__(
            acquisition_function,
            validation_type,
            validation_at=validation_at,
            constraint_on_y=constraint_on_y,
            train_with_safe_data_only=train_with_safe_data_only,
        )
        self.observation_number = []
        self.validation_metrics = []
        if not self.constraint_on_y:
            self.validation_metrics_safety_models = []
        self.infer_time = []
        self.validate_time = []
        self.kernel_scale = []
        self.kernel_lengthscales = []
        self.measure_safe_area = False
        self.safe_area = SafetyAreaMeasure(run_ccl=run_ccl)
        """SafetyAreaMeasure compute the areas of predictive safe regions and true safe regions,
                where each regions would be clustered."""
        self.query_noisy = query_noisy
        self.tolerance = tolerance
        self._update_gp_hps_in_this_iter = True
        self._update_gp_hps_every_n_iter = update_gp_hps_every_n_iter
        self.__save_model_pars = False

    def initialize_safe_area_measure(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        z_grid: np.ndarray=None,
        label_grid: str=None,
        sheet_name: str=None
    ):
        if self.constraint_on_y:
            self.set_safety_test_set(x_grid, y_grid)
        else:
            self.set_safety_test_set(x_grid, z_grid)
        self.measure_safe_area = True
        
        d = self.pool.get_dimension()
        self.safe_area.set_object_detector(self.pool.get_dimension())

        if not label_grid is None:
            self.safe_area.true_safe_lands_from_file(label_grid, sheet_name)
        else:
            safe_bool = self.acquisition_function.compute_safe_data_set(self.z_test).reshape([-1,1])
            self.safe_area.true_safe_lands(self.x_test_z[..., :d], safe_bool.astype(int))

    def update(self, converge_check: bool):
        """
        Main update function - infers the model on the current dataset, optimizes the acquisition function and returns the query location -
        Acquisition function optimization is specified in self.acquisition_function_optimization_type
        """
        if self.constraint_on_y:
            assert self.acquisition_function.number_of_constraints == 1
        else:
            assert self.acquisition_function.number_of_constraints == len(self.safety_models)
        self._make_infer(self.train_with_safe_data_only)

        x_pool = self.pool.possible_queries()
        mask = np.random.choice(x_pool.shape[0], size=min(10000, x_pool.shape[0]), replace=False)
        x_pool = x_pool[mask]
        idx_dim = self._return_variable_idx()

        acq_score, S = self.acquisition_function.acquisition_score(
            x_pool[:, idx_dim],
            model = self.model,
            safety_models = self.safety_models,
            x_data = self.x_data[:, idx_dim],
            y_data = self.y_data,
            return_safe_set=True
        )

        if not np.any(S):
            #raise StopIteration("There are no safe points to evaluate.")
            logger.warning("There are no safe points, evaluate the entire pool.")
            S = np.ones_like(S, dtype=S.dtype)
        converge = 0#np.all( std[S] <= tolerance)
        if converge_check and converge:
            raise StopIteration("Converge.")
        
        x_safe = x_pool[S]
        new_query = x_safe[np.argmax(acq_score[S])]

        if self.do_plotting:
            mu, std = self._update_posterior(x_pool)
            self.plotting_booklet = {
                'x_pool': x_pool,
                'safety_mask': S.astype(int),
                'acq_score': acq_score,
                'posterior': (mu, std)
            }
        return new_query

    def learn(self, n_steps: int):
        """
        Main maximization loop - makes n_steps queries to oracle and returns collected validation metrics and query locations

        Arguments:
        n_steps : int - number of BO steps

        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
            int - true step we finish
        """
        self.validate(make_infer=True, idx=-1)
        true_steps = 0
        for i in range(0, n_steps):
            try:
                max_iter = 100
                for j in range(max_iter):
                    query = self.update(converge_check=(i>=4))
                    print(f"Iter {i}: Query")
                    print(query)

                    if self.constraint_on_y:
                        new_y = self.pool.query(query, noisy=self.query_noisy)
                        if np.isnan(new_y) and j < max_iter:
                            print('Qeury is nan, repeat')
                            continue
                        self.add_train_point(i, query, new_y)
                        break
                    else:
                        new_y, new_z = self.pool.query(query, noisy=self.query_noisy)
                        nan_check = np.isnan(
                            np.concatenate([np.reshape(new_y, -1), np.reshape(new_z, -1)])
                        )
                        if np.all(nan_check) and j < max_iter:
                            print('Qeury is nan, repeat')
                            continue
                        elif np.any(nan_check) and j < max_iter:
                            print('Qeurying output has some nan values, be careful')
                        self.add_train_point(i, query, new_y, new_z)
                        break

                self.validate(make_infer=False, idx=i)
                true_steps += 1
                self._update_gp_hps_in_this_iter = (true_steps % self._update_gp_hps_every_n_iter == 0 )

            except StopIteration as e:
                print(f'Finish early: {e}')
                break
            except KeyboardInterrupt:
                raise KeyboardInterrupt

            if self.do_plotting:
                try:
                    self.plot(query, i)
                except:
                    pass
                
        if self.save_result:
            self.save_experiment_summary()
        return np.array(self.validation_metrics), self.x_data, true_steps

    def validate(self, make_infer: bool=False, idx: int=0):
        """
        validation method - calculates validation metric (self.validation_type specifies which one) and stores it to self.validation_metrics list

        Arguments:
            query : np.array - selected query
        """
        idx_dim = self._return_variable_idx()
        if make_infer:
            self._make_infer(self.train_with_safe_data_only)

        print('Validate')
        self.observation_number.append(self.x_data.shape[0])

        t_start = time.perf_counter()
        
        if self.validation_type == ValidationType.RMSE:
            pred_mu, pred_sigma = self.model.predictive_dist(self.x_test[:, idx_dim])
            rmse = np.sqrt(np.mean(np.power(pred_mu - np.squeeze(self.y_test), 2.0)))
            self.validation_metrics.append(rmse)
            if not self.constraint_on_y:
                z_rmse = []
                for model in self.safety_models:
                    pred_mu, pred_sigma = model.predictive_dist(self.x_test[:, idx_dim])
                    rmse = np.sqrt(np.mean(np.power(pred_mu - np.squeeze(self.y_test), 2.0)))
                    z_rmse.append(rmse)
                self.validation_metrics_safety_models.append(tuple(z_rmse))
        elif self.validation_type == ValidationType.NEG_LOG_LIKELI:
            pred_mu, pred_sigma = self.model.predictive_dist(self.x_test[:, idx_dim])
            neg_log_likeli = np.mean(-1 * norm.logpdf(np.squeeze(self.y_test), pred_mu, pred_sigma))
            self.validation_metrics.append(neg_log_likeli)
            if not self.constraint_on_y:
                z_neg_log_likeli = []
                for model in self.safety_models:
                    pred_mu, pred_sigma = model.predictive_dist(self.x_test[:, idx_dim])
                    neg_log_likeli = np.mean(-1 * norm.logpdf(np.squeeze(self.y_test), pred_mu, pred_sigma))
                    z_neg_log_likeli.append(neg_log_likeli)
                self.validation_metrics_safety_models.append(tuple(z_neg_log_likeli))
        
        t_end = time.perf_counter()
        self.validate_time.append(t_end - t_start)

        if self.measure_safe_area:
            print('Measure safety quality')
            
            if self.constraint_on_y:
                S = self.acquisition_function.compute_safe_set(
                    self._get_variable_input(self.x_test_z),
                    [self.model]
                )
            else:
                S = self.acquisition_function.compute_safe_set(
                    self._get_variable_input(self.x_test_z),
                    self.safety_models
                )
            self.safe_area.true_positive_lands(S.astype(int))
            self.safe_area.false_positive_lands(S.astype(int))

    def plot(self, query, step: int):
        if self.ground_truth_available:
            raise NotImplementedError("unfinished")
        x_pool = self.plotting_booklet['x_pool']
        S_mask = self.plotting_booklet['safety_mask']
        acq_score = self.plotting_booklet['acq_score']
        mu, std = self.plotting_booklet['posterior']
        
        var_dim = self.pool.get_variable_dimension()
        idx_dim = self._return_variable_idx()

        # len(*) - 1 because we validate once before running the experiment, which means len(*) = num_iter + 1
        if var_dim == 1:
            if self.ground_truth_available:
                raise NotImplementedError("unfinished")
            else:
                z_data = self.z_data if not self.constraint_on_y else self.y_data
                safe_bayesian_optimization_1d_plot(
                    self.pool.output_type, x_pool[..., idx_dim], acq_score,
                    mu, 2*std,
                    self.acquisition_function.safety_thresholds_lower,
                    self.acquisition_function.safety_thresholds_upper,
                    S_mask, self.x_data[...,idx_dim], self.y_data, z_data, query[...,idx_dim],
                    save_plot=self.save_plots, file_name="query_" + str(step) + ".png", file_path=self.plot_path
                )
        elif var_dim == 2:
            if self.ground_truth_available:
                raise NotImplementedError("unfinished")
                safe_bayesian_optimization_2d_plot("use this function maybe")
            else:
                pred_mu, _ = self.model.predictive_dist(x_pool[:, self._return_variable_idx()])
                safe_bayesian_optimization_2d_plot_without_gt_with_only_safepoints(
                    self.pool.output_type, x_pool[...,idx_dim], acq_score[S_mask.astype(bool)],
                    pred_mu, S_mask, self.x_data[...,idx_dim], self.y_data, query[...,idx_dim], save_plot=self.save_plots, file_name="query_" + str(step) + ".png", file_path=self.plot_path
                )
            z_data = self.z_data if not self.constraint_on_y else self.y_data
            if self.constraint_on_y:
                safety_function_2d_plot(self.pool.output_type, x_pool[..., idx_dim],
                    mu, 2*std,
                    self.acquisition_function.safety_thresholds_lower,
                    self.acquisition_function.safety_thresholds_upper,
                    self.x_data[:-1, idx_dim], z_data[:-1],
                    save_plot=self.save_plots, file_name="query_" + str(step) + "Z.png", file_path=self.plot_path)
            if not self.constraint_on_y:
                safety_function_2d_plot(self.pool.output_type, x_pool[..., idx_dim],
                mu[:, 1:], 2*std[:, 1:],
                self.acquisition_function.safety_thresholds_lower,
                self.acquisition_function.safety_thresholds_upper,
                self.x_data[:-1, idx_dim], z_data[:-1],
                save_plot=self.save_plots, file_name="query_" + str(step) + "Z.png", file_path=self.plot_path)
        else:
            print("Dimension too high for plotting", file=sys.stderr)

    def plot_validation_curve(self, x_start_idx: int=None, filename: str=None):
        metric_curve_plotter = MetricCurvePlotter(1)
        if x_start_idx is None:
            x = np.arange(0, len(self.validation_metrics))
        else:
            x = np.arange(x_start_idx, len(self.validation_metrics))
        metric_curve_plotter.add_metrics_curve(x, self.validation_metrics[-len(x):], "blue", self.validation_type.name.lower(), 0, False)
        if self.validation_type == ValidationType.RMSE:
            ylim = [-0.05, max( 0.5, 1.1*max(self.validation_metrics) )]
            metric_curve_plotter.configure_axes(0, self.validation_type.name.lower(), x_label='iteration', y_label='RMSE', log_scale_y=False, add_legend=True, y_lim=ylim)
        else:
            metric_curve_plotter.configure_axes(0, self.validation_type.name.lower(), x_label='iteration', y_label=None, log_scale_y=False, add_legend=True)

        if self.save_result:
            if filename is None:
                filename = 'metric_target.png'
            metric_curve_plotter.save_fig(self.plot_path, filename)
        else:
            metric_curve_plotter.show()

    def plot_safe_dist(self, filename="safety_histogram"):
        z = self.y_data if self.constraint_on_y else self.z_data
        safety_histogram(
            self.pool.output_type,
            self.x_data, z,
            self.acquisition_function.safety_thresholds_lower, self.acquisition_function.safety_thresholds_upper,
            save_plot=self.save_plots, file_name=filename,
            file_path=self.plot_path)

    def plot_safe_area(self, filename="safety_area"):
        self.safe_area.export_plot(save_plot=self.save_plots, file_name=filename, file_path=self.plot_path)

    def save_experiment_summary(self):
        columns = ['iter_idx']
        columns.extend( [f'x{i}' for i in range(self.x_data.shape[1])] )
        columns.extend( ['y', 'safe_bool', self.validation_type.name.lower()] )
        if not self.constraint_on_y:
            columns.extend( [f'z{i}_'+self.validation_type.name.lower() for i in range(self.acquisition_function.number_of_constraints)] )
        columns.extend( [f'infer_time_m{i}' for i in range(self.num_of_models)])
        columns.append('validate_time')
        
        N = self.x_data.shape[0]
        iter_idx = np.empty([N, 1]) * np.nan
        metric = np.empty([N, 1]) * np.nan
        if not self.constraint_on_y:
            z_metric = np.empty([N, self.acquisition_function.number_of_constraints]) * np.nan
        infer_time = np.empty([N, self.num_of_models]) * np.nan
        validate_time = np.empty([N, 1]) * np.nan
        for i, n in enumerate(self.observation_number):
            iter_idx[n-1] = i
            metric[n-1] = self.validation_metrics[i]
            if not self.constraint_on_y:
                z_metric[n-1,:] = self.validation_metrics_safety_models[i]
            infer_time[n-1,:] = self.infer_time[i]
            validate_time[n-1] = self.validate_time[i]

        safety_observations = self.y_data if self.constraint_on_y else self.z_data
        safe_bool = self.acquisition_function.compute_safe_data_set(safety_observations).reshape([-1,1])

        if self.measure_safe_area:
            D = self.pool.get_variable_dimension()
            safe_area_label = np.empty([self.x_data.shape[0], 1])

            if self.pool.output_type == OutputType.SINGLE_OUTPUT:
                safe_area_label[:,0] = self.safe_area.label_points(self.x_data[:,:D])
            elif self.pool.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
                safe_area_label[self.x_data[:,-1]==self.pool.task_index,0] = self.safe_area.label_points(self.x_data[self.x_data[:,-1]==self.pool.task_index,:D])
            if self.constraint_on_y:
                data = np.hstack((iter_idx, self.x_data, self.y_data, safe_bool, metric, infer_time, validate_time, safe_area_label))
            else:
                data = np.hstack((iter_idx, self.x_data, self.y_data, safe_bool, metric, z_metric, infer_time, validate_time, safe_area_label))
            columns.append('safe_label_data')
        else:
            data = np.hstack((iter_idx, self.x_data, self.y_data, safe_bool, metric, infer_time, validate_time))

        dataframe = pd.DataFrame(data, columns=columns)
        if self.measure_safe_area:
            n = self.safe_area.get_total_iter_num_true_positive()
            safe_df = self.safe_area.export_df( dataframe[dataframe['iter_idx']>=iter_idx[-n,0]].index )
            dataframe = pd.concat([dataframe, safe_df], axis=1)

        # save model k0
        if self.__save_model_pars:
            k_scale = pd.DataFrame(
                columns=[f'model{i}' for i in self.kernel_scale[0].keys()],
                index = iter_idx[~np.isnan(iter_idx[:,0]),0]
            )
            for i, k0_dict in enumerate(self.kernel_scale):
                for key, v in k0_dict.items():
                    k_scale.loc[i, f'model{key}'] = v
        
        # then save models lengthscale
        if self.__save_model_pars:
            k_lengs = pd.DataFrame(
                columns=pd.MultiIndex.from_tuples(self.kernel_lengthscales[0].keys(), names=['model_idx','kernel_idx']),
                index = iter_idx[~np.isnan(iter_idx[:,0]),0]
            )
            for i, kl_dict in enumerate(self.kernel_lengthscales):
                for key, v in kl_dict.items():
                    k_lengs.loc[i, key] = v

        if self.save_result:
            with pd.ExcelWriter(
                os.path.join(self.summary_path, self.summary_filename),
                mode = 'w'
            ) as writer:
                dataframe.to_excel(writer)
            if self.__save_model_pars:
                with pd.ExcelWriter(
                    os.path.join(self.summary_path, self.summary_filename.split('.')[0]+'_model_parameters.xlsx'),
                    mode='w'
                ) as writer:
                    k_scale.to_excel(writer, sheet_name='kernel_scale')
                    k_lengs.to_excel(writer, sheet_name='kernel_lgths')
        else:
            if self.__save_model_pars:
                return dataframe, k_scale, k_lengs
            else:
                return dataframe

    def _make_no_train_infer(self):
        print('    No gp update in this iter')
        idx_dim = self._return_variable_idx()
        infer_time = []

        self.model.set_model_data(*filter_nan(self.x_data[..., idx_dim], self.y_data))        
        infer_time.append(0.0)
        if not self.constraint_on_y:
            for i, model in enumerate(self.safety_models):
                model.set_model_data(*filter_nan(self.x_data[..., idx_dim], self.z_data[..., i, None]))
                infer_time.append(0.0)

        self.infer_time.append(tuple(infer_time))

        if self.__save_model_pars:
            self._track_model_parameters()

        return True


    def _make_infer(self, train_with_safe_data_only: List[bool]):
        if not self._update_gp_hps_in_this_iter:
            return self._make_no_train_infer()

        print('    Update GPs')
        self.model.reset_model()
        idx_dim = self._return_variable_idx()
        infer_time = []

        if any(train_with_safe_data_only):
            safety_observations = self.y_data if self.constraint_on_y else self.z_data
            safe_mask = self.acquisition_function.compute_safe_data_set(safety_observations)

        if not train_with_safe_data_only[0] or (~safe_mask).sum()==0:
            self.model.infer(*filter_nan(self.x_data[..., idx_dim], self.y_data))
        else:
            self.model.infer(*filter_nan(self.x_data[..., idx_dim][safe_mask], self.y_data[safe_mask]))
            y_pseudo, _ = self.model.predictive_dist(self.x_data[..., idx_dim][~safe_mask])
            self.model.set_model_data(*filter_nan(
                np.vstack((self.x_data[..., idx_dim][safe_mask], self.x_data[..., idx_dim][~safe_mask])),
                np.vstack((self.y_data[safe_mask], y_pseudo[..., None]))
            ))
        
        infer_time.append(self.model.get_last_inference_time())
        if not self.constraint_on_y:
            for i, model in enumerate(self.safety_models):
                model.reset_model()
                if not train_with_safe_data_only[i+1] or (~safe_mask).sum()==0:
                    model.infer(*filter_nan(self.x_data[..., idx_dim], self.z_data[..., i, None]))
                else:
                    if hasattr(model, 'classification') and model.classification:
                        print('success!!')
                        labels = self.acquisition_function.compute_safe_data_set_per_constraint(
                            self.z_data[..., i, None], i
                        ).reshape([-1,1]).astype(self.z_data.dtype)
                        labels = np.where(safe_mask[..., None], self.z_data[..., i, None], labels )
                        model.infer(*filter_nan(self.x_data[..., idx_dim], labels))
                    else:
                        model.infer(*filter_nan(self.x_data[..., idx_dim][safe_mask], self.z_data[safe_mask, i, None]))
                        zi_pseudo, _ = model.predictive_dist(self.x_data[..., idx_dim][~safe_mask])
                        model.set_model_data(*filter_nan(
                            np.vstack((self.x_data[..., idx_dim][safe_mask], self.x_data[..., idx_dim][~safe_mask])),
                            np.vstack((self.z_data[safe_mask, i, None], zi_pseudo[..., None]))
                        ))
                infer_time.append(model.get_last_inference_time())

        self.infer_time.append(tuple(infer_time))

        if self.__save_model_pars:
            self._track_model_parameters()

        return True

    def _track_model_parameters(self):
        k0 = {0: self.model.model.kernel.prior_scale}
        if not self.constraint_on_y:
            for j, sm in enumerate(self.safety_models):
                k0[j] = sm.model.kernel.prior_scale
        
        self.kernel_scale.append(k0)

        models = [self.model]
        if not self.constraint_on_y:
            models.extend(self.safety_models)

        if self.pool.output_type == OutputType.SINGLE_OUTPUT:
            self.kernel_lengthscales.append({(i, 0): tuple(m.model.kernel.kernel.lengthscales.numpy(),) for i, m in enumerate(models)})
        elif self.pool.output_type == OutputType.MULTI_OUTPUT_FLATTENED:
            kl_dict = {}
            for i, m in enumerate(models):
                for j, k in enumerate(m.model.kernel.latent_kernels):
                    kl_dict[(i, j)] = tuple(k.lengthscales.numpy())
            self.kernel_lengthscales.append(kl_dict)


if __name__ == "__main__":
    pass
