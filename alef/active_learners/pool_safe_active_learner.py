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
from scipy.optimize import minimize, NonlinearConstraint, Bounds, differential_evolution
from alef.utils.utils import filter_nan
from alef.utils.plot_utils import safe_bayesian_optimization_1d_plot, safe_bayesian_optimization_2d_plot, safe_bayesian_optimization_2d_plot_without_gt_with_only_safepoints, safety_function_2d_plot
from alef.enums.data_structure_enums import OutputType
from alef.enums.active_learner_enums import ValidationType
from alef.acquisition_functions.safe_acquisition_functions.base_safe_acquisition_function import BaseSafeAcquisitionFunction
from alef.utils.custom_logging import getLogger

logger = getLogger(__name__)

class PoolSafeActiveLearner(BasePoolSafeActiveLearner, PoolContextualHelper):
    def __init__(
        self,
        acquisition_function: BaseSafeAcquisitionFunction,
        validation_type: Union[ValidationType, List[ValidationType]],
        constraint_on_y: bool=False,
        validation_at: Optional[List[int]] = None,
        train_with_safe_data_only: List[bool]=[False],
        update_by_gradient: bool=False,
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

        acq_score, S = self.acquisition_function.acquisition_score(
            x_pool[:, idx_dim],
            model = self.model,
            safety_models = self.safety_models,
            x_data = self.x_data[:, idx_dim],
            y_data = self.y_data,
            return_safe_set=True
        )

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

    def learn(self, n_steps: int, start_index: int =0):
        """
        Main maximization loop - makes n_steps queries to oracle and returns collected validation metrics and query locations

        Arguments:
        n_steps : int - number of BO steps
        start_index : int - starting index for the iteration count

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
        for i in range(start_index, start_index + n_steps):
            validate_this_iter = self.validation_at is None or len(self.validation_at)==0 or i in self.validation_at
            try:
                max_iter = 100
                for j in range(max_iter):
                    self._make_infer(idx=i, train_with_safe_data_only=self.train_with_safe_data_only)
                    query = self.gradient_based_update(idx=i) if self.update_by_gradient else self.update(idx=i)
                    print(f"Iter {i}: Query")
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

