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
import numpy as np
import pandas as pd
import torch
import json
import time
from scipy.spatial.distance import cdist
from typing import Tuple, List, Union, Optional
from alef.active_learners.amortized_policies.amortized_policy_factory import AmortizedPolicyFactory
from alef.active_learners.amortized_policies.training.training_factory import _LossPicker
from alef.configs.active_learners.amortized_policies.loss_configs import SafeGPMI2LossConfig
from alef.configs.active_learners.amortized_policies.policy_configs import SafetyAwareContinuousGPPolicyConfig
from alef.enums.active_learner_enums import ValidationType
from alef.enums.active_learner_amortized_policy_enums import SafetyProbability
from alef.utils.plot_utils import active_learning_1d_plot_with_true_safety_measure, active_learning_2d_plot_with_true_safety_measure

from .base_safe_active_learners import BasePoolSafeActiveLearner

class PoolPolicySafeActiveLearner(BasePoolSafeActiveLearner):

    def __init__(
        self,
        validation_type: Union[ValidationType, List[ValidationType]],
        policy_path: str='',
        constraint_on_y: bool = False,
        validation_at: Optional[List[int]] = None,
        train_with_safe_data_only: List[bool]=[False],
        *,
        pytest: bool = False, # use this only to do pytest
        policy_dimension: int = 2, # specify only to do pytest
        **kwargs
    ):
        if constraint_on_y:
            raise NotImplementedError('The policy was trained to take (x, y, z) data')
        super().__init__()
        self.validation_type = validation_type
        self.validation_at = validation_at
        self.constraint_on_y = constraint_on_y
        assert len(train_with_safe_data_only) == 1, 'we currently use only one model for y evaluation'
        self.train_with_safe_data_only = train_with_safe_data_only
        self.num_of_models = 2 # y and one z
        if isinstance(self.validation_type, ValidationType):
            assert not self.validation_type == ValidationType.RMSE_MULTIOUTPUT, NotImplementedError
            _metric_names = [self.validation_type.name]
        else:
            assert not any([v == ValidationType.RMSE_MULTIOUTPUT for v in self.validation_type]), NotImplementedError
            _metric_names = [v.name for v in self.validation_type]
        self.initialize_validation_metrics(
            metric_name_list=[
                *_metric_names, 'safe_bool', 'unsafe_distance', 'inference_time', 'query_time', 'validate_time'
                ])
        if pytest:
            self.load_test_policy(policy_dimension)
        else:
            self.load_policy(policy_path)
        self.inference_time = {}
        self.query_time = {}

    def load_policy(self, path):
        policy_config = AmortizedPolicyFactory.load_config(path)
        self.policy = AmortizedPolicyFactory.build(policy_config)
        self.policy.eval()

    def load_test_policy(self, policy_dimension: int):
        D = policy_dimension
        policy_config = SafetyAwareContinuousGPPolicyConfig(input_dim=D)
        self.policy = AmortizedPolicyFactory.build(policy_config)
        self.policy.eval()

    def run_policy(self, x_data: np.ndarray, y_data: np.ndarray, z_data: np.ndarray, remaining_budget: int=None):
        X = torch.from_numpy(x_data).to(torch.get_default_dtype()) # [N, D]
        Y = torch.from_numpy(y_data[..., 0]).to(torch.get_default_dtype()) # [N]
        Z = torch.from_numpy(z_data[..., 0]).to(torch.get_default_dtype()) # [N]

        if self.policy.forward_with_budget:
            assert not remaining_budget is None
            T = torch.tensor([remaining_budget], dtype=torch.get_default_dtype())
            query = self.policy(T, X, Y, Z).cpu().detach().numpy() # [1, D]
        else:
            query = self.policy(X, Y, Z).cpu().detach().numpy() # [1, D]

        return query[0, :] # [D]

    def update_gp_model(self):
        self.model.reset_model()
        safety_observations = self.y_data if self.constraint_on_y else self.z_data
        safe_mask = np.all(safety_observations >= 0, axis=1) if self.train_with_safe_data_only[0] else np.ones([self.x_data.shape[0]], dtype=bool)
        # we train the policy under safety_constraint >= 0
        self.model.infer(self.x_data[safe_mask], self.y_data[safe_mask])

    def update_safe_gp_models(self):
        safe_mask = np.all(self.z_data >= 0, axis=1) if self.train_with_safe_data_only[0] else np.ones([self.x_data.shape[0]], dtype=bool)
        for i, m in enumerate(self.safety_models):
            m.reset_model()
            m.infer(self.x_data[safe_mask], self.z_data[..., i, None][safe_mask])

    def update(self, remaining_budget: int=None):
        D = self.pool.get_dimension()
        z = self.y_data if self.constraint_on_y else self.z_data

        query = self.run_policy(self.x_data, self.y_data, z, remaining_budget).reshape([1, D])

        # return the nearest point from the pool
        x_pool = self.pool.possible_queries() # [N, D]
        d = cdist(x_pool, query)# [N, 1]

        return x_pool[np.argmin(d.reshape(-1)), :]

    def learn(self, n_steps: Union[int, List[int]]):
        """
        Arguments:
            n_steps : number of active learning iteration/number of queries
                if a list is given, query n_steps[i] consecutively, 
                this would be different from providing one single sum(n_steps) if the policy takes budget input
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        if self.do_plotting and self.pool.get_dimension() <=2:
            if not self.plot_data_available:
                self.set_plot_data(*self.get_plot_data())
        self.n_steps = sum(n_steps) if hasattr(n_steps, '__len__') else n_steps
        n_steps_iter = iter(n_steps) if hasattr(n_steps, '__len__') else iter([n_steps])
        budget = next(n_steps_iter)
        for i in range(0, self.n_steps):
            need_gp = False
            validate_this_iter = self.validation_at is None or len(self.validation_at)==0 or i in self.validation_at
            #if self.do_plotting and self.pool.get_dimension() <=2:
            #    need_gp = True
            #elif validate_this_iter:
            #    need_gp = True
            need_gp = validate_this_iter

            t0 = time.perf_counter()
            if need_gp:
                self.update_gp_model()
            t1 = time.perf_counter()
            query = self.update(budget)
            t2 = time.perf_counter()
            budget = budget - 1 if budget > 1 or i == self.n_steps - 1 else next(n_steps_iter)
            self.inference_time[i] = t1 - t0
            self.query_time[i] = t2 - t1

            print(f"Iter {i}: Query")
            print(query)
            if self.constraint_on_y:
                new_y = self.pool.query(query)
                new_z = None
                z_for_validate = np.array([new_y])
            else:
                new_y, new_z = self.pool.query(query)
                z_for_validate = new_z

            if self.do_plotting and self.pool.get_dimension() <=2 and validate_this_iter:
                self.plot(query, new_y, z_for_validate, i)

            self.add_train_point(i, query, new_y, new_z)

            if validate_this_iter:
                self.validate(z_for_validate, i)
            else:
                self.empty_validate(z_for_validate, i) # safety bool and timer can still be recorded

        if self.save_result:
            self.save_experiment_summary()
        return self.validation_metrics, self.x_data

    def empty_validate(self, z, idx: int):
        metrics = super().compute_empty_validation_on_y()
        safe_bool = int(all(z >= 0)) # we train the policy under safety_constraint >= 0
        unsafe_distance = np.mean([-zj if zj < 0 else 0 for zj in z])
        self.add_validation_value(
            idx, [
                *metrics,
                safe_bool,
                unsafe_distance,
                self.inference_time[idx],
                self.query_time[idx],
                0.0
            ])

    def validate(self, z, idx: int):
        """
        Validates the currently infered model on the test set - uses the validation_type to choose which metric should be used
        """
        print('Validate')

        t_start = time.perf_counter()
        metrics = super().compute_validation_on_y()
        t_end = time.perf_counter()
        validate_time = t_end - t_start

        safe_bool = int(all(z >= 0)) # we train the policy under safety_constraint >= 0
        unsafe_distance = np.mean([-zj if zj < 0 else 0 for zj in z])

        self.add_validation_value(
            idx, [
                *metrics,
                safe_bool,
                unsafe_distance,
                self.inference_time[idx],
                self.query_time[idx],
                validate_time
            ])

    def plot(self, query: np.array, new_y: float, new_z: float, n_step: int):
        dimension = self.pool.get_dimension()
        x_plot, y_plot, z_plot = self.get_plot_data() # if constraint_on_y, z_plot is the same as y_plot

        if dimension == 1:
            pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
            plot_name = "query_" + str(n_step) + ".png"
            active_learning_1d_plot_with_true_safety_measure(
                x_plot, pred_mu, pred_sigma,
                self.x_data, self.y_data, self.z_data,
                query, new_y, new_z,
                0.0, np.inf, # safety_thresholds
                True, x_plot, y_plot, z_plot,
                save_plot=self.save_plots,
                file_name=plot_name,
                file_path=self.plot_path,
            )
        elif dimension == 2:
            pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
            acquisition_function = self.model.entropy_predictive_dist(x_plot)
            plot_name = "query_" + str(n_step) + ".png"

            policy_training_config = SafeGPMI2LossConfig()
            if policy_training_config.probability_function == SafetyProbability.GP_POSTERIOR:
                safety_score = None
            else:
                loss, _, _ = _LossPicker.pick_loss(SafeGPMI2LossConfig())
                pz_function = loss.loss_computer.safety_probability
                safety_score = pz_function(
                    torch.from_numpy(z_plot).to(torch.get_default_dtype())
                ).squeeze(-1).cpu().detach().numpy()

            active_learning_2d_plot_with_true_safety_measure(
                x_plot,
                acquisition_function,
                safety_score,
                pred_mu,
                y_plot,
                z_plot,
                (z_plot >= 0).astype(int),
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
