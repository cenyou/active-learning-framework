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
from typing import Tuple, List, Union, Optional
from alef.active_learners.amortized_policies.amortized_policy_factory import AmortizedPolicyFactory
from alef.configs.active_learners.amortized_policies.policy_configs import ContinuousGPPolicyConfig
from alef.enums.active_learner_enums import ValidationType
from alef.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot

from .base_active_learners import BaseOracleActiveLearner

class OraclePolicyActiveLearner(BaseOracleActiveLearner):

    """
    Main class for non-batch oracle-based active learning - one query at a time - collects queries by calling its oracle object

    Main Attributes:
        validation_type : Union[ValidationType, List[ValidationType]] - Enum which validation metric should be used e.g. RMSE, NegLoglikeli
        acquisiton_optimization_type : OracleALAcquisitionOptimizationType - specifies how the acquisition function should be optimized
    """

    def __init__(
        self,
        validation_type: Union[ValidationType, List[ValidationType]],
        policy_path: str='',
        validation_at: Optional[List[int]] = None,
        *,
        pytest: bool = False, # use this only to do pytest
        policy_dimension: int = 2, # specify only to do pytest
        **kwargs
    ):
        super().__init__()
        self.validation_type = validation_type
        self.validation_at = validation_at
        if isinstance(self.validation_type, ValidationType):
            assert not self.validation_type == ValidationType.RMSE_MULTIOUTPUT, NotImplementedError
            _metric_names = [self.validation_type.name]
        else:
            assert not any([v == ValidationType.RMSE_MULTIOUTPUT for v in self.validation_type]), NotImplementedError
            _metric_names = [v.name for v in self.validation_type]
        self.initialize_validation_metrics(metric_name_list=[*_metric_names, 'inference_time', 'query_time', 'validate_time'])
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
        policy_config = ContinuousGPPolicyConfig(input_dim=D)
        self.policy = AmortizedPolicyFactory.build(policy_config)
        self.policy.eval()

    def run_policy(self, x_data: np.ndarray, y_data: np.ndarray, remaining_budget: int=None):
        X = torch.from_numpy(x_data).to(torch.get_default_dtype()) # [N, D]
        Y = torch.from_numpy(y_data[..., 0]).to(torch.get_default_dtype()) # [N]

        if self.policy.forward_with_budget:
            assert not remaining_budget is None
            T = torch.tensor([remaining_budget], dtype=torch.get_default_dtype())
            query = self.policy(T, X, Y).cpu().detach().numpy()
        else:
            query = self.policy(X, Y).cpu().detach().numpy()

        return query[0, :] # [D]

    def update_gp_model(self):
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)

    def update(self, remaining_budget: int=None):
        assert np.allclose(self.policy.input_domain, self.oracle.get_box_bounds())
        return self.run_policy(self.x_data, self.y_data, remaining_budget)

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
        if self.do_plotting and self.oracle.get_dimension() <=2:
            self.sample_ground_truth()
        self.n_steps = sum(n_steps) if hasattr(n_steps, '__len__') else n_steps
        n_steps_iter = iter(n_steps) if hasattr(n_steps, '__len__') else iter([n_steps])
        budget = next(n_steps_iter)
        for i in range(0, self.n_steps):
            need_gp = False
            validate_this_iter = self.validation_at is None or len(self.validation_at)==0 or i in self.validation_at
            #if self.do_plotting and self.oracle.get_dimension() <=2:
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
            new_y = self.oracle.query(query)
            if self.do_plotting and self.oracle.get_dimension() <=2 and validate_this_iter:
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
        self.add_validation_value(idx, [*metrics, self.inference_time[idx], self.query_time[idx], 0.0])

    def validate(self, idx: int):
        """
        Validates the currently infered model on the test set - uses the validation_type to choose which metric should be used
        """
        print('Validate')

        t_start = time.perf_counter()
        metrics = super().compute_validation_on_y()
        t_end = time.perf_counter()
        validate_time = t_end - t_start

        self.add_validation_value(idx, [*metrics, self.inference_time[idx], self.query_time[idx], validate_time])

    def plot(self, query: np.array, new_y: float, step: int):
        dimension = self.oracle.get_dimension()
        x_plot = self.gt_X
        y_plot = self.gt_function_values
        if dimension == 1:
            pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
            plot_name = "query_" + str(step) + ".png"
            active_learning_1d_plot(
                x_plot, pred_mu, pred_sigma, self.x_data, self.y_data, query, new_y, True, x_plot, y_plot,
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
