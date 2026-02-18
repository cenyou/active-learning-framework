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

from typing import Union, Sequence, Tuple, Optional, Callable
import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint
from alef.acquisition_functions.al_acquisition_functions import pred_sigma
from alef.acquisition_functions.safe_acquisition_functions.base_safe_acquisition_function import StandardAlphaAcquisitionFunction
from alef.models.base_model import BaseModel
from alef.acquisition_functions.safe_acquisition_functions.utils import compute_gp_posterior
from alef.acquisition_functions.safe_acquisition_functions.utils import (
    get_safety_models,
)
from alef.utils.utils import normal_entropy

class SafePredEntropy(StandardAlphaAcquisitionFunction):
    support_gradient_based_optimization: bool=True

    def get_gradient_based_objectives(
        self,
        model: BaseModel,
        *,
        safety_models: Optional[Sequence[BaseModel]] = None,
        x_data: Optional[np.ndarray] = None,
        y_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[Callable, Union[NonlinearConstraint, LinearConstraint, None]]:
        r"""
        Returns the objective function and the constraints for gradient based optimization.
        The objective function should take a single argument x [D, ] and return a scalar value meant to be minimized.
        
        :param model: BaseModel, surrogate model used to calculate acquisition score
        :param safety_models: None or list of BaseModel classes, surrogate models for constraint evaluation
        :param x_data: input locations of the datapoints already observed by the model - can be used to deduce current posterior of maximum (of latent f)
        :param y_data: output values of datapoints already observed by the model - can be used to determine currently observed maximum y value
        :return: objective function, scipy.optimize constraints
        """
        def objective(x: np.ndarray) -> float:
            ent = model.entropy_predictive_dist(x.reshape(1, -1))
            return -np.squeeze(ent)
        
        sm_list = get_safety_models(model, safety_models)
        def constraint_func(x: np.ndarray) -> float:
            p = self.compute_safe_likelihood(x.reshape(1, -1), safety_models=sm_list)
            return p[0]
        
        constraint = NonlinearConstraint(
            constraint_func,
            (1-self.alpha)**self.number_of_constraints,
            np.inf
        )
        return objective, constraint

    def acquisition_score(self,
        x_grid: np.ndarray,
        model: BaseModel,
        *,
        safety_models: Optional[Sequence[BaseModel]] = None,
        return_safe_set: bool = False,
        **kwargs
    ):
        r"""
        x_grid: [N, D] array, location for which acquisiton score should be calcluated
        model: BaseModel, surrogate model used to calculate acquisition score
        safety_models: None or list of BaseModel classes, surrogate models for safety functions
                        if None then one may consider model as the safety model as well
        
        return:
            if return_safe_set is True:
                [N, ] array, acquisition score (later maximize this to get query point)
                [N, ] array, bool, True for safe points and False for unsafe points
            if return_safe_set is False:
                [N, ] array, acquisition score (later maximize this to get query point)
        """
        S = self.compute_safe_set(x_grid, get_safety_models(model, safety_models))
        score = -np.inf * np.ones_like(S, dtype=float)
        
        ent = model.entropy_predictive_dist(x_grid[S])
        score[S] = np.squeeze(ent)
        
        if return_safe_set:
            return score, S
        else:
            return score

class SafePredEntropyAll(StandardAlphaAcquisitionFunction):
    support_gradient_based_optimization: bool=True

    def get_gradient_based_objectives(
        self,
        model: BaseModel,
        *,
        safety_models: Optional[Sequence[BaseModel]] = None,
        x_data: Optional[np.ndarray] = None,
        y_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[Callable, Union[NonlinearConstraint, LinearConstraint, None]]:
        r"""
        Returns the objective function and the constraints for gradient based optimization.
        The objective function should take a single argument x [D, ] and return a scalar value meant to be minimized.
        
        :param model: BaseModel, surrogate model used to calculate acquisition score
        :param safety_models: None or list of BaseModel classes, surrogate models for constraint evaluation
        :param x_data: input locations of the datapoints already observed by the model - can be used to deduce current posterior of maximum (of latent f)
        :param y_data: output values of datapoints already observed by the model - can be used to determine currently observed maximum y value
        :return: objective function, scipy.optimize constraints
        """
        def objective(x: np.ndarray) -> float:
            pred_sigma = compute_gp_posterior(x.reshape(1, -1), model, safety_models)
            entropy = normal_entropy(pred_sigma)
            score = np.sum(entropy, axis=1)
            return -np.squeeze(score)
        
        sm_list = get_safety_models(model, safety_models)
        def constraint_func(x: np.ndarray) -> float:
            p = self.compute_safe_likelihood(x.reshape(1, -1), safety_models=sm_list)
            return p[0]
        
        constraint = NonlinearConstraint(
            constraint_func,
            (1-self.alpha)**self.number_of_constraints,
            np.inf
        )
        return objective, constraint

    def acquisition_score(
        self,
        x_grid: np.ndarray,
        model: BaseModel,
        safety_models: Optional[Sequence[BaseModel]] = None,
        return_safe_set: bool = False,
        **kwargs
    ):
        r"""
        x_grid: [N, D] array, location for which acquisiton score should be calcluated
        model: BaseModel, surrogate model of main function
        safety_models: None or list of BaseModel classes, surrogate models for safety functions
                        if None then one may consider model as the safety model as well
        
        return:
            if return_safe_set is True:
                [N, ] array, acquisition score (later maximize this to get query point)
                [N, ] array, bool, True for safe points and False for unsafe points
            if return_safe_set is False:
                [N, ] array, acquisition score (later maximize this to get query point)
        """
        S = self.compute_safe_set(x_grid, get_safety_models(model, safety_models))
        score = -np.inf * np.ones_like(S, dtype=float)

        _, pred_sigma = compute_gp_posterior(x_grid[S], model, safety_models)
        entropy = normal_entropy(pred_sigma)
        score[S] = np.sum(entropy, axis=1)

        if return_safe_set:
            return score, S
        else:
            return score

if __name__ == '__main__':
    pass