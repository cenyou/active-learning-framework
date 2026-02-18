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
from scipy.stats import norm
from scipy.optimize import NonlinearConstraint, LinearConstraint
from alef.acquisition_functions.safe_acquisition_functions.base_safe_acquisition_function import StandardAlphaAcquisitionFunction
from alef.models.base_model import BaseModel
from alef.acquisition_functions.safe_acquisition_functions.utils import compute_gp_posterior
from alef.acquisition_functions.safe_acquisition_functions.utils import (
    get_safety_models,
)
from alef.configs.base_parameters import NUMERICAL_POSITIVE_LOWER_BOUND
from alef.utils.utils import normal_entropy

class MinUnsafePredEntropy(StandardAlphaAcquisitionFunction):
    support_gradient_based_optimization: bool=True

    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] =-np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        alpha: float=None,
        lagrange_multiplier: float = 1.0,
        **kwargs
    ):
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, alpha = alpha, **kwargs)
        self._lm = lagrange_multiplier

    @property
    def lagrange_multiplier(self) -> float:
        return self._lm

    @lagrange_multiplier.setter
    def lagrange_multiplier(self, value: float):
        self._lm = value

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
            x_2d = x.reshape(1, -1)
            ent = model.entropy_predictive_dist(x_2d)
            main_score = np.squeeze(ent)
            
            sm_list = get_safety_models(model, safety_models)
            safety_discount = 0
            for i, sm in enumerate( sm_list ):
                p_safe = self.compute_safe_likelihood_per_constraint(x_2d, sm, i)
                p_safe = np.minimum( # clamp at max 1 - alpha
                    np.squeeze(p_safe), 1-self.alpha
                )
                safety_discount -= np.log(# - log unsafe safe prob
                    1 - p_safe + NUMERICAL_POSITIVE_LOWER_BOUND
                )

            return -(main_score + self.lagrange_multiplier * safety_discount)
        
        return objective, None

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
        S = np.ones(x_grid.shape[0], dtype=bool)
        acq_score = np.squeeze(
            model.entropy_predictive_dist(x_grid)
        )

        sm_list = get_safety_models(model, safety_models)
        safety_discount = np.zeros_like(acq_score)
        for i, sm in enumerate( sm_list ):
            p_safe = self.compute_safe_likelihood_per_constraint(x_grid, sm, i)
            # pred_mu, pred_sigma = sm.predictive_dist(x_grid)
            # prob_below_lower = norm.cdf(self.safety_thresholds_lower[i], pred_mu, pred_sigma)
            # prob_below_upper = norm.cdf(self.safety_thresholds_upper[i], pred_mu, pred_sigma)
            # p_safe = prob_below_upper - prob_below_lower
            p_safe = np.minimum( # clamp at max 1 - alpha
                p_safe, (1-self.alpha)*np.ones_like(p_safe)
            )
            safety_discount -= np.log(# - log unsafe safe prob
                np.reshape(1 - p_safe, -1) + NUMERICAL_POSITIVE_LOWER_BOUND
            )
            
        if return_safe_set:
            return acq_score + self.lagrange_multiplier * safety_discount, S
        else:
            return acq_score + self.lagrange_multiplier * safety_discount

class MinUnsafePredEntropyAll(StandardAlphaAcquisitionFunction):
    support_gradient_based_optimization: bool=True

    def __init__(
        self,
        safety_thresholds_lower: Union[float, Sequence[float]] =-np.inf,
        safety_thresholds_upper: Union[float, Sequence[float]] = np.inf,
        alpha: float=None,
        lagrange_multiplier: float = 1.0,
        **kwargs
    ):
        super().__init__(safety_thresholds_lower, safety_thresholds_upper, alpha = alpha, **kwargs)
        self._lm = lagrange_multiplier

    @property
    def lagrange_multiplier(self) -> float:
        return self._lm

    @lagrange_multiplier.setter
    def lagrange_multiplier(self, value: float):
        self._lm = value

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
            x_2d = x.reshape(1, -1)
            _, pred_sigma = compute_gp_posterior(x_2d, model, safety_models)
            entropy = normal_entropy(pred_sigma)
            main_score = np.squeeze( np.sum(entropy, axis=1) )
            
            sm_list = get_safety_models(model, safety_models)
            safety_discount = 0
            for i, sm in enumerate( sm_list ):
                p_safe = self.compute_safe_likelihood_per_constraint(x_2d, sm, i)
                p_safe = np.minimum( # clamp at max 1 - alpha
                    np.squeeze(p_safe), 1-self.alpha
                )
                safety_discount -= np.log(# - log unsafe safe prob
                    1 - p_safe + NUMERICAL_POSITIVE_LOWER_BOUND
                )

            return -(main_score + self.lagrange_multiplier * safety_discount)
        
        return objective, None

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
        S = np.ones(x_grid.shape[0], dtype=bool)
        
        _, pred_sigma = compute_gp_posterior(x_grid, model, safety_models)
        entropy = normal_entropy(pred_sigma)
        acq_score = np.sum(entropy, axis=1)

        sm_list = get_safety_models(model, safety_models)
        safety_discount = np.zeros_like(acq_score)
        for i, sm in enumerate( sm_list ):
            p_safe = self.compute_safe_likelihood_per_constraint(x_grid, sm, i)
            # pred_mu, pred_sigma = sm.predictive_dist(x_grid)
            # prob_below_lower = norm.cdf(self.safety_thresholds_lower[i], pred_mu, pred_sigma)
            # prob_below_upper = norm.cdf(self.safety_thresholds_upper[i], pred_mu, pred_sigma)
            # p_safe = prob_below_upper - prob_below_lower
            p_safe = np.maximum( # clamp at max 1 - alpha
                p_safe, (1-self.alpha)*np.ones_like(p_safe)
            )
            safety_discount -= np.log(# - log unsafe safe prob
                np.reshape(1 - p_safe, -1) + NUMERICAL_POSITIVE_LOWER_BOUND
            )
        if return_safe_set:
            return acq_score + self.lagrange_multiplier * safety_discount, S
        else:
            return acq_score + self.lagrange_multiplier * safety_discount

if __name__ == '__main__':
    pass