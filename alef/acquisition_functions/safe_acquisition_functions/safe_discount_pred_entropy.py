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

from typing import Union, Sequence, Optional
import numpy as np
from scipy.stats import norm
from alef.acquisition_functions.safe_acquisition_functions.base_safe_acquisition_function import StandardAlphaAcquisitionFunction
from alef.models.base_model import BaseModel
from alef.acquisition_functions.safe_acquisition_functions.utils import compute_gp_posterior
from alef.acquisition_functions.safe_acquisition_functions.utils import (
    get_safety_models,
)
from alef.configs.base_parameters import NUMERICAL_POSITIVE_LOWER_BOUND

class SafeDiscountPredEntropy(StandardAlphaAcquisitionFunction):
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
        _, pred_sigma = model.predictive_dist(x_grid)
        pred_sigma = np.squeeze(pred_sigma)
        
        acq_score = 0.5 * np.log(
            (2*np.pi*np.e) * pred_sigma**2
        )

        sm_list = get_safety_models(model, safety_models)
        safety_discount = np.zeros_like(acq_score)
        for i, sm in enumerate( sm_list ):
            pred_mu, pred_sigma = sm.predictive_dist(x_grid)
            
            prob_below_lower = norm.cdf(self.safety_thresholds_lower[i], pred_mu, pred_sigma)
            prob_below_upper = norm.cdf(self.safety_thresholds_upper[i], pred_mu, pred_sigma)
            p_safe = prob_below_upper - prob_below_lower
            p_safe = np.minimum( # clamp at max 1 - alpha
                p_safe, (1-self.alpha)*np.ones_like(p_safe)
            )
            safety_discount += np.log(# log safe prob
                np.reshape(p_safe, -1) + NUMERICAL_POSITIVE_LOWER_BOUND
            )
            
        if return_safe_set:
            return acq_score + safety_discount, S
        else:
            return acq_score + safety_discount

class SafeDiscountPredEntropyAll(StandardAlphaAcquisitionFunction):
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
        entropy = 0.5 * np.log((2 * np.pi * np.e) * pred_sigma**2)
        acq_score = np.sum(entropy, axis=1)
        
        sm_list = get_safety_models(model, safety_models)
        safety_discount = np.zeros_like(acq_score)
        for i, sm in enumerate( sm_list ):
            pred_mu, pred_sigma = sm.predictive_dist(x_grid)
            
            prob_below_lower = norm.cdf(self.safety_thresholds_lower[i], pred_mu, pred_sigma)
            prob_below_upper = norm.cdf(self.safety_thresholds_upper[i], pred_mu, pred_sigma)
            p_safe = prob_below_upper - prob_below_lower
            p_safe = np.maximum( # clamp at max 1 - alpha
                p_safe, (1-self.alpha)*np.ones_like(p_safe)
            )
            safety_discount += np.log(# log safe prob
                np.reshape(p_safe, -1) + NUMERICAL_POSITIVE_LOWER_BOUND
            )
        if return_safe_set:
            return acq_score + safety_discount, S
        else:
            return acq_score + safety_discount

if __name__ == '__main__':
    pass