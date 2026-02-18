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
from alef.models.base_model import BaseModel


def sort_generator(array: np.ndarray):
    """Return the sorted array, largest element first."""
    return array.argsort()[::-1]


def get_safety_models(
    model: BaseModel,
    safety_models: Optional[Sequence[BaseModel]] = None
):
    if safety_models is None:
        return [model]
    else:
        return safety_models


def compute_gp_posterior(
    x_grid: np.ndarray,
    model: BaseModel,
    safety_models: Optional[Sequence[BaseModel]] = None
):
    if safety_models is None:
        num_models = 1
        models = [model]
    else:
        num_models = 1 + len(safety_models)
        models = [model] + safety_models
    
    pred_mu = np.zeros([x_grid.shape[0], num_models])
    pred_sigma = np.zeros([x_grid.shape[0], num_models])

    if num_models == 1:
        pred_mu[:, 0], pred_sigma[:, 0] = model.predictive_dist(x_grid)
    else:
        for j, m in enumerate(models):
            pred_mu[:, j], std = m.predictive_dist(x_grid)
            if hasattr(m.model, 'kernel') and hasattr(m.model.kernel, 'prior_scale'):
                pred_sigma[:, j] = std / m.model.kernel.prior_scale
            else:
                print('warning: uncertainty is not normalized, avoid comparing multiple models')
                pred_sigma[:, j] = std
                #raise NotImplementedError
    return pred_mu, pred_sigma


def compute_safe_gp_posterior(
    x_grid: np.ndarray,
    safety_models: Sequence[BaseModel]
):
    num_models = len(safety_models)
    
    pred_mu = np.zeros([x_grid.shape[0], num_models])
    pred_sigma = np.zeros([x_grid.shape[0], num_models])

    for j, m in enumerate(safety_models):
        pred_mu[:, j], std = m.predictive_dist(x_grid)
        if hasattr(m.model.kernel, 'prior_scale'):
            pred_sigma[:, j] = std / m.model.kernel.prior_scale
        else:
            raise NotImplementedError
    return pred_mu, pred_sigma


def compute_confidence_intervals(
    x_grid: np.ndarray,
    beta: float,
    model: Optional[BaseModel] = None,
    safety_models: Optional[Sequence[BaseModel]] = None
):
    assert beta > 0
    sqrt_beta = np.sqrt(beta)
    if model is None:
        assert not safety_models is None
        mu, std = compute_safe_gp_posterior(x_grid, safety_models)
    else:
        mu, std = compute_gp_posterior(x_grid, model, safety_models)
    
    N, D = mu.shape
    Q = np.empty((N, 2 * D), dtype=float)
    Q[:, ::2] = mu - sqrt_beta * std
    Q[:, 1::2] = mu + sqrt_beta * std

    return Q


def compute_safe_probability(
    x_grid: np.ndarray,
    lower_bound: Sequence[float],
    upper_bound: Sequence[float],
    model: BaseModel,
    safety_models: Optional[Sequence[BaseModel]] = None
):
    if safety_models is None:
        mu, std = model.predictive_dist(x_grid)
        mu = np.squeeze(mu)
        std = np.squeeze(std)

        prob_below_lower = norm.cdf( lower_bound[0], mu, std )
        prob_below_upper = norm.cdf( upper_bound[0], mu, std )

        score = (prob_below_upper - prob_below_lower)
    else:
        score = np.ones(x_grid.shape[0], dtype=float)
            
        for i, sm in enumerate(safety_models):
            mu, std = sm.predictive_dist(x_grid)
            mu = np.squeeze(mu)
            std = np.squeeze(std)

            prob_below_lower = norm.cdf( lower_bound[i], mu, std )
            prob_below_upper = norm.cdf( upper_bound[i], mu, std )
            
            score *= (prob_below_upper - prob_below_lower)
    return score