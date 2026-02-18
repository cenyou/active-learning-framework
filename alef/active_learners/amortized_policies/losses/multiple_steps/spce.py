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

import math
import torch
import pyro
from pyro.util import warn_if_nan
from pyro.infer.util import torch_item

from .base_multiple_steps_gp_loss import BaseMultipleStepsGPLoss
from alef.active_learners.amortized_policies.utils.gp_computers import GaussianProcessComputer

"""
We implement sPCE losses, for mathematical details, please refer to the following papers:



"""

class BasePCELoss(BaseMultipleStepsGPLoss):
    def __init__(
        self,
        batch_size,
        num_kernels,
        num_functions_per_kernel,
        num_splits=1,
        data_source=None,
        name=None
    ):
        super().__init__(batch_size, num_kernels, num_functions_per_kernel, num_splits, data_source=data_source, name=name)
        self.process_kwargs = {
            'batch_size': self.batch_size,
            'num_kernels': self.num_kernels,
            'num_functions': self.num_functions_per_kernel + 1,
            'sample_domain_grid_points': False
        }

    def differentiable_loss(self, process, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        return self.compute_loss(
            process, self.process_kwargs
        )

    def loss(self, process, *args, **kwargs):
        """
        :returns: returns an estimate of the entropy
        :rtype: float
        Evaluates the minus entropy
        """
        loss_to_constant = torch_item(self.differentiable_loss(process, *args, **kwargs))
        return loss_to_constant


######
### Test idea: iDAD loss propos 3 by replacing U(...) to gp log p(y|f, \phi)
######

class _GPPCEComputer(GaussianProcessComputer):
    def forward(self, kernel_list, noise_var_list, X, Y, *args, **kwargs):
        """
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param X: [B, Nk, Nf+1, T + n_init, D] tensor on a single device
        :param Y: [B, Nk, Nf+1, T + n_init] tensor on a single device

        :return: [B, Nk] tensor on a single device, where the mean should be the loss
        """
        X_shape = X.shape
        Y_shape = Y.shape
        L = Y_shape[-2]
        X_primary = X[:, :, 0, None, :, :] # [B, Nk, 1, T, D]
        Y_primary = Y[:, :, 0, None, :] # [B, Nk, 1, T]

        K = self.compute_kernel_batch(
            kernel_list,
            X_primary.expand(X_shape),
            X_primary.expand(X_shape),
            noise_var_list = noise_var_list,
            batch_dim=1,
            return_linear_operator=False
        ) # [B, num_kernels, num_functions, T, T]
        log_prob = self.compute_gaussian_log_likelihood(Y_primary.expand(Y_shape), 0, K) # [B, num_kernels, num_functions+1]

        primary_log_prob = log_prob[:, :, 0] # [B, num_kernels]
        combined_mean_log_prob = log_prob.logsumexp(dim=-1) / L # [B, num_kernels]

        sPCE = combined_mean_log_prob - primary_log_prob # [B, num_kernels]
        return sPCE

import logging
from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)

class GPMutualInformationPCELoss(BasePCELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.critical(f'{self.__class__.__name__} is under development and is not tested.')
        raise NotImplementedError
        self.loss_computer = _GPPCEComputer()
