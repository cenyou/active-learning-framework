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
from alef.active_learners.amortized_policies.global_parameters import OVERALL_VARIANCE
from alef.active_learners.amortized_policies.utils.gp_computers import GaussianProcessComputer

import logging
from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)

class BaseMutualInformationLoss(BaseMultipleStepsGPLoss):
    def __init__(
        self,
        batch_size,
        num_kernels,
        num_functions_per_kernel,
        num_grid_points,
        num_splits=1,
        data_source=None,
        name=None
    ):
        super().__init__(batch_size, num_kernels, num_functions_per_kernel, num_splits, data_source=data_source, name=name)
        self.num_grid_points = num_grid_points
        self.process_kwargs = {
            'batch_size': self.batch_size,
            'num_kernels': self.num_kernels,
            'num_functions': self.num_functions_per_kernel,
            'sample_domain_grid_points': True,
            'num_grid_points': self.num_grid_points
        }

    def differentiable_loss(self, process, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the design
        """
        self.loss_computer.load_sequence_milestones(process.n_initial_min, process.n_initial_max, process.n_steps_min, process.n_steps_max)
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



class _GPMutualInformationComputer1(GaussianProcessComputer):
    def __init__(
        self,
        myopic: bool = False,
        reuse_prestart_samples: bool = True,
    ):
        r"""
        :param myopic: flag, whether we only consider the conditional entropy of the last point
        :param reuse_prestart_samples: flag, whether we use n_init:n_init+n_query1 data as well (this points have gradient as well)
        """
        super().__init__()
        self.myopic = myopic
        self.reuse_prestart_samples = reuse_prestart_samples

    def entropy_reduction_loss(
        self,
        mean_list, kernel_list, noise_var_list,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init,
        X_query, Y_query,
        X_grid, Y_grid,
        *args, **kwargs
    ):
        """
        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialGaussianProcessContinuousDomain.process(*)
        output is negative entropy reduction values
        
        :param mean_list: a list of Nk gpytorch mean on a single device
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param mask_dim: [B, Nk, Nf, max_dim] tensor of 1 or 0, mask of the sampled D
        :param n_init: [B, Nk, Nf] tensor, actual num of initial samples
        :param n_query1: [B, Nk, Nf] tensor, num of queries prestart (will be 0 unless we use NN to query 2 seqs)
        :param n_query2: [B, Nk, Nf] tensor, actual num of data points queried by NN
        :param X_init: [B, Nk, Nf, n_init_max, D] tensor on a single device
        :param Y_init: [B, Nk, Nf, n_init_max] tensor on a single device
        :param X_query: [B, Nk, Nf, T, D] tensor on a single device
        :param Y_query: [B, Nk, Nf, T] tensor on a single device
        :param X_grid: [B, Nk, Nf, n_grid, D] tensor on a single device
        :param Y_grid: [B, Nk, Nf, n_grid] tensor on a single device

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        # if we optimize random sub sequences of queries
        post_mask_idx = torch.cat([
            n_query1.unsqueeze(-1),
            (n_query1 + n_query2).unsqueeze(-1)
        ], dim=-1)

        if torch.all(n_init <= 0):
            K = self.compute_kernel_batch(kernel_list, X_query, X_query, noise_var_list = noise_var_list, mask=mask_dim, batch_dim=1, return_linear_operator=False) # [B, num_kernels, num_functions, T, T]
            entropy = self.compute_gaussian_entropy(K, mask_idx=post_mask_idx) # [B, num_kernels, num_functions]
            #
            # now compute conditional entropy
            #
            with torch.no_grad():
                K_grid = self.compute_kernel_batch(kernel_list, X_grid, X_grid, noise_var_list = noise_var_list, mask=mask_dim, batch_dim=1, return_linear_operator=False) # [B, num_kernels, num_functions, n_grid, n_grid]
            K_cross = self.compute_kernel_batch(kernel_list, X_grid, X_query, mask=mask_dim, batch_dim=1, return_linear_operator=False) # [B, num_kernels, num_functions, n_grid, T]

            K_conditional = self.compute_gaussian_process_posterior(
                K_grid,
                K_cross,
                K,
                return_mu=False, return_cov=True
            ) # [B, num_kernels, num_functions, T, T]

            regularizer = self.compute_gaussian_entropy(K_conditional, mask_idx=post_mask_idx) # [B, num_kernels, num_functions]

            neg_mi = regularizer - entropy # [B, num_kernels, num_functions]
        else:
            # compute H[Y_queries | Y_init]
            with torch.no_grad():
                K_init = self.compute_kernel_batch(
                    kernel_list,
                    X_init,
                    X_init,
                    noise_var_list = noise_var_list,
                    mask=mask_dim,
                    batch_dim=1,
                    return_linear_operator=False
                ) # [B, num_kernels, num_functions, n_init, n_init]
            K_cross = self.compute_kernel_batch(
                kernel_list,
                X_init,
                X_query,
                mask=mask_dim,
                batch_dim=1,
                return_linear_operator=False
            ) # [B, num_kernels, num_functions, n_init, T]
            K = self.compute_kernel_batch(
                kernel_list,
                X_query,
                X_query,
                noise_var_list = noise_var_list,
                mask=mask_dim,
                batch_dim=1,
                return_linear_operator=False
            ) # [B, num_kernels, num_functions, T, T]
            K_conditional = self.compute_gaussian_process_posterior(
                K_init,
                K_cross,
                K,
                prior_mask_idx=n_init,
                return_mu=False, return_cov=True
            ) # [B, num_kernels, num_functions, T, T]

            entropy = self.compute_gaussian_entropy(K_conditional, mask_idx=post_mask_idx) # [B, num_kernels, num_functions]
            #
            # now compute H[Y_queries | Y_init, Y_grid]
            #
            with torch.no_grad():
                X_prior = torch.cat([X_grid, X_init], dim=-2)
                K_prior = self.compute_kernel_batch(
                    kernel_list,
                    X_prior,
                    X_prior,
                    noise_var_list = noise_var_list,
                    mask=mask_dim,
                    batch_dim=1,
                    return_linear_operator=False
                ) # [B, num_kernels, num_functions, n_grid + n_init, n_grid + n_init]
            K_cross = self.compute_kernel_batch(
                kernel_list,
                X_prior,
                X_query,
                mask=mask_dim,
                batch_dim=1,
                return_linear_operator=False
            ) # [B, num_kernels, num_functions, n_grid + n_init, T]
            K_conditional = self.compute_gaussian_process_posterior(
                K_prior,
                K_cross,
                K,
                prior_mask_idx=X_grid.shape[-2] + n_init,
                return_mu=False, return_cov=True
            ) # [B, num_kernels, num_functions, T, T]

            regularizer = self.compute_gaussian_entropy(K_conditional, mask_idx=post_mask_idx) # [B, num_kernels, num_functions]

            neg_mi = regularizer - entropy # [B, num_kernels, num_functions]
        # return loss values in batch
        return neg_mi

    def forward(
        self,
        mean_list, kernel_list, noise_var_list,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init,
        X_query, Y_query,
        X_grid, Y_grid,
        *args, **kwargs
    ):
        """
        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialGaussianProcessContinuousDomain.process(*)
        output is negative entropy reduction values
        
        :param mean_list: a list of Nk gpytorch mean on a single device
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param mask_dim: [B, Nk, Nf, max_dim] tensor of 1 or 0, mask of the sampled D
        :param n_init: [B, Nk, Nf] tensor, actual num of initial samples
        :param n_query1: [B, Nk, Nf] tensor, num of queries prestart (will be 0 unless we use NN to query 2 seqs)
        :param n_query2: [B, Nk, Nf] tensor, actual num of data points queried by NN
        :param X_init: [B, Nk, Nf, n_init_max, D] tensor on a single device
        :param Y_init: [B, Nk, Nf, n_init_max] tensor on a single device
        :param X_query: [B, Nk, Nf, T, D] tensor on a single device
        :param Y_query: [B, Nk, Nf, T] tensor on a single device
        :param X_grid: [B, Nk, Nf, n_grid, D] tensor on a single device
        :param Y_grid: [B, Nk, Nf, n_grid] tensor on a single device

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        normalize_factor = n_init + n_query1 + n_query2  # [B, num_kernels, num_functions]
        entropy_reduction_loss = self.entropy_reduction_loss(
            mean_list, kernel_list, noise_var_list,
            mask_dim,
            n_init, n_query1, n_query2,
            X_init, Y_init,
            X_query, Y_query,
            X_grid, Y_grid,
            *args, **kwargs)
        entropy_reduction_loss = entropy_reduction_loss / normalize_factor
        if not self.reuse_prestart_samples or torch.allclose(n_query1, torch.zeros_like(n_query1)):
            return entropy_reduction_loss
        else:
            # compute 1/(n_init + n_q1) { H( y_q1 | y_init ) - H( y_q1 | y_init, y_grid ) }
            #         + 1/(n_init + n_q1 + n_q2) { H( y_q2 | y_init, y_q1 ) - H( y_q1 | y_init, y_q1, y_grid ) }
            prestart_entropy_reduction_loss = self.entropy_reduction_loss(
                mean_list, kernel_list, noise_var_list,
                mask_dim,
                n_init, torch.zeros_like(n_query1), n_query1,
                X_init, Y_init,
                X_query, Y_query,
                X_grid, Y_grid,
                *args, **kwargs)
            prestart_entropy_reduction_loss = prestart_entropy_reduction_loss / (n_init + n_query1).clamp(min=1) # n_query1==0 have loss==0 anyways
            return entropy_reduction_loss + prestart_entropy_reduction_loss


class GPMutualInformation1Loss(BaseMutualInformationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPMutualInformationComputer1(myopic=False)


class _GPMutualInformationComputer2(GaussianProcessComputer):
    def __init__(
        self,
        myopic: bool = False,
        reuse_prestart_samples: bool = True,
    ):
        r"""
        :param myopic: flag, whether we only consider the conditional entropy of the last point
        :param reuse_prestart_samples: flag, whether we use n_init:n_init+n_query1 data as well (this points have gradient as well)
        """
        super().__init__()
        self.myopic = myopic
        self.reuse_prestart_samples = reuse_prestart_samples

    def log_prob_reduction(
        self,
        mean_list, kernel_list, noise_var_list,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init,
        X_query, Y_query,
        X_grid, Y_grid,
        *args, **kwargs
    ):
        """
        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialGaussianProcessContinuousDomain.process(*)
        output is log_prob_reduction values
        
        :param mean_list: a list of Nk gpytorch mean on a single device
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param mask_dim: [B, Nk, Nf, max_dim] tensor of 1 or 0, mask of the sampled D
        :param n_init: [B, Nk, Nf] tensor, actual num of initial samples
        :param n_query1: [B, Nk, Nf] tensor, num of queries prestart (will be 0 unless we use NN to query 2 seqs)
        :param n_query2: [B, Nk, Nf] tensor, actual num of data points queried by NN
        :param X_init: [B, Nk, Nf, n_init_max, D] tensor on a single device
        :param Y_init: [B, Nk, Nf, n_init_max] tensor on a single device
        :param X_query: [B, Nk, Nf, T, D] tensor on a single device
        :param Y_query: [B, Nk, Nf, T] tensor on a single device
        :param X_grid: [B, Nk, Nf, n_grid, D] tensor on a single device
        :param Y_grid: [B, Nk, Nf, n_grid] tensor on a single device

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        # if we optimize random sub sequences of queries
        post_mask_idx = torch.cat([
            n_query1.unsqueeze(-1),
            (n_query1 + n_query2).unsqueeze(-1)
        ], dim=-1)

        if torch.all(n_init <= 0):
            mu = self.compute_mean_batch(mean_list, X_query, mask=mask_dim, batch_dim=1)
            K = self.compute_kernel_batch(kernel_list, X_query, X_query, noise_var_list = noise_var_list, mask=mask_dim, batch_dim=1, return_linear_operator=False) # [B, num_kernels, num_functions, T, T]
            log_prob = self.compute_gaussian_log_likelihood(Y_query, mu, K, mask_idx=post_mask_idx) # [B, num_kernels, num_functions]
            #
            # now compute conditional log_prob
            #
            with torch.no_grad():
                mu_grid = self.compute_mean_batch(mean_list, X_grid, mask=mask_dim, batch_dim=1)
                K_grid = self.compute_kernel_batch(kernel_list, X_grid, X_grid, noise_var_list = noise_var_list, mask=mask_dim, batch_dim=1, return_linear_operator=False) # [B, num_kernels, num_functions, n_grid, n_grid]
            K_cross = self.compute_kernel_batch(kernel_list, X_grid, X_query, mask=mask_dim, batch_dim=1, return_linear_operator=False) # [B, num_kernels, num_functions, n_grid, T]

            mu_conditional, K_conditional = self.compute_gaussian_process_posterior(
                K_grid,
                K_cross,
                K,
                Y_observation=Y_grid,
                mean_prior=mu_grid,
                mean_test=mu,
                return_mu=True, return_cov=True
            )

            regularizer = self.compute_gaussian_log_likelihood(Y_query, mu_conditional, K_conditional, mask_idx=post_mask_idx) # [B, num_kernels, num_functions]

            neg_mi = log_prob - regularizer # [B, num_kernels, num_functions]
        else:
            # log p(Y_queries | Y_init)
            with torch.no_grad():
                mu_init = self.compute_mean_batch(
                    mean_list,
                    X_init,
                    mask=mask_dim,
                    batch_dim=1
                ) # [B, num_kernels, num_functions, n_init]
                K_init = self.compute_kernel_batch(
                    kernel_list,
                    X_init,
                    X_init,
                    noise_var_list = noise_var_list,
                    mask=mask_dim,
                    batch_dim=1,
                    return_linear_operator=False
                ) # [B, num_kernels, num_functions, n_init, n_init]
            K_cross = self.compute_kernel_batch(
                kernel_list,
                X_init,
                X_query,
                mask=mask_dim,
                batch_dim=1,
                return_linear_operator=False
            ) # [B, num_kernels, num_functions, n_init, T]
            mu = self.compute_mean_batch(
                mean_list,
                X_query,
                mask=mask_dim,
                batch_dim=1
            ) # [B, num_kernels, num_functions, T]
            K = self.compute_kernel_batch(
                kernel_list,
                X_query,
                X_query,
                noise_var_list = noise_var_list,
                mask=mask_dim,
                batch_dim=1,
                return_linear_operator=False
            ) # [B, num_kernels, num_functions, T, T]
            mu_conditional, K_conditional = self.compute_gaussian_process_posterior(
                K_init,
                K_cross,
                K,
                Y_observation=Y_init,
                mean_prior=mu_init,
                mean_test=mu,
                prior_mask_idx=n_init,
                return_mu=True, return_cov=True
            ) # [B, num_kernels, num_functions, T, T]
            log_prob = self.compute_gaussian_log_likelihood(Y_query, mu_conditional, K_conditional, mask_idx=post_mask_idx) # [B, num_kernels, num_functions]
            #
            # now compute log p(Y_queries | Y_init, Y_grid)
            #
            with torch.no_grad():
                X_prior = torch.cat([X_grid, X_init], dim=-2)
                Y_prior = torch.cat([Y_grid, Y_init], dim=-1)
                mu_prior = self.compute_mean_batch(
                    mean_list, X_prior, mask=mask_dim, batch_dim=1
                ) # [B, num_kernels, num_functions, n_grid + n_init]
                K_prior = self.compute_kernel_batch(
                    kernel_list,
                    X_prior,
                    X_prior,
                    noise_var_list = noise_var_list,
                    mask=mask_dim,
                    batch_dim=1,
                    return_linear_operator=False
                ) # [B, num_kernels, num_functions, n_grid + n_init, n_grid + n_init]
            K_cross = self.compute_kernel_batch(
                kernel_list,
                X_prior,
                X_query,
                mask=mask_dim,
                batch_dim=1,
                return_linear_operator=False
            ) # [B, num_kernels, num_functions, n_grid + n_init, T]

            mu_conditional, K_conditional = self.compute_gaussian_process_posterior(
                K_prior,
                K_cross,
                K,
                Y_observation=Y_prior,
                mean_prior=mu_prior, mean_test=mu,
                prior_mask_idx=X_grid.shape[-2] + n_init,
                return_mu=True, return_cov=True
            )

            regularizer = self.compute_gaussian_log_likelihood(Y_query, mu_conditional, K_conditional, mask_idx=post_mask_idx) # [B, num_kernels, num_functions]

            neg_mi = log_prob - regularizer # [B, num_kernels, num_functions]
        # return loss values in batch
        return neg_mi

    def forward(
        self,
        mean_list, kernel_list, noise_var_list,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init,
        X_query, Y_query,
        X_grid, Y_grid,
        *args, **kwargs
    ):
        """
        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialGaussianProcessContinuousDomain.process(*)
        output is log_prob_reduction values
        
        :param mean_list: a list of Nk gpytorch mean on a single device
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param mask_dim: [B, Nk, Nf, max_dim] tensor of 1 or 0, mask of the sampled D
        :param n_init: [B, Nk, Nf] tensor, actual num of initial samples
        :param n_query1: [B, Nk, Nf] tensor, num of queries prestart (will be 0 unless we use NN to query 2 seqs)
        :param n_query2: [B, Nk, Nf] tensor, actual num of data points queried by NN
        :param X_init: [B, Nk, Nf, n_init_max, D] tensor on a single device
        :param Y_init: [B, Nk, Nf, n_init_max] tensor on a single device
        :param X_query: [B, Nk, Nf, T, D] tensor on a single device
        :param Y_query: [B, Nk, Nf, T] tensor on a single device
        :param X_grid: [B, Nk, Nf, n_grid, D] tensor on a single device
        :param Y_grid: [B, Nk, Nf, n_grid] tensor on a single device

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        normalize_factor = n_init + n_query1 + n_query2  # [B, num_kernels, num_functions]
        log_prob_reduction = self.log_prob_reduction(
            mean_list, kernel_list, noise_var_list,
            mask_dim,
            n_init, n_query1, n_query2,
            X_init, Y_init,
            X_query, Y_query,
            X_grid, Y_grid,
            *args, **kwargs)
        log_prob_reduction = log_prob_reduction / normalize_factor
        if not self.reuse_prestart_samples or torch.allclose(n_query1, torch.zeros_like(n_query1)):
            return log_prob_reduction
        else:
            # compute 1/(n_init + n_q1) { log p( y_q1 | y_init ) - log p( y_q1 | y_init, y_grid ) }
            #         + 1/(n_init + n_q1 + n_q2) { log p( y_q2 | y_init, y_q1 ) - log p( y_q1 | y_init, y_q1, y_grid ) }
            prestart_log_prob_reduction = self.log_prob_reduction(
                mean_list, kernel_list, noise_var_list,
                mask_dim,
                n_init, torch.zeros_like(n_query1), n_query1,
                X_init, Y_init,
                X_query, Y_query,
                X_grid, Y_grid,
                *args, **kwargs)
            prestart_log_prob_reduction = prestart_log_prob_reduction / (n_init + n_query1).clamp(min=1) # n_query1==0 have loss==0 anyways
            return log_prob_reduction + prestart_log_prob_reduction


class GPMutualInformation2Loss(BaseMutualInformationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPMutualInformationComputer2(myopic=False)

