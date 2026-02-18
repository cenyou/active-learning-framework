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

class BaseEntropyLoss(BaseMultipleStepsGPLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.process_kwargs = {
            'batch_size': self.batch_size,
            'num_kernels': self.num_kernels,
            'num_functions': self.num_functions_per_kernel,
            'sample_domain_grid_points': False
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



class _GPEntropyComputer1(GaussianProcessComputer):
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

    def entropy_loss(
        self,
        mean_list, kernel_list, noise_var_list,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init,
        X_query, Y_query,
        *args, **kwargs
    ):
        """
        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialGaussianProcessContinuousDomain.process(*)
        output is negative entropy values
        
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
            loss = -entropy
        else:
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
            loss = -entropy
        # return loss values in batch
        return loss

    def forward(
        self,
        mean_list, kernel_list, noise_var_list,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init,
        X_query, Y_query,
        *args, **kwargs
    ):
        """
        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialGaussianProcessContinuousDomain.process(*)
        output is negative entropy values
        
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

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        normalize_factor = n_init + n_query1 + n_query2  # [B, num_kernels, num_functions]
        entropy_loss = self.entropy_loss(
            mean_list, kernel_list, noise_var_list,
            mask_dim,
            n_init, n_query1, n_query2,
            X_init, Y_init,
            X_query, Y_query,
            *args, **kwargs)
        entropy_loss = entropy_loss / normalize_factor
        if not self.reuse_prestart_samples or torch.allclose(n_query1, torch.zeros_like(n_query1)):
            return entropy_loss
        else:
            # compute 1/(n_init + n_q1) H( y_q1 | y_init )
            #         + 1/(n_init + n_q1 + n_q2) H( y_q2 | y_init, y_q1 )
            prestart_entropy_loss = self.entropy_loss(
                mean_list, kernel_list, noise_var_list,
                mask_dim,
                n_init, torch.zeros_like(n_query1), n_query1,
                X_init, Y_init,
                X_query, Y_query,
                *args, **kwargs)
            prestart_entropy_loss = prestart_entropy_loss / (n_init + n_query1).clamp(min=1) # n_query1==0 have loss==0 anyways
            return entropy_loss + prestart_entropy_loss

class GPEntropy1Loss(BaseEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPEntropyComputer1(myopic=False)


class _GPEntropyComputer2(GaussianProcessComputer):
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

    def log_prob_loss(
        self,
        mean_list, kernel_list, noise_var_list,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init,
        X_query, Y_query,
        *args, **kwargs
    ):
        """
        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialGaussianProcessContinuousDomain.process(*)
        output is log_prob values
        
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
        else:
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
        # return loss values in batch
        return log_prob

    def forward(
        self,
        mean_list, kernel_list, noise_var_list,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init,
        X_query, Y_query,
        *args, **kwargs
    ):
        """
        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialGaussianProcessContinuousDomain.process(*)
        output is log_prob values
        
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

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        normalize_factor = n_init + n_query1 + n_query2  # [B, num_kernels, num_functions]
        log_prob_loss = self.log_prob_loss(
            mean_list, kernel_list, noise_var_list,
            mask_dim,
            n_init, n_query1, n_query2,
            X_init, Y_init,
            X_query, Y_query,
            *args, **kwargs)
        log_prob_loss = log_prob_loss / normalize_factor
        if not self.reuse_prestart_samples or torch.allclose(n_query1, torch.zeros_like(n_query1)):
            return log_prob_loss
        else:
            # compute 1/(n_init + n_q1) log p( y_q1 | y_init )
            #         + 1/(n_init + n_q1 + n_q2) log p( y_q2 | y_init, y_q1 )
            prestart_log_prob_loss = self.log_prob_loss(
                mean_list, kernel_list, noise_var_list,
                mask_dim,
                n_init, torch.zeros_like(n_query1), n_query1,
                X_init, Y_init,
                X_query, Y_query,
                *args, **kwargs)
            prestart_log_prob_loss = prestart_log_prob_loss / (n_init + n_query1).clamp(min=1) # n_query1==0 have loss==0 anyways
            return log_prob_loss + prestart_log_prob_loss


class GPEntropy2Loss(BaseEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPEntropyComputer2(myopic=False)


