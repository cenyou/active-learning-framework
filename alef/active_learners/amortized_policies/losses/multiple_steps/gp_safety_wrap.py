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
from typing import Tuple, List, Optional

from .base_multiple_steps_gp_loss import BaseMultipleStepsGPLoss
from alef.active_learners.amortized_policies.utils.gp_computers import GaussianProcessComputer
from alef.active_learners.amortized_policies.utils.safety_probabilities import (
    TrivialSafetyProbability,
    SigmoidSafetyProbability,
    SigmoidSoftplusSafetyProbability,
)
from alef.configs.base_parameters import NUMERICAL_POSITIVE_LOWER_BOUND
from alef.enums.active_learner_amortized_policy_enums import SafetyProbability, SafetyProbabilityWrapper

import logging
from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)

class _GPRMSEWrapComputer(GaussianProcessComputer):
    def __init__(self, base_computer: GaussianProcessComputer):
        super().__init__()
        self.base_computer = base_computer

    def forward(
        self,
        mean_list,
        kernel_list,
        noise_var_list,
        mean_list_safety,
        kernel_list_safety,
        noise_var_list_safety,
        mask_dim,
        n_init,
        n_queries,
        X,
        Y,
        Z,
        *args, **kwargs
    ):
        """
        filter out arguments not used in the base computer, and run the base computer as it is.
        the result is the same as the base computer's result

        :param mean_list: a list of Nk gpytorch mean on a single device
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param mean_list_safety: a list of Nk gpytorch mean on a single device
        :param kernel_list_safety: a list of Nk gpytorch kernel on a single device
        :param noise_var_list_safety: a list of Nk observation noise variance on a single device
        :param mask_dim: [B, Nk, Nf, max_dim] tensor of 1 or 0, mask of the sampled D
        :param n_init: [B, Nk, Nf] tensor, actual num of initial samples
        :param n_queries: [B, Nk, Nf] tensor, actual num of data points queried by NN
        :param X: [B, Nk, Nf, n_init + T, D] tensor on a single device
        :param Y: [B, Nk, Nf, n_init + T] tensor on a single device
        :param Z: [B, Nk, Nf, n_init + T] tensor on a single device
        :param args: additional arguments
        :param kwargs: additional keyword arguments

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        return self.base_computer(
            mean_list, kernel_list, noise_var_list, mask_dim, n_init, n_queries, X, Y, *args, **kwargs
        ) # [B, Nk, Nf]


class _GPSafetyProbabilityComputer(GaussianProcessComputer):
    def __init__(self, alpha: float=0.05, *args, **kwargs):
        """
        :param alpha: p(z >= safety_lower_bound) >= 1 - alpha is safe
        """
        super().__init__()
        assert alpha <= 1 - NUMERICAL_POSITIVE_LOWER_BOUND, f'invalid, alpha={alpha} >= 1, means safety condition with p(z >= safety_lower_bound).clamp(1-alpha) == 0 '
        if alpha < 0:
            logger.warning(f'alpha={alpha} < 0, optimize safety condition with p(z >= safety_lower_bound) ')
        else:
            logger.info(f'alpha={alpha} >= 1, optimize safety condition with p(z >= safety_lower_bound).clamp(1 - alpha)')
        self.__alpha = alpha

    def forward(
        self,
        mean_list_safety,
        kernel_list_safety,
        noise_var_list_safety,
        mask_dim,
        n_init,
        X_init,
        Z_init,
        X_query,
        Z_query,
        safety_lower_bound: float=0.0,
        *args, **kwargs
    ):
        """
        compute cdf(threshold | Z[:-1]), Z ~ [GP(0, kernel) + N(0, noise_var)]

        :param mean_list_safety: a list of Nk gpytorch mean on a single device
        :param kernel_list_safety: a list of Nk gpytorch kernel on a single device
        :param noise_var_list_safety: a list of Nk observation noise variance on a single device
        :param mask_dim: [B, Nk, Nf, max_dim] tensor of 1 or 0, mask of the sampled D
        :param n_init: [B, Nk, Nf] tensor, actual num of initial samples
        :param X_init: [B, Nk, Nf, n_init_max, D] tensor on a single device
        :param Z_init: [B, Nk, Nf, n_init_max] tensor on a single device
        :param X_query: [B, Nk, Nf, T, D] tensor on a single device
        :param Z_query: [B, Nk, Nf, T] tensor on a single device
        :param safety_lower_bound: p(z >= safety_lower_bound)
        :param args: additional arguments
        :param kwargs: additional keyword arguments

        :return: [B, Nk, Nf, T] tensor on a single device, p(z >= safety_lower_bound).clamp(max=alpha)
        """
        n_init_max = Z_init.shape[-1]

        if n_init_max <= 0:
            raise NotImplementedError("Currently, initial data is required for safety computation")
        else:
            with torch.no_grad():
                mu_init = self.compute_mean_batch(
                    mean_list_safety,
                    X_init,
                    mask=mask_dim,
                    batch_dim=1
                ) # [B, num_kernels, num_functions, n_init]
                K_init = self.compute_kernel_batch(
                    kernel_list_safety,
                    X_init,
                    X_init,
                    noise_var_list = noise_var_list_safety,
                    mask=mask_dim,
                    batch_dim=1,
                    return_linear_operator=False
                ) # [B, num_kernels, num_functions, n_init, n_init]
                
            K_cross = self.compute_kernel_batch(
                kernel_list_safety,
                X_init,
                X_query,
                mask=mask_dim,
                batch_dim=1,
                return_linear_operator=False
            ) # [B, num_kernels, num_functions, n_init, T]
            K = self.compute_kernel_batch(
                kernel_list_safety,
                X_query,
                X_query,
                noise_var_list = noise_var_list_safety,
                mask=mask_dim,
                batch_dim=1,
                return_linear_operator=False
            ) # [B, num_kernels, num_functions, T, T]
            # want to compute: p(z_t >= 0 | Z_init, z_1:t-1), but Z_init may have different nums in a batch
            # so we first compute q(z_1:T) = p(z_1:T | Z_init) = N(mu_post, cov_post)
            # treat q as a GP as if there is no initial data,
            # then just compute the GP posterior q(z_t | z_1:t-1)
            # (cov[1:t-1, 1:t-1] as prior, cov_post[1:t-1,t] as cross covariances, cov_post[t,t] as unconditioned target variance)
            mu_query = self.compute_mean_batch(
                mean_list_safety,
                X_query,
                mask=mask_dim,
                batch_dim=1
            ) # [B, num_kernels, num_functions, T]
            mu_post, cov_post = self.compute_gaussian_process_posterior(
                K_init,
                K_cross, # [B, num_kernels, num_functions, n_init, T]
                K, # [B, num_kernels, num_functions, T, T]
                Y_observation=Z_init, # [B, num_kernels, num_functions, n_init]
                mean_prior=mu_init, # [B, num_kernels, num_functions, n_init]
                mean_test=mu_query, # [B, num_kernels, num_functions, T]
                prior_mask_idx=n_init,
                return_mu=True, return_cov=True
            )
            for t in range(self.n_steps_max):
                if t==0:
                    mu_conditional, K_conditional = mu_post[..., 0, None], cov_post[..., None, 0, None, 0]
                else:
                    mu_conditional, K_conditional = self.compute_gaussian_process_posterior(
                        cov_post[..., :t, :t],
                        cov_post[..., :t, t, None], # [B, num_kernels, num_functions, t, 1]
                        cov_post[..., t, None, t, None], # [B, num_kernels, num_functions, 1, 1]
                        Y_observation=Z_query[..., :t], # [B, num_kernels, num_functions, t]
                        mean_prior=mu_post[..., :t], # [B, num_kernels, num_functions, t]
                        mean_test=mu_post[..., t, None], # [B, num_kernels, num_functions, 1]
                        return_mu=True, return_cov=True
                    )
                # mu_conditional: [B, num_kernels, num_functions, 1]
                # K_conditional: [B, num_kernels, num_functions, 1, 1]
                # threshold: [B, num_kernels, num_functions, 1]
                std_conditional = K_conditional.sqrt().squeeze(-1) # [B, num_kernels, num_functions, 1]
                cdf = 1.0 - 0.5 * (
                    1 + torch.erf((safety_lower_bound - mu_conditional) * std_conditional.reciprocal() / math.sqrt(2))
                ) # [B, num_kernels, num_functions, 1]
                if t == 0:
                    p_z = cdf
                else:
                    p_z = torch.cat([p_z, cdf], dim=-1) # [B, num_kernels, num_functions, t+1]
            return p_z.clamp( max=(1 - self.__alpha) ) # [B, num_kernels, num_functions, T]


class _GPSafetyWrapBaseComputer(GaussianProcessComputer):
    def __init__(
        self,
        information_loss_computer: GaussianProcessComputer,
        probability_function: SafetyProbability,
        probability_function_args: Tuple[float, float] = (0.05, -0.05),
        probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT,
        safety_discount_ratio: float = 1.0,
    ):
        r"""
        :param information_loss_computer: the base loss computer
        :param probability_function: the safety probability function specified by an enum
        :param probability_function_args: the arguments for the safety probability function
        """
        #if probability_function == SafetyProbability.GP_POSTERIOR:
        #    assert information_loss_computer.myopic
        assert safety_discount_ratio >= 0
        super().__init__()
        self.information_loss_computer = information_loss_computer
        self.set_safety_probability_function(probability_function, probability_function_args)
        self.probability_wrap_mode = probability_wrap_mode
        self.safety_discount_ratio = safety_discount_ratio

    def load_sequence_milestones(self, n_initial_min, n_initial_max, n_steps_min, n_steps_max):
        """
        We might need only the queries for certain computations.
        The pipeline always have the initial set placed first, followed by the queries.
        The learner does AL from n_initial for n_steps, but training pipeline may sample n_initial & n_steps.
        
        :param n_initial_min: min num of initial points
        :param n_initial_max: max num of initial points
        :param n_steps_min: min num of policy selected in each batch
        :param n_steps_max: max num of policy selected in each batch
        """
        super().load_sequence_milestones(n_initial_min, n_initial_max, n_steps_min, n_steps_max)
        self.information_loss_computer.load_sequence_milestones(n_initial_min, n_initial_max, n_steps_min, n_steps_max)
        if isinstance(self.safety_probability, _GPSafetyProbabilityComputer):
            self.safety_probability.load_sequence_milestones(n_initial_min, n_initial_max, n_steps_min, n_steps_max)

    @property
    def reuse_prestart_samples(self):
        return self.information_loss_computer.reuse_prestart_samples

    @reuse_prestart_samples.setter
    def reuse_prestart_samples(self, flag: bool):
        self.information_loss_computer.reuse_prestart_samples = flag

    def set_safety_probability_function(self, probability_function, probability_function_args):
        if probability_function == SafetyProbability.TRIVIAL:
            self.safety_probability = TrivialSafetyProbability(*probability_function_args)
        elif probability_function == SafetyProbability.SIGMOID:
            self.safety_probability = SigmoidSafetyProbability(*probability_function_args)
        elif probability_function == SafetyProbability.SIGMOID_SOFTPLUS:
            self.safety_probability = SigmoidSoftplusSafetyProbability(*probability_function_args)
        elif probability_function == SafetyProbability.GP_POSTERIOR:
            self.safety_probability = _GPSafetyProbabilityComputer(*probability_function_args)
        else:
            raise NotImplementedError

    def wrap_losses(self, information_loss, p_z, n_init, n_query1, n_query2):
        """
        in each forward function, we compute the information loss and safety probability,
        and then call this method to wrap the information loss and prob together

        Args:
            information_loss [B, Nk, Nf] torch.Tensor
            p_z [B, Nk, Nf, T] torch.Tensor
            n_init: [B, Nk, Nf] tensor, actual num of initial samples
            n_query1: [B, Nk, Nf] int tensor, num of queries prestart (will be 0 unless we use NN to query 2 seqs)
            n_query2: [B, Nk, Nf] int tensor, actual num of data points queried by NN
                note that information loss already take n_init, n_query1, n_query2 into account

        Returns:
            [B, Nk, Nf] torch.Tensor
        """
        if self.probability_wrap_mode == SafetyProbabilityWrapper.NONE:
            return information_loss
        elif self.probability_wrap_mode == SafetyProbabilityWrapper.PRODUCT:
            assert torch.allclose(n_query1, torch.zeros_like(n_query1)), NotImplementedError
            # do p_z.log().mean(-1).exp() but masked
            # the masked sum is done with super()._sum_masked()
            post_mask_idx = torch.cat([
                n_query1.unsqueeze(-1),
                (n_query1 + n_query2).unsqueeze(-1)
            ], dim=-1)
            p_z_geometric_mean = super()._sum_masked( p_z.log() / p_z.shape[-1], post_mask_idx ).exp()# [B, Nk, Nf]
            return information_loss * (p_z_geometric_mean ** self.safety_discount_ratio)
        elif self.probability_wrap_mode in [
            SafetyProbabilityWrapper.LOGCONDITION, SafetyProbabilityWrapper.JOINTPROBABILITY
        ]:
            if self.probability_wrap_mode == SafetyProbabilityWrapper.LOGCONDITION:
                # minimize - log safe prob
                # do p_z.log().neg().sum(-1) but masked
                # the masked sum is done with super()._sum_masked()
                log_pz = (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log().neg()
            elif self.probability_wrap_mode == SafetyProbabilityWrapper.JOINTPROBABILITY:
                # minimize unsafe prob
                # do (1 - p_z).log().sum(-1) but masked
                # the masked sum is done with super()._sum_masked()
                log_pz = (1 - p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log()
            post_mask_idx = torch.cat([
                n_query1.unsqueeze(-1),
                (n_query1 + n_query2).unsqueeze(-1)
            ], dim=-1)
            normalize_length = n_init + n_query1 + n_query2
            safe_score = super()._sum_masked( log_pz, post_mask_idx ) / normalize_length
            if self.reuse_prestart_samples and not torch.allclose(n_query1, torch.zeros_like(n_query1)):
                post_mask_idx = torch.cat([
                    torch.zeros_like(n_query1).unsqueeze(-1),
                    n_query1.unsqueeze(-1)
                ], dim=-1)
                safe_score += super()._sum_masked( log_pz, post_mask_idx ) / (n_init + n_query1).clamp(min=1) # n_query1==0 have sum==0 anyways
            return information_loss + self.safety_discount_ratio * safe_score
        else:
            raise NotImplementedError

class _GPSafetyWrapWithoutGridsComputer(_GPSafetyWrapBaseComputer):

    def forward(
        self,
        mean_list,
        kernel_list,
        noise_var_list,
        mean_list_safety,
        kernel_list_safety,
        noise_var_list_safety,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init, Z_init,
        X_query, Y_query, Z_query,
        *args, **kwargs
    ):
        """
        run the base computer as it is, wrap the result with the safety loss
        this is used for the safety wrap loss

        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialSafeGaussianProcessContinuousDomain.process(*)
        
        :param mean_list: a list of Nk gpytorch mean on a single device
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param mean_list_safety: a list of Nk gpytorch mean on a single device
        :param kernel_list_safety: a list of Nk gpytorch kernel on a single device
        :param noise_var_list_safety: a list of Nk observation noise variance on a single device
        
        :param n_init: [B, Nk, Nf] tensor, actual num of initial samples
        :param n_query1: [B, Nk, Nf] tensor, num of queries prestart (will be 0 unless we use NN to query 2 seqs)
        :param n_query2: [B, Nk, Nf] tensor, actual num of data points queried by NN
        :param X_init: [B, Nk, Nf, n_init_max, D] tensor on a single device
        :param Y_init: [B, Nk, Nf, n_init_max] tensor on a single device
        :param Z_init: [B, Nk, Nf, n_init_max] tensor on a single device
        :param X_query: [B, Nk, Nf, T, D] tensor on a single device
        :param Y_query: [B, Nk, Nf, T] tensor on a single device
        :param Z_query: [B, Nk, Nf, T] tensor on a single device
        :param args: additional arguments
        :param kwargs: additional keyword arguments

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        if isinstance(self.safety_probability, _GPSafetyProbabilityComputer):
            p_z = self.safety_probability(
                mean_list_safety, kernel_list_safety, noise_var_list_safety,
                mask_dim,
                n_init,
                X_init, Z_init,
                X_query, Z_query,
                safety_lower_bound=0.0
            ) # [B, Nk, Nf, T]
        else:
            p_z = self.safety_probability(Z_query) # [B, Nk, Nf, T]

        information_loss = self.information_loss_computer( # [B, Nk, Nf]
            mean_list, kernel_list, noise_var_list,
            mask_dim,
            n_init, n_query1, n_query2,
            X_init, Y_init,
            X_query, Y_query,
            *args, **kwargs
        )

        return self.wrap_losses(information_loss, p_z, n_init, n_query1, n_query2)


class GPSafetyEntropyWrapLoss(BaseMultipleStepsGPLoss):
    def __init__(
        self,
        information_loss: BaseMultipleStepsGPLoss,
        probability_function: SafetyProbability = SafetyProbability.TRIVIAL,
        probability_function_args: Tuple[float, float] = (0.05, -0.05),
        probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT,
        safety_discount_ratio: float = 1.0,
        data_source=None,
        name=None
    ):
        super().__init__(
            information_loss.batch_size,
            information_loss.num_kernels,
            information_loss.num_functions_per_kernel,
            information_loss.num_splits,
            data_source=data_source,
            name=name
        )
        self.loss_computer = _GPSafetyWrapWithoutGridsComputer(
            information_loss.loss_computer,
            probability_function,
            probability_function_args,
            probability_wrap_mode,
            safety_discount_ratio,
        )
        self.test_computer = _GPRMSEWrapComputer(information_loss.test_computer)
        self.process_kwargs = information_loss.process_kwargs

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


class _GPSafetyWrapWithGridsComputer(_GPSafetyWrapBaseComputer):
    
    def forward(
        self,
        mean_list,
        kernel_list,
        noise_var_list,
        mean_list_safety,
        kernel_list_safety,
        noise_var_list_safety,
        mask_dim,
        n_init, n_query1, n_query2,
        X_init, Y_init, Z_init,
        X_query, Y_query, Z_query,
        X_grid, Y_grid, Z_grid,
        *args, **kwargs
    ):
        """
        run the base computer as it is, wrap the result with the safety loss
        this is used for the safety wrap loss

        inputs here are the output of alef.active_learners.amortized_policies.simulated_processes.multiple_steps.SequentialSafeGaussianProcessContinuousDomain.process(*)
        
        :param mean_list: a list of Nk gpytorch mean on a single device
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param mean_list_safety: a list of Nk gpytorch mean on a single device
        :param kernel_list_safety: a list of Nk gpytorch kernel on a single device
        :param noise_var_list_safety: a list of Nk observation noise variance on a single device
        
        :param n_init: [B, Nk, Nf] tensor, actual num of initial samples
        :param n_query1: [B, Nk, Nf] tensor, num of queries prestart (will be 0 unless we use NN to query 2 seqs)
        :param n_query2: [B, Nk, Nf] tensor, actual num of data points queried by NN
        :param X_init: [B, Nk, Nf, n_init_max, D] tensor on a single device
        :param Y_init: [B, Nk, Nf, n_init_max] tensor on a single device
        :param Z_init: [B, Nk, Nf, n_init_max] tensor on a single device
        :param X_query: [B, Nk, Nf, T, D] tensor on a single device
        :param Y_query: [B, Nk, Nf, T] tensor on a single device
        :param Z_query: [B, Nk, Nf, T] tensor on a single device
        :param X_grid: [B, Nk, Nf, n_grid, D] tensor on a single device
        :param Y_grid: [B, Nk, Nf, n_grid] tensor on a single device
        :param Z_grid: [B, Nk, Nf, n_grid] tensor on a single device
        :param args: additional arguments
        :param kwargs: additional keyword arguments

        :return: [B, Nk, Nf] tensor on a single device, where the mean should be the loss
        """
        if isinstance(self.safety_probability, _GPSafetyProbabilityComputer):
            p_z = self.safety_probability(
                mean_list_safety, kernel_list_safety, noise_var_list_safety, mask_dim, n_init, X_init, Z_init, X_query, Z_query, safety_lower_bound=0.0
            )
        else:
            p_z = self.safety_probability(Z_query) # [B, Nk, Nf, T]

        information_loss = self.information_loss_computer( # [B, Nk, Nf]
            mean_list, kernel_list, noise_var_list,
            mask_dim,
            n_init, n_query1, n_query2,
            X_init, Y_init,
            X_query, Y_query,
            X_grid, Y_grid,
            *args, **kwargs
        )

        return self.wrap_losses(information_loss, p_z, n_init, n_query1, n_query2)



class GPSafetyMIWrapLoss(BaseMultipleStepsGPLoss):
    def __init__(
        self,
        information_loss: BaseMultipleStepsGPLoss,
        probability_function: SafetyProbability = SafetyProbability.TRIVIAL,
        probability_function_args: Tuple[float, float] = (0.05, -0.05),
        probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT,
        safety_discount_ratio: float = 1.0,
        data_source=None,
        name=None
    ):
        super().__init__(
            information_loss.batch_size,
            information_loss.num_kernels,
            information_loss.num_functions_per_kernel,
            information_loss.num_splits,
            data_source=data_source,
            name=name
        )
        self.loss_computer = _GPSafetyWrapWithGridsComputer(
            information_loss.loss_computer,
            probability_function,
            probability_function_args,
            probability_wrap_mode,
            safety_discount_ratio,
        )
        self.test_computer = _GPRMSEWrapComputer(information_loss.test_computer)
        self.process_kwargs = {
            **information_loss.process_kwargs,
            'constraint_domain_grid_points': True
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



