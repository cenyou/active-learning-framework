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
from torch.nn.parallel.scatter_gather import scatter, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch._utils import _get_all_device_indices

import pyro
from pyro.util import warn_if_nan

from ..base_loss import BaseLoss
from alef.active_learners.amortized_policies.utils.utils import compute_kernel_batch
from alef.active_learners.amortized_policies.utils.gp_computers import GaussianProcessComputer

from enum import Enum

class ProcessMethod(Enum):
    process = 0
    validation = 1

class _GPRMSEComputer(GaussianProcessComputer):
    
    def forward(self, mean_list, kernel_list, noise_var_list, mask_dim, n_init, n_queries, X, Y, X_test, Y_test, *args, **kwargs):
        """
        :param mean_list: a list of Nk gpytorch mean on a single device
        :param kernel_list: a list of Nk gpytorch kernel on a single device
        :param noise_var_list: a list of Nk observation noise variance on a single device
        :param mask_dim: [B, Nk, Nf, max_dim] tensor of 1 or 0, mask of the sampled D
        :param n_init: [B, Nk, Nf] tensor, actual num of initial samples
        :param n_queries: [B, Nk, Nf] tensor, actual num of data points queried by NN
        :param X: [B, Nk, Nf, T + n_init, D] tensor on a single device
        :param Y: [B, Nk, Nf, T + n_init] tensor on a single device
        :param X_test: [B, Nk, Nf, n_test, D] tensor on a single device
        :param Y_test: [B, Nk, Nf, n_test] tensor on a single device

        :return: [B, Nk, Nf] tensor of GP RMSEs on a single device
        """
        ###
        ### first compute p(f|X_test, X, Y)
        ### then compare Y_test to the mean of the posterior
        ###
        K_prior = self.compute_kernel_batch(kernel_list, X, X, noise_var_list = noise_var_list, batch_dim=1) # [B, num_kernels, num_functions, T, T]
        K_cross = self.compute_kernel_batch(kernel_list, X, X_test, batch_dim=1) # [B, num_kernels, num_functions, T, n_test]
        mean_prior = self.compute_mean_batch(mean_list, X, batch_dim=1)
        mean_test = self.compute_mean_batch(mean_list, X_test, batch_dim=1)

        mu_conditional = self.compute_gaussian_process_posterior(
            K_prior.to_dense(),
            K_cross.to_dense(),
            cov_test = None,
            Y_observation=Y,
            mean_prior=mean_prior,
            mean_test=mean_test,
            return_mu=True, return_cov=False
        ) # [B, num_kernels, num_functions, n_test]
        rmse = (mu_conditional - Y_test).pow(2).mean(dim=-1).sqrt() # [B, num_kernels, num_functions]

        return rmse


class BaseMultipleStepsGPLoss(BaseLoss):
    def __init__(
        self,
        batch_size,
        num_kernels,
        num_functions_per_kernel,
        num_splits=1,
        data_source=None, 
        name=None
    ):
        super().__init__(batch_size=batch_size, data_source=data_source, name=name)
        self.num_kernels = num_kernels
        self.num_functions_per_kernel = num_functions_per_kernel
        self.num_splits = num_splits # this control how many times we split the computation in sequence (for large batch size)
        self.loss_computer = None
        self.test_computer = _GPRMSEComputer()
        """
        we later write main computation in the loss_computer.forward method
        """

    def compute_loss(self, process, process_kwargs):
        """
        compute the loss which needs to be differentiated
        """
        #base_device = torch.device(process.device)
        #n_devices = torch.cuda.device_count()
        #if base_device.type == 'cuda' and n_devices > 1:
        #    return self._compute_loss_in_parallel(process, process_kwargs)
        #else:
        return self._compute_loss_in_sequence(process, process_kwargs, num_splits=self.num_splits)

    def validation(self, process):
        """
        compute the validation values which does not need to be differentiated (for visualization, analysis etc)

        return tuple of 2 floats, the mean and standard error of the validation values
        """
        B = min(self.batch_size, 10)
        Nk = min(self.num_kernels, 10)
        Nf = min(self.num_functions_per_kernel, 10)

        with torch.no_grad():
            base_device = torch.device(process.device)
            process_return_list = self.__execute_experiment_in_parallel(
                process,
                'validation',
                {
                    'batch_size': B,
                    'num_kernels': Nk,
                    'num_functions': Nf,
                    'num_test_points': 200
                }
            )
            rmse = self.__apply_computer_to_gp_rollout_in_parallel(self.test_computer, process_return_list, base_device) # [B, num_kernels, num_functions]
            rmse_flatten = rmse.flatten() # [B * Nk * Nf]

            rmse_mean = rmse_flatten.mean()
            rmse_stderr = torch.sqrt(rmse_flatten.var() / rmse_flatten.shape[0])
        return rmse_mean, rmse_stderr

    """
    def _compute_loss_in_parallel(self, process, process_kwargs):
        #process_return_list = self.__execute_experiment_in_parallel(process, '__call__', process_kwargs)
        assert 'num_kernels' in process_kwargs
        assert not 'device' in process_kwargs
        base_device = torch.device(process.device)
        #n_devices = torch.cuda.device_count()
        # run process(**process_kwargs) sequentially but move to individual devices
        pass_in_process_kwargs = process_kwargs.copy()
        num_kernels = pass_in_process_kwargs.pop('num_kernels')

        device_ids = _get_all_device_indices()
        kernel_batch_template = torch.empty(num_kernels, device=base_device)
        kernel_batch_template = scatter((kernel_batch_template, ), device_ids, dim=0) # kernel batch sizes on individual devices

        process_return_list = []
        for i, kbt in enumerate( kernel_batch_template ):
            process_return_list.append(
                process(num_kernels = kbt[0].shape[0], device=kbt[0].device, **pass_in_process_kwargs)
            )

        loss = self.__apply_computer_to_gp_rollout_in_parallel(self.loss_computer, process_return_list, base_device)
        # loss has shape [B, num_kernels, num_functions] (GPMI losses, GPEntropy losses) or [B, num_kernels] (PCE losses)
        
        loss = loss.mean() # average over different batch samples & prior functions
        warn_if_nan(loss, "loss")
        return loss
    """

    def _compute_loss_in_sequence(self, process, process_kwargs, num_splits):
        # first chunk kernel batch
        assert 'num_kernels' in process_kwargs
        pass_in_process_kwargs = process_kwargs.copy()
        num_kernels = pass_in_process_kwargs.pop('num_kernels')
        kernel_batch_template = torch.empty(max(num_kernels, num_splits), device=process.device).chunk(num_splits, dim=0)

        loss_list = []
        for i, kbt in enumerate( kernel_batch_template ):
            # for each kernel batch, execute experiment
            process.set_name(f"Trial_{i}")
            process_return = process(num_kernels = kbt.shape[0], **pass_in_process_kwargs)
            # then compute list for each batch
            loss_list.append( self.loss_computer(*process_return) )
        # then concatenate all loss back to one tensor
        loss = torch.cat(loss_list, dim=1)
        # loss has shape [B, num_kernels, num_functions] (GPMI losses, GPEntropy losses) or [B, num_kernels] (PCE losses)
        
        loss = loss.mean() # average over different batch samples & prior functions
        warn_if_nan(loss, "loss")
        return loss


    def __execute_experiment_in_parallel(
        self,
        process,
        method_name = '__call__',
        process_kwargs = None
    ):
        r"""
        return a list, number of list == number of devices
        each element is the tuple of process return
        """
        ### currently, the gradient is not as we want with this method
        assert 'num_kernels' in process_kwargs
        assert not 'device' in process_kwargs
        """
        base_device = torch.device(process.device)
        n_devices = torch.cuda.device_count()
        if base_device.type == 'cuda' and n_devices > 1:
            # run process(**process_kwargs) on multiple devices
            # separate on different kernels
            pass_in_process_kwargs = process_kwargs.copy()
            num_kernels = pass_in_process_kwargs.pop('num_kernels')

            device_ids = _get_all_device_indices()
            kernel_batch_template = torch.empty(num_kernels, device=base_device)
            kernel_batch_template = scatter((kernel_batch_template, ), device_ids, dim=0) # kernel batch sizes on individual devices

            process_list = process.replicate(device_ids, not torch.is_grad_enabled())
            # clone process (including design_net, gp sampler) onto individual devices
            # replicate method also sets process name
            inputs = tuple(tuple() for _ in process_list)
            kwargs_list = [
                {
                    **pass_in_process_kwargs,
                    'num_kernels': kbt_d[0].shape[0],
                    'device': kbt_d[0].device,
                } for d, kbt_d in enumerate(kernel_batch_template)
            ]
            return parallel_apply(process_list, inputs, kwargs_list, device_ids[:len(process_list)]) # run process(*input, **kwargs) on individual devices
        else:
            process.set_name("Trial_0")
            return [process(**process_kwargs)]
        """
        process.set_name("Trial_0")
        return [getattr(process, method_name)(**process_kwargs)]

    def __apply_computer_to_gp_rollout_in_parallel(self, computer, process_return_list, output_device):
        r"""
        compute loss or validation values using the given computer module
        process datasets parallelly on different devices if possible
        return computation result on the original device
        """
        n_devices = len(process_return_list)
        if n_devices == 1:
            return computer(*process_return_list[0]) # [B, num_kernels, num_functions]
        else:
            device_ids = range(n_devices)
            replicas = replicate(computer, device_ids, not torch.is_grad_enabled())
            outputs = parallel_apply(replicas, process_return_list, None, device_ids)
            return gather(outputs, output_device, dim=1) # [B, num_kernels, num_functions]

