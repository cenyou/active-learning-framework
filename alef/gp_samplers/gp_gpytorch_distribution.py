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
import numpy as np
import torch
import gpytorch
from copy import deepcopy
from pyro.distributions import Gamma, MultivariateNormal, Uniform, Delta
from typing import List, Union

from alef.enums.environment_enums import GPFramework
from alef.gp_samplers.base_distribution import BaseDistribution
from alef.kernels.pytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from alef.means.pytorch_means.pytorch_mean_factory import PytorchMeanFactory

from alef.configs.base_parameters import NOISE_VARIANCE_LOWER_BOUND
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import BaseElementaryKernelPytorchConfig
from alef.configs.means.pytorch_means import BaseMeanPytorchConfig, BasicZeroMeanPytorchConfig
from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)

class GPTorchDistribution(BaseDistribution):

    def __init__(
        self,
        kernel_config: BaseElementaryKernelPytorchConfig,
        observation_noise: float,
        *,
        expected_observation_noise: float=EXPECTED_OBSERVATION_NOISE,
        mean_config: Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]] = BasicZeroMeanPytorchConfig(batch_shape=[]),
        device=torch.device('cpu')
    ):
        r"""
        :param kernel_config: BaseKernelPytorchConfig
        :param observation_noise: float, observation noise standard deviation
        :param expected_observation_noise: float, mean of noise prior (noise in standard deviation)
        :param mean_config: mean config of gp prior
        :param device: torch.device, device to run the computation
        """
        super().__init__(GPFramework.GPYTORCH)
        self._kernel_config = kernel_config
        self._mean_config = mean_config
        self._device = device
        self._original_kernel = PytorchKernelFactory.build(kernel_config).to(device)
        self.kernel = deepcopy(self._original_kernel)
        self.kernel_list = [self.kernel.kernel]
        self.mean_list = None

        self._original_noise_variance = torch.tensor(math.pow(observation_noise, 2.0), device=device)
        self.noise_variance = deepcopy(self._original_noise_variance)
        self.noise_prior = Gamma(torch.tensor(1.0, device=device), 1 / torch.pow(torch.tensor(expected_observation_noise, device=device), 2.0))

    def set_gp_mean(
        self,
        mean_config: Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]],
        num_priors: int=1
    ):
        """
        please do not assign batch size in the configs, our loss functions assume no batch in each mean function
        if needed, assign one config per priors (len(men_config)=num_priors)
        """
        out = []
        if hasattr(mean_config, '__len__'):
            assert len(mean_config)==1 or len(mean_config)==num_priors
            configs = mean_config if len(mean_config)==num_priors else list(mean_config)*num_priors
        else:
            configs = [mean_config]*num_priors
        for m_config in configs:
            m_func = PytorchMeanFactory.build(m_config)
            m_func.to(self._device)
            # TODO: ideally we should initiate the mean directly in our target device or move device for only once
            out.append(m_func)
        return out

    @property
    def input_dimension(self):
        return self._original_kernel.input_dimension

    def draw_parameter(
        self,
        num_priors: int=1,
        num_functions: int=1,
        draw_hyper_prior: bool=False
    ):
        """
        draw hyper-priors, f, or noise of y|f
        
        arguments:

        num_priors: batch size of kernel hyperparameters (i.e. num of kernels).
        num_functions: number of functional sample given a GP prior.
        draw_hyper_prior: whether to draw parameters from hyper-priors.
        
        """
        assert num_priors > 0
        assert num_functions > 0
        self._num_functions = num_functions
        self.mean_list = self.set_gp_mean(self._mean_config, num_priors=num_priors)
        if num_priors > 1:
            self.kernel.kernel = self._original_kernel.kernel.expand_batch((num_priors, ))
            self.noise_variance = self._original_noise_variance.expand((num_priors, ))
        else: # reset to original
            self.kernel = deepcopy(self._original_kernel)
            self.noise_variance = deepcopy(self._original_noise_variance)

        if draw_hyper_prior:
            kernel = gpytorch.kernels.AdditiveKernel(self.kernel.kernel).pyro_sample_from_prior()
            self.kernel.kernel = kernel.kernels[0]

            self.noise_variance = self.noise_prior(self.noise_variance.size()) + torch.tensor(NOISE_VARIANCE_LOWER_BOUND, device=self._device)

        if num_priors > 1:
            self.kernel_list = [self.kernel.kernel[i] for i in range(num_priors)]
        else:
            self.kernel_list = [self.kernel.kernel]

    def show_parameter(self):
        for name, param, constraint in self.kernel.named_parameters_and_constraints():
            if constraint is not None:
                print(f"Parameter name: {name:55} value = {constraint.transform(param)}")
            else:
                print(f"Parameter name: {name:55} value = {param}")
        print(f'observation noise variance: {self.noise_variance}, type {self.noise_variance.dtype}, prior {self.noise_prior}')

    def mean(self, x_data: torch.Tensor):
        """
        compute GP mean(x_data), return in raw torch type 

        Arguments:
        x_data: Input array with shape (n,d) or (num_priors, num_functions_per_prior, n, d) where d is the input dimension and n the number of training points
        """
        x_torch = x_data
        N = x_torch.size(dim=-2)
        assert len(x_torch.shape) in [2, 4]
        if len(x_torch.shape) == 4:
            assert x_torch.shape[1] == self._num_functions

            gp_mean = torch.concat([
                m(x_torch[None, i, ...])
                for i, m in enumerate(self.mean_list)
            ], dim=0)
        else:
            gp_mean = self.mean_list[0](x_torch)

        return gp_mean

    def mean_numpy(self, x_data: np.ndarray):
        with torch.no_grad():
            return self.mean(
                torch.from_numpy(x_data).to(torch.get_default_dtype())
            ).cpu().numpy()

    def f_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ), return in raw torch type 

        Arguments:
        x_data: Input array with shape (n,d) or (num_priors, num_functions_per_prior, n, d) where d is the input dimension and n the number of training points
        """
        x_torch = x_data
        N = x_torch.size(dim=-2)
        assert len(x_torch.shape) in [2, 4]

        if len(x_torch.shape) == 4:
            assert x_torch.shape[1] == self._num_functions

            K = torch.cat([
                kernel(x_torch[None, i, ...]).to_dense().to(self._device)
                for i, kernel in enumerate(self.kernel_list)
            ], dim=0)
        else:
            K = self.kernel(x_torch).to_dense().to(self._device)

        gp_mean = self.mean(x_data)

        if (torch.linalg.eigvals(K).real>=0).all():
            return MultivariateNormal( gp_mean, K )
        else:
            logger.info('Kernel gram matrix is not positive definite, add small diagonal values')
            return MultivariateNormal(
                gp_mean,
                K + torch.tensor(NOISE_VARIANCE_LOWER_BOUND, device=self._device) * torch.eye(N, device=self._device)
            ) # NOISE_VARIANCE_LOWER_BOUND*K.diagonal(dim1=-2, dim2=-1).diag()

    def y_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ) + noise_dist, return in raw torch type 

        Arguments:
        x_data: Input array with shape (n,d) or (num_priors, num_functions_per_prior, n, d) where d is the input dimension and n the number of training points
        """
        x_torch = x_data
        N = x_torch.size(dim=-2)
        assert len(x_torch.shape) in [2, 4]
        
        if len(x_torch.shape) == 4:
            assert x_torch.shape[1] == self._num_functions

            K = torch.cat([
                kernel(x_torch[None, i, ...]).to_dense().to(self._device) +\
                    self.noise_variance[i] * torch.eye(N, device=self._device)
                for i, kernel in enumerate(self.kernel_list)
            ], dim=0)
        else:
            K = self.kernel(x_torch).to_dense().to(self._device) +\
                self.noise_variance * torch.eye(N, device=self._device)

        gp_mean = self.mean(x_data)

        return MultivariateNormal( gp_mean, K )

    def sample_f(self, x_data: np.ndarray):
        """
        sample f from GP( mean(x_data), kernel(x_data) )

        Arguments:
        x_data: Input array with shape (n,d) or (num_priors, num_functions_per_prior, n, d) where d is the input dimension and n the number of training points
        """
        return self.f_sampler(
            torch.from_numpy(x_data).to(torch.get_default_dtype())
        ).sample().cpu().numpy()

    def sample_y(self, x_data: np.ndarray):
        """
        sample y from GP( mean(x_data), kernel(x_data) ) + noise_dist(x_data)

        Arguments:
        x_data: Input array with shape (n,d) or (num_priors, num_functions_per_prior, n, d) where d is the input dimension and n the number of training points
        """
        return self.y_sampler(
            torch.from_numpy(x_data).to(torch.get_default_dtype())
        ).sample().cpu().numpy()

class NormalizedGPTorchDistribution(GPTorchDistribution):
    def __init__(
        self,
        kernel_config: BaseElementaryKernelPytorchConfig,
        observation_noise: float,
        overall_variance: float=1.0001,
        *,
        mean_config: Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]] = BasicZeroMeanPytorchConfig(batch_shape=[]),
        device=torch.device('cpu')
    ):
        r"""
        :param kernel_config: BaseKernelPytorchConfig
        :param observation_noise: float, observation noise standard deviation
        :param overall_variance: float, overall variance of the GP (variance scale of kernel + noise variance)
        :param device: torch.device, device to run the computation
        """
        super().__init__(kernel_config, observation_noise, expected_observation_noise=EXPECTED_OBSERVATION_NOISE, mean_config=mean_config, device=device)
        self._overall_variance = torch.tensor(overall_variance, device=device)

    def draw_parameter(
        self,
        num_priors: int=1,
        num_functions: int=1,
        draw_hyper_prior: bool=False
    ):
        """
        draw hyper-priors, f, or noise of y|f
        
        arguments:

        num_priors: batch size of kernel hyperparameters (i.e. num of kernels).
        num_functions: number of functional sample given a GP prior.
        draw_hyper_prior: whether to draw parameters from hyper-priors.

        """
        assert num_priors > 0
        assert num_functions > 0
        self._num_functions = num_functions
        self.mean_list = self.set_gp_mean(self._mean_config, num_priors=num_priors)
        if num_priors > 1:
            self.kernel.kernel = self._original_kernel.kernel.expand_batch((num_priors, ))
            self.noise_variance = self._original_noise_variance.expand((num_priors, ))
        else: # reset to original
            self.kernel = deepcopy(self._original_kernel)
            self.noise_variance = deepcopy(self._original_noise_variance)

        if draw_hyper_prior:
            kernel = gpytorch.kernels.AdditiveKernel(self.kernel.kernel).to_random_module()
            for prior_name, module, prior, closure, setting_closure in kernel.named_priors():
                #value = prior.sample(closure(module).shape)
                if 'outputscale' in prior_name or 'variance' in prior_name:
                    low = self._overall_variance * 0.5
                    high = self._overall_variance * 0.99
                elif 'offset' in prior_name:
                    low = - 1.0
                    high = 1.0
                else:
                    low = 0.05
                    high = 1.0
                value = Uniform( low, high * torch.ones_like(closure(module)) ).sample()
                setting_closure(module, value)
            self.kernel.kernel = kernel.kernels[0]

            self.noise_variance = self._overall_variance * torch.ones_like(self.noise_variance) - self.kernel.prior_scale.pow(2)

        if num_priors > 1:
            self.kernel_list = [self.kernel.kernel[i] for i in range(num_priors)]
        else:
            self.kernel_list = [self.kernel.kernel]



if __name__=="__main__":
    from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
    from matplotlib import pyplot as plt
    D = 1
    kernel_config = RBFWithPriorPytorchConfig(
            input_dimension=D
        )

    dist = GPTorchDistribution(kernel_config, 0.1)
    dist.draw_parameter(1, 1, True)

    x = np.sort( np.random.standard_normal([10, D]), axis=0)
    dist.draw_parameter()
    #print(dist.kernel.state_dict())
    #print(dist.noise_variance)
    print(dist.f_sampler(torch.tensor(x, dtype=torch.get_default_dtype() )))
    print(dist.f_sampler(torch.tensor(x, dtype=torch.get_default_dtype() )).sample().unsqueeze(-1))
    #print(dist.y_dist(torch.tensor(x)))
    #print(dist.sample_f(x))
    #print(dist.sample_y(x))
    fig, axs = plt.subplots()
    for _ in range(10):
        axs.plot(x, dist.sample_y(x), '-')
    plt.show()

