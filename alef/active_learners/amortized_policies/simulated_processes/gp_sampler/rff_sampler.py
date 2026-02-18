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

import numpy as np
import torch
from typing import Tuple
from pyro.distributions import Normal, Delta
from typing import List, Tuple, Optional, Union

from alef.active_learners.amortized_policies.simulated_processes.gp_sampler.base_sampler import BaseSampler
from alef.active_learners.amortized_policies.global_parameters import OVERALL_VARIANCE, AL_MEAN_VARIANCE, AL_FUNCTION_VARIANCE_LOWERBOUND, AL_FUNCTION_VARIANCE_UPPERBOUND
from alef.active_learners.amortized_policies.utils.oed_primitives import prior_sample
from alef.configs.base_parameters import INPUT_DOMAIN
from alef.configs.means.pytorch_means import BaseMeanPytorchConfig, BasicZeroMeanPytorchConfig
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.enums.active_learner_amortized_policy_enums import LengthscaleDistribution

class RandomFourierFeatureSampler(BaseSampler):

    def __init__(
        self,
        kernel_config: BaseKernelPytorchConfig,
        observation_noise: float,
        overall_variance: float = OVERALL_VARIANCE,
        function_variance_interval: Tuple[float, float] = (AL_FUNCTION_VARIANCE_LOWERBOUND, AL_FUNCTION_VARIANCE_UPPERBOUND),
        mean_variance: float = AL_MEAN_VARIANCE,
        mean_config: Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]] = BasicZeroMeanPytorchConfig(batch_shape=[]),
        lengthscale_distribution: LengthscaleDistribution=LengthscaleDistribution.GAMMA,
        name: str = 'rff_gp_sampler'
    ):
        r"""
        :param kernel_config: BaseKernelPytorchConfig
        :param observation_noise: float, observation noise standard deviation
        :param overall_variance: float, overall variance of the GP (variance scale of kernel + noise variance)
            although we put this into input arguments, we actually don't use it in the code,
            please stay with the default value unless the following are also updated:
                alef.active_learners.amortized_policies.simulated_processes.*
                alef.active_learners.amortized_policies.losses.multiple_steps.gp_mi.*
                    search for 'OVERALL_VARIANCE' to see where it is used
        :param function_variance_interval: (low, up), the kernel variance is sample from this interval
        :param mean_variance: overall_variance = mean_variance + kernel_var + noise_var
                    we use noise_var = overall_variance - function_variance
                           function_variance is then distributed to mean and kernel
        :param mean_config: mean config of gp prior
        :param lengthscale_distribution: type of lengthscale sampler
        :param name: str, name of the GP sampler
        """
        super().__init__(
            kernel_config=kernel_config,
            observation_noise=observation_noise,
            overall_variance=overall_variance,
            function_variance_interval=function_variance_interval,
            mean_variance=mean_variance,
            mean_config=mean_config,
            lengthscale_distribution=lengthscale_distribution,
            name=name
        )
        self.register_buffer( "_num_fourier_samples", torch.tensor(100) )
        assert self.kernel.has_fourier_feature
        self.draw_parameter(draw_hyper_prior=False)

    def draw_parameter(
        self,
        num_priors: int=1,
        num_functions: int=1,
        draw_hyper_prior: bool=False,
        input_domain: Tuple[float, float]=INPUT_DOMAIN,
        *,
        shift_mean: bool=False,
        flip_center: bool=False,
        center_length_ratio: float=0.1,
        mask: Optional[Union[np.ndarray, torch.Tensor]]=None,
        sample_executor = prior_sample,
    ):
        """
        draw hyper-priors, f, or noise of y|f
        
        :param num_priors: batch size of kernel hyperparameters (i.e. num of kernels).
        :param num_functions: number of functional sample given a GP prior.
        :param draw_hyper_prior: whether to draw parameters from hyper-priors.
        :param input_domain: the interval to compute the mean.
        :param shift_mean: whether to shift the mean of the sampled functions (GP function in a small window doesn't always average to prior mean).
        :param flip_center: whether to ensure central area is positive or not
        :param center_length_ratio: width of center comparing to input domain width (per dimension, i.e. the volume of center is ratio**D * volume of input domain)
        :param mask: [num_priors, num_functions, D] if given, can mask out dimension per function (reduce dimension)
        :param sample_executor: sample, or sample with type, for Fourier features. See alef.active_learners.amortized_policies.utils.oed_primitives
        """
        assert num_functions > 0
        if not mask is None:
            assert mask.shape[0] in [1, num_priors]
            assert mask.shape[1] in [1, num_functions]
            assert mask.shape[2] in [1, self._original_kernel.input_dimension]
        super().draw_parameter(num_priors=num_priors, draw_hyper_prior=draw_hyper_prior, input_domain=input_domain, mask=mask)
        self.kernel.sample_fourier_features(self._num_fourier_samples, num_functions, sample_executor=sample_executor, sample_name=f'{self.name}.BayesianLinearModel')
        max_interval = self.max_interval( # [Nk, D, 2], this interval matters only if we flip the center
            input_domain=input_domain,
            length_ratio=center_length_ratio
        ).unsqueeze(-3) # [Nk, 1, D, 2]
        self._f_map = self.kernel.bayesian_linear_model(x_expanded_already=True, input_domain=input_domain, shift_mean=shift_mean, flip_center=flip_center, central_interval=(max_interval[..., 0], max_interval[..., 1]), mask=self._mask)

    @property
    def bayesian_linear_model(self):
        return self._f_map

    @property
    def num_priors(self):
        return self._f_map.num_kernels

    @property
    def num_functions(self):
        return self._f_map.num_functions

    @property
    def num_fourier_samples(self):
        return self._f_map.num_fourier_samples

    def mean(self, x_data: torch.Tensor):
        """
        compute GP mean(x_data), return in raw torch type.

        :param x_data: torch.Tensor, size [B, Nk, Nf, n, d]

        :return: torch.Tensor, size [B, Nk, Nf, n]
        """
        if self._mask is None:
            gp_mean = torch.concat(
                [m(x_data[:, i, None, ...]) for i, m in enumerate(self.mean_list)], dim=1
            )
        else:
            gp_mean = torch.concat(
                [
                    m( x_data[:, i, None, ...], mask=self._mask[None, i, None] )
                    for i, m in enumerate(self.mean_list)
                ], dim=1
            )
        return gp_mean

    def f_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ), return in raw torch type 

        :param x_data: torch.Tensor, size [B, Nk, Nf, n, d]

        :return: pyro.distributions.Delta, size [B, Nk, Nf, n]
        """
        # self._mask: [Nk, Nf, D]
        mask = None if self._mask is None else self._mask[None,...] # [1, Nk, Nf, D]
        x_torch = self._mask_x(x_data, mask=mask)
        gp_mean = self.mean(x_torch)
        # note: bayesian_linear_model was masked when sampled
        f = gp_mean + self.bayesian_linear_model(x_torch) # [batch_size, num_priors, num_functions]
        return Delta(f)

    def y_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ) + noise_dist, return in raw torch type 

        :param x_data: torch.Tensor, size [B, Nk, Nf, n, d]

        :return: pyro.distributions.Normal, size [B, Nk, Nf, n]
        """
        # self._mask: [Nk, Nf, D]
        mask = None if self._mask is None else self._mask[None,...] # [1, Nk, Nf, D]
        x_torch = self._mask_x(x_data, mask=mask)
        gp_mean = self.mean(x_torch)
        # note: bayesian_linear_model was masked when sampled
        f = gp_mean + self.bayesian_linear_model(x_torch) # [batch_size, num_priors, num_functions]

        var_shape = [1] * f.dim()
        var_shape[1] = f.shape[1]
        var = self.noise_variance.reshape(var_shape)
        noise_std = torch.sqrt( var ) * torch.ones_like(f)

        return Normal(
            f,
            noise_std
        )
