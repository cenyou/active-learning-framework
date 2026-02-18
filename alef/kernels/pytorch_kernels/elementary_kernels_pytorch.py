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

from abc import abstractmethod
from typing import Tuple, Union, Sequence
import gpytorch
import torch
import math
from pyro import distributions as dist
from gpytorch.constraints import Positive
from .bayes_linear_model import BayesianLinearModel
from alef.configs.base_parameters import INPUT_DOMAIN, CENTRAL_DOMAIN, NUMERICAL_POSITIVE_LOWER_BOUND, NUMERICAL_POSITIVE_LOWER_BOUND_IN_LOG
from alef.kernels.pytorch_kernels.customized_gpytorch_kernels import PeriodicKernel
from alef.utils.torch_sample_executors import torch_sample, pyro_sample


class BaseElementaryKernelPytorch(gpytorch.kernels.Kernel):
    
    has_fourier_feature = False
    
    def __init__(
        self,
        input_dimension: int,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        self.input_dimension = input_dimension
        self.active_dimension = active_dimension
        self.active_on_single_dimension = active_on_single_dimension

        if active_on_single_dimension:
            self.name = name + "_on_" + str(active_dimension)
            super().__init__(active_dims=torch.tensor([active_dimension]))
            self.num_active_dimensions = 1
        else:
            self.name = name
            super().__init__()
            self.num_active_dimensions = input_dimension
        self.kernel = None

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, *, mask=None, **params):
        """
        :param x1: torch.Tensor of size [*batch_shape, N, D]
        :param x2: None or torch.Tensor of size [*batch_shape, M, D]
        :param diag: bool, compute the whole kernel or just the diag. If True, ensure x1 == x2 or x2 = None.
        :param last_dim_is_batch: bool. If True, treat the last dimension of `x1` and `x2` as another batch dimension.
        :param mask: None or torch.Tensor of size [*batch_shape, D], allow one to compute different dimension at different batch instance
        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x D x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x D x N`
        """
        if not mask is None:
            assert not last_dim_is_batch or (mask.shape[-1]==1 and torch.all(mask==1)), "no dimension to be masked (last_dim_is_batch take only one dim)"
            # note: replacing by 0 works for stationary kernels
            # however this is not a valid masking for general kernels
            x1_ = x1.masked_fill(mask.unsqueeze(-2)==0, 0)
            x2_ = None if x2 is None else x2.masked_fill(mask.unsqueeze(-2)==0, 0)
        else:
            x1_, x2_ = x1, x2
        return self.kernel.forward(x1_, x2_, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)

    def get_input_dimension(self):
        return self.input_dimension

    @abstractmethod
    def get_parameters_flattened(self) -> torch.tensor:
        raise NotImplementedError

    @property
    def prior_scale(self): # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        D = self.get_input_dimension()
        dumpy_point = torch.ones([1, D], device=self.kernel.device)

        var_scale = self(dumpy_point, diag=True).to_dense().squeeze(-1) # size is kernel batch size
        std_scale = torch.sqrt(var_scale)
        return std_scale

    @prior_scale.setter
    def prior_scale(self, value):
        raise NotImplementedError

    ### the followings are for fourier features
    ### the followings are for fourier features
    ### the followings are for fourier features
    def _sample_feature_frequencies(self, L: int, num_functions: int=1, sample_executor=torch_sample, sample_name='rff_omega'):
        r"""
        :param L: int, number of fourier features, see [1] page 3, this is the D in the paper
        :param num_functions: int, number of functions to sample (for batch purpose)
        :param sample_executor: callable, a function to sample (see below)
        :param sample_name: str, name of the sample, (see below)

        we would define a distribution, and then sample by sample_executor(sample_name, distribution)

        return:
        torch.Tensor of shape [..., num_functions, input_dim, L], see [1] page 3, this is the omega in the paper

        [1] Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
        """
        if self.has_fourier_feature:
            raise NotImplementedError
        else:
            pass

    def _sample_feature_weights(self, L: int, num_functions: int=1, sample_executor=torch_sample, sample_name='BLM.weight'):
        r"""
        :param L: int, number of fourier features, see [1] page 3, this is the D in the paper
        :param num_functions: int, number of functions to sample (for batch purpose)
        :param sample_executor: callable, a function to sample (see below)
        :param sample_name: str, name of the sample, (see below)

        we would define a distribution, and then sample by sample_executor(sample_name, distribution)

        return:

        torch.Tensor of shape [..., num_functions, L], see [1] page 3 and [2] page 3, this is the w in paper [2]
        
        notice that, if we sample w as in paper [2], then cov(f(x), f(x')) = 1 * <phi(x), phi(x') >

        [1] Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
        [2] Wilson et al., ICML 2020, Efficiently Sampling Functions from Gaussian Process Posteriors
        """
        if self.has_fourier_feature:
            raise NotImplementedError
        else:
            pass

    def sample_fourier_features(self, L: int, num_functions: int=1, sample_executor=torch_sample, sample_name='BLM'):
        r"""
        :param L: int, number of fourier features, see [1] page 3, this is the D in the paper
        :param num_functions: int, number of functions to sample (for batch purpose)
        :param sample_executor: callable, a function to sample (see below)
        :param sample_name: str, name of the sample, (see below)

        we would define distributions of Bayesian linear model's parameters,
        and then sample each parameter by sample_executor(sample_name, distribution)

        sample random fourier features
        see [2], page 3 for details

        [1] Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
        [2] Wilson et al., ICML 2020, Efficiently Sampling Functions from Gaussian Process Posteriors
        """
        if not self.has_fourier_feature:
            raise NotImplementedError
        else:
            self._num_functions = num_functions
            self._num_fourier_samples = L

            self._omega = self._sample_feature_frequencies(
                self._num_fourier_samples,
                num_functions,
                sample_executor = sample_executor,
                sample_name = sample_name + '.omega'
            ) # [..., num_functions, input_dim, L]
            bias_size = self._omega.shape[:-2] + (self._num_fourier_samples, )
            bias_dist = dist.Uniform(
                torch.zeros(bias_size).to(self._omega.device),
                2 * math.pi * torch.ones(bias_size).to(self._omega.device)
            )
            self._bias = sample_executor(sample_name + '.bias', bias_dist) # [..., num_functions, L]
            #self._bias = 2 * math.pi * torch.rand(size=bias_size).to(self._omega.device) # [..., num_functions, L]
            self._weight = self._sample_feature_weights(
                self._num_fourier_samples,
                num_functions,
                sample_executor = sample_executor,
                sample_name = sample_name + '.weight'
            ) # [..., num_functions, L]

    def bayesian_linear_model(self, x_expanded_already: bool=False, input_domain=INPUT_DOMAIN, shift_mean: bool=False, flip_center: bool=False, central_interval=CENTRAL_DOMAIN, mask=None):
        r"""
        :param x_expanded_already: bool, whether the input x is already expanded to
                [B, num_kernels, num_functions, ..., D] or [B, 1, 1, ..., D]
        :param input_domain: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]], the interval to calculate the mean of the function
                if torch.Tensor is provided, please ensure the shape can be broadcasted into [num_kernels, num_functions, D]
        :param shift_mean: bool, whether to shift the mean of the function to zero (GP function in a small window doesn't always average to prior mean)
        :param flip_center: bool, weather center of the interval is flip to positive
        :param central_interval: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]], the center interval if flip_center==True
                if torch.Tensor is provided, please ensure the shape can be broadcasted into [num_kernels, num_functions, D]
        :param mask: [num_kernels, num_functions, D] if given, can mask out dimensions
        
        return Bayesian linear model f as a function
        see [1] and page 3 of [2] for details

        [1] Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
        [2] Wilson et al., ICML 2020, Efficiently Sampling Functions from Gaussian Process Posteriors
        """
        if not self.has_fourier_feature:
            raise NotImplementedError
        else:
            assert_msg = 'no fourier samples, please first make samples (method \'sample_fourier_features(number_of_fourier_features)\')'
            assert hasattr(self, '_weight'), assert_msg
            assert hasattr(self, '_omega'), assert_msg
            assert hasattr(self, '_bias'), assert_msg
            return BayesianLinearModel(self._omega, self._bias, self._weight, x_expanded_already, input_domain=input_domain, shift_mean=shift_mean, flip_center=flip_center, central_interval=central_interval, mask=mask)

class RBFKernelPytorch(BaseElementaryKernelPytorch):

    has_fourier_feature = True

    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior)
            self.kernel = gpytorch.kernels.ScaleKernel(rbf_kernel, outputscale_prior=outputscale_prior)
        else:
            rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(rbf_kernel)

        if not hasattr(base_lengthscale, '__len__'):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, f'number of base_lengthscale \'{lengthscales.shape[-1]}\' does not match input dimension \'{self.num_active_dimensions}\''

        rbf_kernel.lengthscale = lengthscales
        self.kernel.outputscale = torch.tensor(base_variance)

    @property
    def prior_scale(self): # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale) #  # size is kernel batch size

    @prior_scale.setter
    def prior_scale(self, value: torch.Tensor):
        kernel = gpytorch.kernels.AdditiveKernel(self.kernel).to_random_module()
        kernel.kernels[0].outputscale = value.pow(2)
        self.kernel = kernel.kernels[0]

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, '__len__'):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        return torch.concat((lengthscales_flattened, variance_flattened))

    ### the followings are for fourier features
    ### the followings are for fourier features
    ### the followings are for fourier features
    def _sample_feature_frequencies(self, L: int, num_functions: int=1, sample_executor=torch_sample, sample_name='BLM.omega'):
        # lengthscale shape: [1, input_dim] or [batch_size, 1, input_dim]
        lghscale = self.kernel.base_kernel.lengthscale
        device = lghscale.device
        if len(lghscale.shape) > 2:
            kernel_batch_size = lghscale.shape[:-2]
        else:
            kernel_batch_size = torch.Size([1])
        omega_transpose_size = kernel_batch_size + (num_functions, self.num_active_dimensions, L)
        omega_dist = dist.Normal(
            torch.zeros(omega_transpose_size, device=device),
            scale=1/lghscale[..., None, :, :].transpose(-1, -2).expand(omega_transpose_size)
        )
        return sample_executor(sample_name, omega_dist)
        #omega = torch.randn(
        #    size= kernel_batch_size + (num_functions, L, self.num_active_dimensions),
        #    device = device
        #).div(lghscale[..., None, :, :]) # ~ N(0, 1 / lengthscale**2)
        #return torch.transpose(omega, -1, -2)

    def _sample_feature_weights(self, L: int, num_functions: int=1, sample_executor=torch_sample, sample_name='BLM.weight'):
        # outputscale shape: no shape (scalar) or [batch_size, ]
        scale = self.kernel.outputscale
        device = scale.device
        if len(scale.shape) >= 1:
            kernel_batch_size = scale.shape
        else:
            kernel_batch_size = torch.Size([1])
        weight_dist = dist.Normal(
            torch.zeros(kernel_batch_size + (num_functions, L), device=device),
            scale=torch.sqrt(scale.reshape(kernel_batch_size + (1, 1)))
        )
        return sample_executor(sample_name, weight_dist)
        #weights = torch.sqrt(scale.reshape(kernel_batch_size + (1, 1))) * \
        #            torch.randn(size=kernel_batch_size + (num_functions, L), device=device) # ~ N(0, kernel_variance)
        #return weights

class Matern52KernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            matern_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior
            )
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel, outputscale_prior=outputscale_prior)
        else:
            matern_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel)

        if not hasattr(base_lengthscale, '__len__'):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, f'number of base_lengthscale \'{lengthscales.shape[-1]}\' does not match input dimension \'{self.num_active_dimensions}\''

        matern_kernel.lengthscale = lengthscales
        self.kernel.outputscale = torch.tensor(base_variance)

    @property
    def prior_scale(self): # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale) #  # size is kernel batch size

    @prior_scale.setter
    def prior_scale(self, value: torch.Tensor):
        kernel = gpytorch.kernels.AdditiveKernel(self.kernel).to_random_module()
        kernel.kernels[0].outputscale = value.pow(2)
        self.kernel = kernel.kernels[0]

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, '__len__'):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        return torch.concat((lengthscales_flattened, variance_flattened))


class Matern32KernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            matern_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior
            )
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel, outputscale_prior=outputscale_prior)
        else:
            matern_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(matern_kernel)

        if not hasattr(base_lengthscale, '__len__'):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, f'number of base_lengthscale \'{lengthscales.shape[-1]}\' does not match input dimension \'{self.num_active_dimensions}\''

        matern_kernel.lengthscale = lengthscales
        self.kernel.outputscale = torch.tensor(base_variance)

    @property
    def prior_scale(self): # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale) #  # size is kernel batch size

    @prior_scale.setter
    def prior_scale(self, value: torch.Tensor):
        kernel = gpytorch.kernels.AdditiveKernel(self.kernel).to_random_module()
        kernel.kernels[0].outputscale = value.pow(2)
        self.kernel = kernel.kernels[0]

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, '__len__'):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        return torch.concat((lengthscales_flattened, variance_flattened))


class PeriodicKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        base_period: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        period_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_period, b_period = period_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            period_prior = gpytorch.priors.GammaPrior(a_period, b_period)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            periodic_kernel = PeriodicKernel(
                ard_num_dims=self.num_active_dimensions, period_length_prior=period_prior, lengthscale_prior=lengthscale_prior
            )
            # periodic_kernel.register_prior(
            #    "lengthscale_prior",
            #    lengthscale_prior,
            #    lambda m: torch.sqrt(m.lengthscale) / 2.0,
            #    lambda m, v: m._set_lengthscale(torch.square(v) * 4.0),
            # )
            self.kernel = gpytorch.kernels.ScaleKernel(periodic_kernel, outputscale_prior=outputscale_prior)
        else:
            periodic_kernel = PeriodicKernel(ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(periodic_kernel)

        if not hasattr(base_lengthscale, '__len__'):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, f'number of base_lengthscale \'{lengthscales.shape[-1]}\' does not match input dimension \'{self.num_active_dimensions}\''

        periods = torch.full((1, self.num_active_dimensions), base_period)
        periodic_kernel.lengthscale = lengthscales
        periodic_kernel.period_length = periods
        self.kernel.outputscale = base_variance

    @property
    def prior_scale(self): # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale) #  # size is kernel batch size

    @prior_scale.setter
    def prior_scale(self, value: torch.Tensor):
        kernel = gpytorch.kernels.AdditiveKernel(self.kernel).to_random_module()
        kernel.kernels[0].outputscale = value.pow(2)
        self.kernel = kernel.kernels[0]

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, '__len__'):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        period_flattened = torch.flatten(self.kernel.base_kernel.period_length)

        return torch.concat((lengthscales_flattened, variance_flattened, period_flattened))


class RQKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: Union[float, Sequence[float]],
        base_variance: float,
        base_alpha: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        alpha_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        a_alpha, b_alpha = alpha_prior_parameters
        a_variance, b_variance = variance_prior_parameters
        if add_prior:
            lengthscale_prior = gpytorch.priors.GammaPrior(a_lengthscale, b_lengthscale)
            alpha_prior = gpytorch.priors.GammaPrior(a_alpha, b_alpha)
            outputscale_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            rq_kernel = gpytorch.kernels.RQKernel(ard_num_dims=self.num_active_dimensions, lengthscale_prior=lengthscale_prior)
            rq_kernel.register_prior(
                "alpha_prior",
                alpha_prior,
                lambda m: m.alpha,
                lambda m, v: m.initialize(raw_alpha=m.raw_alpha_constraint.inverse_transform(torch.to_tensor(v))),
            )
            self.kernel = gpytorch.kernels.ScaleKernel(rq_kernel, outputscale_prior=outputscale_prior)
        else:
            rq_kernel = gpytorch.kernels.RQKernel(ard_num_dims=self.num_active_dimensions)
            self.kernel = gpytorch.kernels.ScaleKernel(rq_kernel)

        if not hasattr(base_lengthscale, '__len__'):
            lengthscales = torch.full((1, self.num_active_dimensions), base_lengthscale)
        else:
            lengthscales = torch.tensor(base_lengthscale).squeeze().unsqueeze(-2)
            assert lengthscales.shape[-1] == self.num_active_dimensions, f'number of base_lengthscale \'{lengthscales.shape[-1]}\' does not match input dimension \'{self.num_active_dimensions}\''

        rq_kernel.lengthscale = lengthscales
        rq_kernel.alpha = torch.tensor(base_alpha)
        self.kernel.outputscale = torch.tensor(base_variance)

    @property
    def prior_scale(self): # this would work for matern family, RQ kernel, linear kernel, spectral mixture kernel
        return torch.sqrt(self.kernel.outputscale) #  # size is kernel batch size

    @prior_scale.setter
    def prior_scale(self, value: torch.Tensor):
        kernel = gpytorch.kernels.AdditiveKernel(self.kernel).to_random_module()
        kernel.kernels[0].outputscale = value.pow(2)
        self.kernel = kernel.kernels[0]

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        lengthscales_flattened = torch.flatten(self.kernel.base_kernel.lengthscale)
        if hasattr(self.kernel.outputscale, '__len__'):
            variance_flattened = torch.flatten(self.kernel.outputscale)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.outputscale]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        alpha_flattened = torch.flatten(torch.tensor([self.kernel.base_kernel.alpha]))
        return torch.concat((lengthscales_flattened, variance_flattened, alpha_flattened))


class LinearKernelPytorch(BaseElementaryKernelPytorch):
    def __init__(
        self,
        input_dimension: int,
        base_variance: float,
        base_offset: float,
        add_prior: bool,
        variance_prior_parameters: Tuple[float, float],
        offset_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name, **kwargs)
        a_variance, b_variance = variance_prior_parameters
        a_offset, b_offset = offset_prior_parameters
        if add_prior:
            variance_prior = gpytorch.priors.GammaPrior(a_variance, b_variance)
            self.kernel = gpytorch.kernels.LinearKernel(num_dimensions=self.num_active_dimensions, variance_prior=variance_prior)
        else:
            self.kernel = gpytorch.kernels.LinearKernel(num_dimensions=self.num_active_dimensions)
        self.kernel.variance = torch.tensor(base_variance)

        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        offset_constraint = Positive()

        self.register_constraint("raw_offset", offset_constraint)

        if add_prior:
            offset_prior = gpytorch.priors.GammaPrior(a_offset, b_offset)
            self.register_prior(
                "offset_prior",
                offset_prior,
                lambda m: m.offset,
                lambda m, v: m._set_offset(v),
            )

        self.offset = base_offset

    @property
    def offset(self):
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value):
        return self._set_offset(value)

    def _set_offset(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, *, mask=None, **params):
        assert not last_dim_is_batch
        if not mask is None:
            assert not last_dim_is_batch or (mask.shape[-1]==1 and torch.all(mask==1)), "no dimension to be masked (last_dim_is_batch take only one dim)"
            # note: replacing by 0 works for the linear kernel because we compute inner(x1, x2)
            # however this is not a valid masking for general kernels
            x1_ = x1.masked_fill(mask.unsqueeze(-2)==0, 0)
            x2_ = x2
        else:
            x1_, x2_ = x1, x2
        K = self.kernel.forward(x1_, x2_, diag, last_dim_is_batch, **params) + self.offset
        return K

    def get_parameters_flattened(self, sqrt_variance=True) -> torch.tensor:
        offset_flattened = torch.flatten(torch.tensor([self.offset]))
        if hasattr(self.kernel.variance, '__len__'):
            variance_flattened = torch.flatten(self.kernel.variance)
        else:
            variance_flattened = torch.flatten(torch.tensor([self.kernel.variance]))

        if sqrt_variance:
            variance_flattened = torch.sqrt(variance_flattened)

        return torch.concat((offset_flattened, variance_flattened))


if __name__ == "__main__":
    linear_kernel = LinearKernelPytorch(3, 1.0, 1.0, True, (1.0, 1.0), (1.0, 1.0), False, 0, "Linear")
    print(linear_kernel.offset)
    X = torch.randn((10, 3))
    K = linear_kernel.forward(X, X, diag=True)
    print(X.numpy())
    print(K.detach().numpy())

    rbf_kernel = RBFKernelPytorch(2, [0.2, 0.4], 1.0, False,(1.0, 1.0), (1.0, 1.0), False, 0, "RBF")
    print(rbf_kernel.kernel.outputscale)
    print(rbf_kernel.kernel.base_kernel.lengthscale)

    from matplotlib import pyplot as plt
    Q = 5
    D = 1
    num_func = 2
    num_gram = 1000
    kernel = RBFKernelPytorch(D, 0.2, 1.0, False, (1, 1), (1, 1), False, 0, "RBF")
    print(kernel.get_parameters_flattened())
    kernel.sample_fourier_features(500, num_func)
    f = kernel.bayesian_linear_model(shift_mean=True, input_domain=(0, 1))

    X = torch.rand([num_gram, D])
    Y_rff = f(X).cpu().detach().squeeze().T
    Y_gp = torch.distributions.MultivariateNormal(
        torch.zeros(num_gram),
        covariance_matrix=kernel(X).to_dense() + 0.0004 * torch.eye(num_gram)
    ).sample([num_func]).cpu().detach()
    print(X.shape, Y_rff.shape, Y_gp.shape)

    func_means = f.mean([0, 1]).detach().cpu().numpy()

    fig, ax = plt.subplots(1, 2)
    ax[0].axhline(0, color='k')
    for i in range(num_func):
        ax[0].plot(X, Y_rff[i], '.', color=f'C{i}', label=f'det_mean : %.2f,\nsamp_mean: %.2f'%(func_means[0,i], Y_rff[i].mean()) )
        ax[0].axhline(func_means[0,i], xmin=-1, xmax=1, color=f'C{i}', linewidth=1.0, linestyle="--")
        ax[1].plot(X, Y_gp[i], '.')
    ax[0].legend()
    ax[0].set_title('rff')
    ax[1].set_title('gp')
    plt.show()

