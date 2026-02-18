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
from typing import List, Tuple, Optional, Union
from abc import ABC,abstractmethod
from pathlib import Path
from copy import deepcopy
from pyro.distributions import Categorical, Gamma, Uniform, TransformedDistribution
from torch.distributions import AffineTransform

from alef.configs.base_parameters import NUMERICAL_POSITIVE_LOWER_BOUND, BASE_KERNEL_LENGTHSCALE, INPUT_DOMAIN
from alef.configs.means.pytorch_means import BaseMeanPytorchConfig, BasicZeroMeanPytorchConfig
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.enums.active_learner_amortized_policy_enums import LengthscaleDistribution
from alef.kernels.pytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from alef.means.pytorch_means.pytorch_mean_factory import PytorchMeanFactory
from alef.means.pytorch_means.pytorch_zero_mean import ZeroPytorchMean
from alef.means.pytorch_means.pytorch_linear_mean import LinearPytorchMean
from alef.means.pytorch_means.pytorch_periodic_mean import PeriodicPytorchMean
from alef.means.pytorch_means.pytorch_quadratic_mean import QuadraticPytorchMean
from alef.means.pytorch_means.pytorch_sech_mean import SechPytorchMean
from alef.active_learners.amortized_policies.utils.oed_primitives import hyper_prior_sample
from alef.utils.pyro_distributions import CategoricalWithValues

class BaseSampler(torch.nn.Module):
    
    def __init__(
        self,
        kernel_config: BaseKernelPytorchConfig,
        observation_noise: float,
        overall_variance: float,
        function_variance_interval: Tuple[float, float],
        mean_variance: float = 0.0,
        mean_config: Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]] = BasicZeroMeanPytorchConfig(batch_shape=[]),
        lengthscale_distribution: LengthscaleDistribution=LengthscaleDistribution.GAMMA,
        name: str = 'base_gp_sampler'
    ):
        r"""
        :param kernel_config: BaseKernelPytorchConfig
        :param observation_noise: float, observation noise standard deviation
        :param overall_variance: float, overall variance of the GP (variance scale of kernel + noise variance)
        :param function_variance_interval: (low, up), the kernel variance is sample from this interval
        :param mean_variance: overall_variance = mean_variance + kernel_var + noise_var
                    we use noise_var = overall_variance - function_variance
                           function_variance is then distributed to mean and kernel
        :param mean_config: mean config of gp prior
        :param lengthscale_distribution: type of lengthscale sampler
        :param name: str, name of the GP sampler
        """
        super().__init__()
        #assert overall_variance > 1.0, 'we assume the overall variance is greater than 1.0'
        assert overall_variance > function_variance_interval[1]
        assert function_variance_interval[1] > function_variance_interval[0]
        assert function_variance_interval[0] >= mean_variance
        mean_variance = max(0.0, mean_variance)

        self.__kernel_config = kernel_config
        self.__mean_config = mean_config
        self.register_buffer(
            "_original_noise_variance",
            torch.tensor([math.pow(observation_noise, 2.0)])
        )
        self.register_buffer(
            "noise_variance",
            self._original_noise_variance.clone()
        )
        self.register_buffer(
            "_overall_variance",
            torch.tensor(overall_variance)
        )
        self.register_buffer(
            "_function_variance_interval",
            torch.tensor(function_variance_interval)
        )
        self.register_buffer("mean_variance", torch.tensor(mean_variance))
        self._original_kernel = PytorchKernelFactory.build(kernel_config)
        self.kernel = deepcopy(self._original_kernel)
        self.kernel_list = [self.kernel.kernel]
        self._original_mean_list = self.set_gp_mean(mean_config)
        self.mean_list = None
        self.set_lengthscale_sample_mode(lengthscale_distribution)
        self.name = name

    def set_lengthscale_sample_mode(self, distribution_type: LengthscaleDistribution=LengthscaleDistribution.GAMMA):
        self._lengthscale_distribution = distribution_type
        if distribution_type == LengthscaleDistribution.UNIFORM:
            low, up = max( 0.2, NUMERICAL_POSITIVE_LOWER_BOUND ), max( 1, BASE_KERNEL_LENGTHSCALE )
            self.register_buffer("_lengthscale_sampler_parameters", torch.tensor([low, up])) # [low, up]
            self._length_dist_str = f'Uniform({low}, {up})'
        elif distribution_type in [LengthscaleDistribution.GAMMA, LengthscaleDistribution.GAMMA_SMOOTH]:
            if distribution_type == LengthscaleDistribution.GAMMA:
                low = torch.tensor(0.2).clamp(min=NUMERICAL_POSITIVE_LOWER_BOUND)
                mean = torch.tensor(0.3).clamp(min=BASE_KERNEL_LENGTHSCALE).clamp(min=low + NUMERICAL_POSITIVE_LOWER_BOUND)
            elif distribution_type == LengthscaleDistribution.GAMMA_SMOOTH:
                low = torch.tensor(0.3).clamp(min=NUMERICAL_POSITIVE_LOWER_BOUND)
                mean = torch.tensor(0.5).clamp(min=BASE_KERNEL_LENGTHSCALE).clamp(min=low + NUMERICAL_POSITIVE_LOWER_BOUND)
            self.register_buffer("_lengthscale_sampler_parameters", torch.concat([low.unsqueeze(0), mean.unsqueeze(0)], dim=0))
            self._length_dist_str = f'AffineTransform({low.numpy()}, 1)(Gamma(alpha=1, beta={1/(mean.numpy() - low.numpy())}))'
        elif distribution_type == LengthscaleDistribution.PERCENTAGE:
            fluc_rate = 0.2
            lr = 1 - fluc_rate
            ur = 1 + fluc_rate
            self.register_buffer("_lengthscale_sampler_parameters", torch.tensor([lr, ur])) # [low, mean]
            self._length_dist_str = f'Uniform({lr*np.atleast_1d(self.__kernel_config.base_lengthscale)}, {ur*np.atleast_1d(self.__kernel_config.base_lengthscale)})'
        else:
            raise NotImplementedError

    def set_gp_mean(
        self,
        mean_config: Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]]
    ):
        """
        please do not assign batch size in the configs, our loss functions assume no batch in each mean function
        if needed, assign one config per priors
        """
        out = []
        if hasattr(mean_config, '__len__'):
            configs = mean_config
        else:
            configs = [mean_config]
        for m_config in configs:
            m_func = PytorchMeanFactory.build(m_config)
            out.append(m_func)
        return out

    def central_interval(self, input_domain: Tuple[float, float]=INPUT_DOMAIN, length_ratio: float = 0.1):
        """
        return [Nk, D, 2] interval tensor, the center of input domain
        
        input_domain is the input space interval
        length_ratio decide how long we want (length_ratio * length of input_domain is the final length of interval)
        """
        assert not self.mean_list is None
        D = self.__kernel_config.input_dimension
        Nk = len(self.mean_list)
        half_width = (input_domain[1] - input_domain[0]) * length_ratio / 2
        c = (input_domain[0] + input_domain[1]) / 2
        c_tensor = torch.concat([
            (c - half_width) * torch.ones([Nk, D, 1], device=self.mean_variance.device),
            (c + half_width) * torch.ones([Nk, D, 1], device=self.mean_variance.device)
        ], dim=-1)
        return c_tensor.clamp(min=input_domain[0], max=input_domain[1])

    def max_interval(self, input_domain: Tuple[float, float]=INPUT_DOMAIN, length_ratio: float = 0.1):
        """
        return [Nk, D, 2] interval tensor, the input domain around the max value of each prior mean
        this is useful when we want to sample safe data
        
        input_domain is the input space interval
        length_ratio decide how long we want (length_ratio * length of input_domain is the final length of interval)
        """
        assert not self.mean_list is None
        D = self.__kernel_config.input_dimension
        output_domains = []
        half_width = (input_domain[1] - input_domain[0]) * length_ratio / 2
        for m in self.mean_list:
            if isinstance(m, ZeroPytorchMean):
                c = (input_domain[0] + input_domain[1]) / 2
                c_tensor = torch.concat([
                    (c - half_width) * torch.ones([1, D, 1], device=self.mean_variance.device),
                    (c + half_width) * torch.ones([1, D, 1], device=self.mean_variance.device)
                ], dim=-1)
                output_domains.append( c_tensor )
            elif isinstance(m, LinearPytorchMean):
                # return the corner/mid of edge with largest value
                with torch.no_grad():
                    w = m.scale.unsqueeze(-1) * m.weights
                    c = torch.where(w >= 0, (input_domain[0] + input_domain[1]) / 2, input_domain[0] + half_width)
                    c = torch.where(w > 0, input_domain[1] - half_width, c) # [D]
                    interval = torch.concat([
                        c.unsqueeze(-1) - half_width,
                        c.unsqueeze(-1) + half_width
                    ], dim=-1)
                    output_domains.append( interval.unsqueeze(-3) )
            elif isinstance(m, (PeriodicPytorchMean, QuadraticPytorchMean, SechPytorchMean)):
                with torch.no_grad():
                    if m.scale > 0:
                        interval = torch.concat([
                            m.center.clone().unsqueeze(-1) - half_width,
                            m.center.clone().unsqueeze(-1) + half_width
                        ], dim=-1)
                    else:
                        c = (input_domain[0] + input_domain[1]) / 2
                        interval = torch.concat([
                            (c - half_width) * torch.ones([D, 1], device=self.mean_variance.device),
                            (c + half_width) * torch.ones([D, 1], device=self.mean_variance.device)
                        ], dim=-1)
                    output_domains.append( interval.unsqueeze(-3) )
            else:
                raise NotImplementedError
        return torch.concat(output_domains, dim=-3).clamp(min=input_domain[0], max=input_domain[1])

    def forward(self, *args, **kwargs):
        raise NotImplementedError # we don't really need this

    def set_device(self, device: torch.device):
        self.to(device)
        for m in self.modules():
            m.to(device)
        for m in self._original_mean_list:
            m.to(device)

        # Optional, because every re-sampling re-initialize kernel_list & mean_list from the _original_[kernel | mean]_list
        if self.kernel_list:
            for k in self.kernel_list:
                k.to(device)
        if self.mean_list:
            for m in self.mean_list:
                m.to(device)

    def clone_module(self):
        new = self.__new__(type(self))
        new.__init__(
            self.__kernel_config,
            math.sqrt(self._original_noise_variance.item()),
            self._overall_variance.item(),
            (self._function_variance_interval[0].item(), self._function_variance_interval[1].item()),
            mean_variance=self.mean_variance.item(),
            mean_config=self.__mean_config
        )
        return new

    def export_state_dict(self, folder):
        fp = Path(folder)
        fp.mkdir(exist_ok=True, parents=True)
        torch.save(
            self.state_dict(),
            fp / 'sampler.pth'
        )

    def import_state_dict(self, folder):
        fp = Path(folder)
        assert (fp / 'sampler.pth').exists()
        state_dict = torch.load(fp / 'sampler.pth')
        self.load_state_dict(state_dict)

    def global_variables_in_dict(self, name: Optional[str]=None):
        r"""
        return global variables in dictionary
        can provide name as dict key prefix
        """
        prefix = '' if name is None else f'{name}.'
        vs = {
            prefix+'overall_observation_variance': self._overall_variance.item(),
            prefix+'function_variance_interval': [self._function_variance_interval[0].item(), self._function_variance_interval[1].item()],
            prefix+'mean_variance': self.mean_variance.item(),
            prefix+'kernel_variance_interval': [self._function_variance_interval[0].item() - self.mean_variance.item(), self._function_variance_interval[1].item() - self.mean_variance.item()],
        }
        # kernel variance
        low = max(
            self._function_variance_interval[0].item() - self.mean_variance,
            NUMERICAL_POSITIVE_LOWER_BOUND
        )
        up = self._function_variance_interval[1].item() - self.mean_variance
        vs[prefix+'kernel_variance'] = f'Uniform({low}, {up})'
        vs[prefix+'kernel_lengthscale'] = self._length_dist_str
        return vs

    def _replace_kernel_prior(self, prior_name, module, closure):
        """
        The kernel object still needs to register kernels.
        The kernels are also used in other AL/BO pipelines so we leave them as they are.
        Here, we replace the registered distributions by pyro distributions for the following reasons:
            1. the main reason is to set parameters that are specific to our amortized training,
               especially kernel variance which are set jointly with noise and prior mean, harder to pre-define
               (gp mean functions are more complicated, so we only set a scale constant)
            2. pyro distributions is easier for testing (because of pyro condition, which allows external control of samples)
            3. gpytorch kernel priors are forced to be gpytorch.priors.Prior,
               but gpytorch priors are often not compatible for GPU usage
        """
        if 'outputscale' in prior_name or 'variance' in prior_name:
            low = max( self._function_variance_interval[0] - self.mean_variance, NUMERICAL_POSITIVE_LOWER_BOUND )
            up = (self._function_variance_interval[1] - self.mean_variance) * torch.ones_like(closure(module))
            dist = Uniform( low, up )
        elif 'offset' in prior_name:
            low = - 1.0
            up = torch.ones_like(closure(module))
            dist = Uniform( low, up )
        elif 'lengthscale' in prior_name:
            if self._lengthscale_distribution == LengthscaleDistribution.UNIFORM:
                low = self._lengthscale_sampler_parameters[0]
                up = self._lengthscale_sampler_parameters[1] * torch.ones_like(closure(module))
                dist = Uniform( low, up )
            elif self._lengthscale_distribution in [LengthscaleDistribution.GAMMA, LengthscaleDistribution.GAMMA_SMOOTH]:
                low = self._lengthscale_sampler_parameters[0]
                beta = 1/(self._lengthscale_sampler_parameters[1] - low)
                dist = TransformedDistribution(
                    Gamma(
                        torch.ones_like(closure(module)), # becomes an exp distribution
                        beta
                    ),
                    [AffineTransform(low, 1.0)]
                ) # mean is 1/beta + low
            elif self._lengthscale_distribution == LengthscaleDistribution.PERCENTAGE:
                # this is valid because we reset kernel everytime before the sampling
                low = self._lengthscale_sampler_parameters[0] * closure(module)
                up = self._lengthscale_sampler_parameters[1] * closure(module)
                dist = Uniform( low, up )
            else:
                raise NotImplementedError
        else:
            low = max( 0.05, NUMERICAL_POSITIVE_LOWER_BOUND )
            up = torch.ones_like(closure(module))
            dist = Uniform( low, up )
        return dist

    def draw_parameter(
        self,
        num_priors: int=1,
        draw_hyper_prior: bool=False,
        input_domain: Tuple[float, float]=INPUT_DOMAIN,
        *,
        mask: Optional[Union[np.ndarray, torch.Tensor]]=None,
    ):
        """
        draw hyper-priors, f, or noise of y|f
        
        :param num_priors: batch size of kernel hyperparameters (i.e. num of kernels).
        :param draw_hyper_prior: whether to draw parameters from hyper-priors
        :param input_domain: input domain interval, this is used if we need mean functions
        :param mask: [num_priors, D] if given, can mask out dimension per function (reduce dimension)
        
        """
        assert num_priors > 0
        assert not draw_hyper_prior or self.__kernel_config.add_prior
        D = self.__kernel_config.input_dimension

        # GP mean, select from pool, sample parameters
        # GP mean functions are diverse, there is no universal way to control them, must be tuned individually
        # we only rescale the mean scale, assuming variance 1 is maintained by itself
        self.mean_list = []
        mean_selector = Categorical(
            torch.ones(len(self._original_mean_list), device=self.mean_variance.device)
        )
        for i in range(num_priors):
            if not draw_hyper_prior:
                m = deepcopy(self._original_mean_list[0])
                m._set_scale( self.mean_variance.sqrt() * m.scale )
            else:
                mean_idx = hyper_prior_sample(f'{self.name}.gp_mean.{i}.index', mean_selector)
                m = deepcopy(self._original_mean_list[mean_idx])
                for prior_name, module, prior, closure, setting_closure in m.named_priors():
                    value = hyper_prior_sample(f'{self.name}.gp_mean.{i}.{prior_name}', prior)
                    setting_closure(module, value)
                m._set_scale( self.mean_variance.sqrt() * m.scale )
            self.mean_list.append(m)

        if num_priors > 1:
            self.kernel.kernel = self._original_kernel.kernel.expand_batch((num_priors, ))
            self.noise_variance = self._original_noise_variance.expand((num_priors, ))
        else: # reset to original
            # do not remove this because shaping will be wrong, lengthscale sampling, if with percentage, will also be wrong
            self.kernel = deepcopy(self._original_kernel)
            self.noise_variance = deepcopy(self._original_noise_variance)

        if not draw_hyper_prior:
            # even if we don't draw kernel, we want to discount kernel scale
            self.kernel.prior_scale = self.kernel.prior_scale * torch.sqrt(
                (self._function_variance_interval[1] - self.mean_variance) / self._function_variance_interval[1]
            )
        else:
            # one can also do self.kernel.kernel = self.kernel.kernel.pyro_sample_from_prior(), but priors need to be predefined when initiating a kernel object
            kernel = gpytorch.kernels.AdditiveKernel(self.kernel.kernel).to_random_module()
            # we must wrap the kernel into gpytorch.kernel.AdditiveKernel, otherwise,
            # after setting_closure, somehow the parameters are removed from
            # the torch.nn.Module parameters list, which cause problems later when we index kernels
            for prior_name, module, prior, closure, setting_closure in kernel.named_priors():
                dist = self._replace_kernel_prior(prior_name, module, closure)
                value = hyper_prior_sample(f'{self.name}.{prior_name}', dist)
                setting_closure(module, value)
            self.kernel.kernel = kernel.kernels[0]
            # one can also do self.kernel.kernel = self.kernel.kernel.pyro_sample_from_prior(), but priors need to be predefined when initiating a kernel object

            self.noise_variance = self._overall_variance * torch.ones_like(self.noise_variance) - \
                self.mean_variance -\
                self.kernel.prior_scale.pow(2)# + NOISE_VARIANCE_LOWER_BOUND

        if num_priors > 1:
            self.kernel_list = [self.kernel.kernel[i] for i in range(num_priors)]
        else:
            self.kernel_list = [self.kernel.kernel]

        # store mask
        
        if mask is None:
            mask_tensor = None
        else:
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask).to(int)
            elif isinstance(mask, torch.Tensor):
                mask_tensor = mask
            else:
                raise ValueError('x_data is not a np.ndarray or torch.Tensor')
        self._mask = mask_tensor

    @property
    def num_priors(self):
        return len(self.kernel_list)

    def _mask_x(self, x_data: torch.Tensor, mask: Optional[torch.Tensor]=None):
        """
        :param x_data: torch.Tensor, size [*batch_sizes, N, d]
        :param mask: torch.Tensor or None, size [*batch_sizes, d] if given, elements 0 (masked out) or 1 (kept in)
        
        :return: torch.Tensor, size [*batch_sizes, N, d]
        """
        if mask is None:
            return x_data
        else:
            return torch.where(
                mask.unsqueeze(-2) == 1, x_data, torch.zeros_like(x_data)
            )

    @abstractmethod
    def f_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ), return in raw torch type 

        :param x_data: torch.Tensor, size [*batch_sizes, d]

        :return: distribution of size [*batch_sizes]
        """
        raise NotImplementedError

    @abstractmethod
    def y_sampler(self, x_data: torch.Tensor):
        """
        compute GP( mean(x_data), kernel(x_data) ) + noise_dist, return in raw torch type 

        :param x_data: torch.Tensor, size [*batch_sizes, d]

        :return: distribution of size [*batch_sizes]
        """
        raise NotImplementedError

    def sample_f(self, x_data: Union[np.ndarray, torch.Tensor]):
        """
        sample f from GP( mean(x_data), kernel(x_data) )

        :param x_data: array, size [*batch_sizes, d]

        :return: array, size [*batch_sizes]
        """
        if isinstance(x_data, np.ndarray):
            X = torch.from_numpy(x_data).to(torch.get_default_dtype())
            return self.f_sampler(X).sample().cpu().numpy()
        elif isinstance(x_data, torch.Tensor):
            return self.f_sampler(x_data).sample()
        else:
            raise ValueError('x_data is not a np.ndarray or torch.Tensor')

    def sample_y(self, x_data: Union[np.ndarray, torch.Tensor]):
        """
        sample y from GP( mean(x_data), kernel(x_data) ) + noise_dist(x_data)

        :param x_data: array, size [*batch_sizes, d]

        :return: array, size [*batch_sizes]
        """
        if isinstance(x_data, np.ndarray):
            X = torch.from_numpy(x_data).to(torch.get_default_dtype())
            return self.y_sampler(X).sample().cpu().numpy()
        elif isinstance(x_data, torch.Tensor):
            return self.y_sampler(x_data).sample()
        else:
            raise ValueError('x_data is not a np.ndarray or torch.Tensor')

