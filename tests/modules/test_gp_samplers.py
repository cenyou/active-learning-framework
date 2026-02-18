import numpy as np
import tensorflow as tf
import torch
import pytest

from alef.kernels.kernel_factory import KernelFactory
from alef.kernels.pytorch_kernels.pytorch_kernel_factory import PytorchKernelFactory
from alef.configs.kernels.rbf_configs import RBFWithPriorConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
from alef.configs.means import BasicZeroMeanConfig
from alef.configs.means.pytorch_means import BasicZeroMeanPytorchConfig, BasicPeriodicMeanPytorchConfig

from alef.gp_samplers.gp_gpflow_distribution import GPDistribution
from alef.gp_samplers.gp_gpytorch_distribution import GPTorchDistribution, NormalizedGPTorchDistribution

num_kernels = 2
num_func_per_kernel = 3
N = 10
D = 2
x = np.random.standard_normal([N, D])
x_expand = np.tile(x[None, None, ...], (num_kernels, num_func_per_kernel, 1, 1) )
x_torch = torch.from_numpy( x ).to( torch.get_default_dtype() )

def test_tf_sampler():
    kernel_config = RBFWithPriorConfig(
        input_dimension=D
    )

    dist = GPDistribution(kernel_config, 0.1)
    dist.draw_parameter(num_kernels, num_func_per_kernel, True)
    assert len(dist.kernel_list) == num_kernels
    assert len(dist.noise_variance) == num_kernels
    f_samples = dist.f_sampler(x_expand).sample()
    y_samples = dist.y_sampler(x_expand).sample()
    assert f_samples.shape == (num_kernels, num_func_per_kernel, N)
    assert y_samples.shape == (num_kernels, num_func_per_kernel, N)
    assert dist.sample_f(x_expand).shape == (num_kernels, num_func_per_kernel, N)
    assert dist.sample_y(x_expand).shape == (num_kernels, num_func_per_kernel, N)

@pytest.mark.parametrize("sampler_class", [GPTorchDistribution, NormalizedGPTorchDistribution])
def test_torch_sampler(sampler_class):
    kernel_config = RBFWithPriorPytorchConfig(input_dimension=D)
    mean_config = BasicPeriodicMeanPytorchConfig(input_dimension=D)

    dist = sampler_class(kernel_config, 0.1, mean_config=mean_config)
    dist.draw_parameter(num_kernels, num_func_per_kernel, True)
    assert dist.kernel.kernel.base_kernel.lengthscale.shape == (num_kernels, 1, D)
    assert dist.kernel.kernel.outputscale.shape == (num_kernels, )
    assert dist.noise_variance.shape == (num_kernels, )
    f_samples = dist.f_sampler(
        x_torch.expand([num_kernels, num_func_per_kernel, N, D])
    ).sample()
    y_samples = dist.y_sampler(
        x_torch.expand([num_kernels, num_func_per_kernel, N, D])
    ).sample()
    assert f_samples.shape == (num_kernels, num_func_per_kernel, N)
    assert y_samples.shape == (num_kernels, num_func_per_kernel, N)
    assert dist.sample_f(x_expand).shape == (num_kernels, num_func_per_kernel, N)
    assert dist.sample_y(x_expand).shape == (num_kernels, num_func_per_kernel, N)
    ###
    dist.draw_parameter(1, 1, True)
    assert dist.kernel.kernel.base_kernel.lengthscale.shape == (1, D)
    f_samples = dist.f_sampler( x_torch ).sample()
    y_samples = dist.y_sampler( x_torch ).sample()
    assert f_samples.shape == (N, )
    assert y_samples.shape == (N, )
    assert dist.sample_f(x).shape == (N, )
    assert dist.sample_y(x).shape == (N, )


if __name__=='__main__':
    test_tf_sampler()
    test_torch_sampler()
