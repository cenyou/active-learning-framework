import numpy as np
import torch

from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
from alef.configs.means.pytorch_means import BasicLinearMeanPytorchConfig
from alef.active_learners.amortized_policies.simulated_processes.gp_sampler.rff_sampler import RandomFourierFeatureSampler


B = 1
num_kernels = 2
num_func_per_kernel = 3
N = 1000
D = 2
x = np.random.rand(B, num_kernels, num_func_per_kernel, N, D)
x_torch = torch.from_numpy( x ).to( torch.get_default_dtype() )

def test_rff_torch_sampler():
    kernel_config = RBFWithPriorPytorchConfig(
        input_dimension=D
    )

    dist = RandomFourierFeatureSampler(kernel_config, 0.1, 1.0001, mean_variance=0.7, mean_config=BasicLinearMeanPytorchConfig(input_dimension=D, batch_shape=[], scale=2.5)).clone_module()
    assert np.isclose(dist.noise_variance.item(), 0.1**2)
    dist.draw_parameter(num_kernels, num_func_per_kernel, True)
    assert dist.kernel.kernel.base_kernel.lengthscale.shape == (num_kernels, 1, D)
    assert dist.kernel.kernel.outputscale.shape == (num_kernels, )
    assert dist.noise_variance.shape == (num_kernels, )
    f_samples = dist.f_sampler(x_torch).sample()
    y_samples = dist.y_sampler(x_torch).sample()
    assert f_samples.shape == (B, num_kernels, num_func_per_kernel, N)
    assert y_samples.shape == (B, num_kernels, num_func_per_kernel, N)
    print(f_samples.shape, f_samples.mean(-1))
    dist.draw_parameter(num_kernels, num_func_per_kernel, True, shift_mean=True, flip_center=True)
    print(dist.max_interval())
    prior_mean = dist.mean(x_torch)
    f_samples = dist.f_sampler(x_torch).sample()
    print(f_samples.shape, (f_samples-prior_mean).mean(-1))


if __name__=='__main__':
    test_rff_torch_sampler()
