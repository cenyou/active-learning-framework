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
from typing import Union, Optional
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.configs.means.pytorch_means import BaseMeanPytorchConfig, BasicZeroMeanPytorchConfig
from alef.gp_samplers.gp_gpytorch_distribution import GPTorchDistribution
from alef.oracles.base_oracle import StandardOracle



class GPOracleHD(StandardOracle):
    def __init__(
        self,
        kernel_config: BaseKernelPytorchConfig,
        observation_noise: float,
        mean_config: Optional[BaseMeanPytorchConfig]=None,
        shift_mean: bool=False
    ):
        """
        GP(mean, kernel) + noise N(0, observation_noise^2)
        we can shift the function to prior mean (GP function in a small window doesn't always average to prior mean)

        to use this class, you need to initialize a GP function with the method initialize
        """
        super().__init__(observation_noise, 0.0, 1.0, kernel_config.input_dimension)

        self._shift_mean = shift_mean
        mean_config = BasicZeroMeanPytorchConfig() if mean_config is None else mean_config
        self.gp_dist = GPTorchDistribution(kernel_config, observation_noise, mean_config=mean_config)
        self.gp_dist.draw_parameter(draw_hyper_prior=False)

    def initialize(self, a, b, n):
        self.set_box_bounds(a, b)
        self.gp_dist.kernel.sample_fourier_features(n, num_functions=1)
        self.f = self.gp_dist.kernel.bayesian_linear_model(
            x_expanded_already=True,
            input_domain=(float(a), float(b)),
            shift_mean=self._shift_mean
        )

    def draw_from_hyperparameter_prior(self):
        print("-Draw from hyperparameter prior")
        self.gp_dist.draw_parameter(draw_hyper_prior=True)
        self.gp_dist.show_parameter()

    def query(self, x, noisy=True):
        D = self.get_variable_dimension()
        shape = torch.broadcast_shapes([1,1,1,1,D], x.shape)
        x_torch = torch.from_numpy(x).to(torch.get_default_dtype()).expand(shape)
        function_value = self.f(x_torch).squeeze().detach().numpy().astype(x.dtype)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def batch_query(self, X: np.ndarray, noisy: bool=True):
        D = self.get_variable_dimension()
        shape = torch.broadcast_shapes([1,1,1,1,D], X.shape) # [1,1,1,N,D]
        x_torch = torch.from_numpy(X).to(torch.get_default_dtype()).expand(shape)
        function_value = self.f(x_torch).detach().numpy().astype(X.dtype)[0,0,0]
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, X.shape[0])
            function_value += epsilon
        return function_value

    def get_random_data_in_interval(self, n, a, b, noisy=True):
        X = np.random.uniform(low=a, high=b, size=(n, self.get_variable_dimension()))
        X = self._decorate_variable_with_context(X)
        return X, self.batch_query(X, noisy).reshape(-1, 1)


if __name__ == "__main__":
    from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
    from alef.configs.means.pytorch_means import BasicSechRotatedMeanPytorchConfig
    from matplotlib import pyplot as plt

    kernel_config = RBFWithPriorPytorchConfig(input_dimension=1)
    mean_config = BasicSechRotatedMeanPytorchConfig(input_dimension=1)

    gpOracle = GPOracleHD(kernel_config, 0.1, mean_config=mean_config, shift_mean=True)

    #gpOracle.draw_from_hyperparameter_prior()
    gpOracle.initialize(0, 1, 100)
    X, Y = gpOracle.get_random_data(100, noisy=True)

    fig, axs = plt.subplots(1,1)
    axs.plot(X, Y, 'o')
    plt.show()
