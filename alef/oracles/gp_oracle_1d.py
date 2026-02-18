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
from typing import Union, Optional
from scipy import interpolate
from gpflow.utilities import print_summary, to_default_float
from alef.configs.kernels.base_elementary_kernel_config import BaseElementaryKernelConfig
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.configs.means import BaseMeanConfig, BasicZeroMeanConfig
from alef.configs.means.pytorch_means import BaseMeanPytorchConfig, BasicZeroMeanPytorchConfig
from alef.enums.environment_enums import GPFramework
from alef.gp_samplers.gp_gpflow_distribution import GPDistribution
from alef.gp_samplers.gp_gpytorch_distribution import GPTorchDistribution
from alef.oracles.base_oracle import Standard1DOracle

f64 = to_default_float


class GPOracle1D(Standard1DOracle):
    def __init__(
        self,
        kernel_config: Union[BaseElementaryKernelConfig, BaseKernelPytorchConfig],
        observation_noise: float,
        mean_config: Optional[Union[BaseMeanConfig, BaseMeanPytorchConfig]]=None,
        shift_mean: bool=False
    ):
        """
        GP(mean, kernel) + noise N(0, observation_noise^2)
        we can shift the function to prior mean (GP function in a small window doesn't always average to prior mean)

        to use this class, you need to initialize a GP function with the method initialize
        """
        super().__init__(observation_noise, 0.0, 1.0)

        assert kernel_config.input_dimension == 1
        self._shift_mean = shift_mean
        if isinstance(kernel_config, BaseElementaryKernelConfig):
            mean_config = BasicZeroMeanConfig() if mean_config is None else mean_config
            self.gp_dist = GPDistribution(kernel_config, observation_noise, mean_config=mean_config)
        elif isinstance(kernel_config, BaseKernelPytorchConfig):
            mean_config = BasicZeroMeanPytorchConfig() if mean_config is None else mean_config
            self.gp_dist = GPTorchDistribution(kernel_config, observation_noise, mean_config=mean_config)
        self.gp_dist.draw_parameter(draw_hyper_prior=False)

    def initialize(self, a, b, n):
        assert self.get_variable_dimension() == 1 # this is a 1D oracle, no variable doesn't make sense
        self.set_box_bounds(a, b)
        grid = np.expand_dims(
            np.linspace(*self.get_box_bounds(), n),
            axis=1
        )
        function_values = np.squeeze(
            self.gp_dist.sample_f(grid)
        )
        if self._shift_mean:
            function_values = function_values - function_values.mean(axis=0) + self.gp_dist.mean_numpy(grid)
        self.f = interpolate.interp1d(grid[..., 0], function_values, kind="linear")

    def draw_from_hyperparameter_prior(self):
        print("-Draw from hyperparameter prior")
        self.gp_dist.draw_parameter(draw_hyper_prior=True)
        self.gp_dist.show_parameter()

    def query(self, x, noisy=True):
        function_value = np.squeeze(self.f(x))
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_random_data_in_interval(self, n, a, b, noisy=True):
        X = np.random.uniform(low=a, high=b, size=(n, self.get_variable_dimension()))
        X = self._decorate_variable_with_context(X)
        return X, self.batch_query(X, noisy).reshape(-1, 1)


if __name__ == "__main__":
    from alef.configs.kernels.rbf_configs import RBFWithPriorConfig
    from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
    from alef.configs.means.pytorch_means import BasicPeriodicMeanPytorchConfig

    for i in range(0, 10):
        #kernel_config = RBFWithPriorConfig(input_dimension=1)
        kernel_config = RBFWithPriorPytorchConfig(input_dimension=1)
        mean_config = BasicPeriodicMeanPytorchConfig(input_dimension=1)

        gpOracle = GPOracle1D(kernel_config, 0.1, mean_config=mean_config, shift_mean=True)

        gpOracle.draw_from_hyperparameter_prior()
        gpOracle.initialize(0, 1, 100)
        gpOracle.plot()
