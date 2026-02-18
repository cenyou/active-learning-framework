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
from alef.oracles.base_oracle import BaseOracle, Standard1DOracle
from enum import Enum


class FunctionType(Enum):
    NONSTATIONARY = 1
    LOW_LENGHTSCALE = 2
    HIGH_LENGTHSCALE = 3


class SafeTestFunc(Standard1DOracle):
    def __init__(self, observation_noise=0.01, function_type=FunctionType.NONSTATIONARY):
        super().__init__(observation_noise, a=-6, b=10)
        self._function_type = function_type

    def f(self, x):
        if self._function_type == FunctionType.NONSTATIONARY:
            if x <= 0.6:
                return np.sin(x - 3.0) + 0.3 * np.sin(5 * (x - 3.0)) + 0.1 * x - 0.15
            else:
                return 0.8 * np.sin(0.5 * (x - 0.6)) + np.sin(0.6 - 3.0) + 0.3 * np.sin(5 * (0.6 - 3.0)) + 0.1 * x - 0.15
        elif self._function_type == FunctionType.HIGH_LENGTHSCALE:
            x = x + 0.12
            return np.sin(x - 3.0) + 0.3 * np.sin(5 * (x - 3.0)) + 0.2 * x + 0.25 - 0.02 * np.power(x, 2)

        elif self._function_type == FunctionType.LOW_LENGHTSCALE:
            x = x + 3.3
            return 0.8 * np.sin(0.5 * (x - 0.6)) + np.sin(0.6 - 3.0) + 0.3 * np.sin(5 * (0.6 - 3.0)) + 0.1 * x

    def query(self, x, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_left_data(self, n, noisy=True):
        return self.get_random_data_in_box(n, -np.inf, 0.6, noisy=noisy)

    def get_random_data_outside_intervall(self, a, b, n, noisy=True):
        n1 = int(n / 2)
        n2 = n - n1
        X1, Y1 = self.get_random_data_in_box(n1, -np.inf, a, noisy=noisy)
        X2, Y2 = self.get_random_data_in_box(n2, b, np.inf, noisy=noisy)
        return np.concatenate((X1, X2), axis=-2), np.concatenate((Y1, Y2), axis=-2)

    def get_right_data(self, n, noisy=True):
        return self.get_random_data_in_box(n, 0.6, np.inf, noisy=noisy)

