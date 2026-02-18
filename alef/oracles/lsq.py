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
from typing import List, Union
from alef.oracles.base_oracle import Standard2DOracle
from alef.oracles.base_constrained_oracle import StandardConstrainedOracle
from alef.oracles.helpers.normalize_decorator import OracleNormalizer

"""
Gramacy, R. B., Gray, G. A., Le Digabel, S., Lee, H. K., Ranjan, P., Wells, G., and Wild, S. M. (2016).
Modeling an augmented lagrangian for blackbox constrained optimization.
Technometrics, 58(1):1–11.
"""

class LSQMain(Standard2DOracle):
    def __init__(
        self,
        observation_noise: float
    ):
        super().__init__(observation_noise, 0.0, 1.0)

    def f(self, x1, x2):
        return x1 + x2

    def query(self, x, noisy=True):
        function_value = self.f(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

class LSQConstraint1(Standard2DOracle):
    def __init__(
        self,
        observation_noise: float
    ):
        super().__init__(observation_noise, 0.0, 1.0)

    def c(self, x1, x2):
        return x1 + 2*x2 + 0.5*np.sin( 2*math.pi*(x1**2-2*x2) - 1.5)

    def query(self, x, noisy=True):
        function_value = self.c(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

class LSQConstraint2(Standard2DOracle):
    def __init__(
        self,
        observation_noise: float
    ):
        super().__init__(observation_noise, 0.0, 1.0)

    def c(self, x1, x2):
        return 1.5 - x1**2 - x2**2

    def query(self, x, noisy=True):
        function_value = self.c(x[0], x[1])
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value



class LSQ(StandardConstrainedOracle):
    def __init__(
        self,
        observation_noise: Union[float, List[float]],
        normalized: bool=True,
    ):
        """
        :param observation_noise: standard deviation of the observation noise (Gaussian noise)
            specify all functions by passing a float or specify each individually by passing a list of 2 floats
        :param normalized: whether to normalize the function values and the constraint values
        """
        if hasattr(observation_noise, '__len__'):
            if len(observation_noise) == 1:
                observation_noise = [observation_noise[0]] * 3
            elif len(observation_noise) == 3:
                pass
            else:
                assert False, f'passing incorrect number of observation_noise to {self.__class__.__name__}.__init__ method'
        else:
            observation_noise = [observation_noise] * 3

        if normalized:
            main_oracle = OracleNormalizer(LSQMain(observation_noise[0]))
            main_oracle.set_normalization_by_sampling()

            constraint_oracle = [
                OracleNormalizer(LSQConstraint1(observation_noise[1])),
                OracleNormalizer(LSQConstraint2(observation_noise[2]))
            ]
            for c in constraint_oracle:
                c.set_normalization_by_sampling()
                mu, scale = c.get_normalization()
                c.set_normalization_manually(0.0, scale)
        else:
            main_oracle = LSQMain(observation_noise[0])
            constraint_oracle = [LSQConstraint1(observation_noise[1]), LSQConstraint2(observation_noise[2])]

        super().__init__(main_oracle, constraint_oracle)


if __name__ == "__main__":
    oracle = LSQMain(0.01)
    oracle.plot()
    oracle = LSQConstraint1(0.01)
    oracle.plot()
    oracle = LSQConstraint2(0.01)
    oracle.plot()