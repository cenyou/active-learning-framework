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
from typing import List, Union
from alef.oracles.base_oracle import StandardOracle
from alef.oracles.base_constrained_oracle import StandardConstrainedOracle
from alef.oracles.helpers.normalize_decorator import OracleNormalizer

class GasTransmissionCompressorDesignMain(StandardOracle):
    def __init__(
        self,
        observation_noise: float,
    ):
        D = 4
        super().__init__(observation_noise, 0.0, 1.0, D)

    def x_scale(self, x: np.ndarray):
        assert x.shape[-1] == 4, x.shape

        desire_low_bound = np.array([20, 1, 20, 0.1])
        desire_up_bound = np.array([50, 10, 50, 60])
        desire_width = desire_up_bound - desire_low_bound
        a, b = self.get_box_bounds()
        return (x - a) / (b - a) * desire_width + desire_low_bound

    def f(self, x):
        xx = self.x_scale(x)
        return 8.61* (10**5) * np.sqrt( xx[...,0] ) * xx[...,1] * np.power(xx[...,2], -2/3) * np.power(xx[...,3], -1/2) + \
            3.69* (10**4) * xx[...,2] + 7.72* (10**8) /xx[...,0] * np.power(xx[...,1], 0.219) - 765.43 * (10**6) / xx[...,0]

    def query(self,x:np.ndarray, noisy=True):
        function_value = self.f(x)
        if noisy:
            epsilon = np.random.normal(0,self.observation_noise,1)[0]
            function_value += epsilon
        return function_value

class GasTransmissionCompressorDesignConstraint(GasTransmissionCompressorDesignMain):

    def c(self, x):
        xx = self.x_scale(x)
        return -1 + xx[..., 3] * np.power(xx[..., 1], -2) + np.power(xx[..., 1], -1)

    def query(self,x:np.ndarray, noisy=True):
        function_value = self.c(x)
        if noisy:
            epsilon = np.random.normal(0,self.observation_noise,1)[0]
            function_value += epsilon
        return function_value

class GasTransmissionCompressorDesign(StandardConstrainedOracle):
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
                observation_noise = [observation_noise[0]] * 2
            elif len(observation_noise) == 2:
                pass
            else:
                assert False, f'passing incorrect number of observation_noise to {self.__class__.__name__}.__init__ method'
        else:
            observation_noise = [observation_noise] * 2

        if normalized:
            main = OracleNormalizer(GasTransmissionCompressorDesignMain(observation_noise[0]))
            main.set_normalization_by_sampling()

            constraint = OracleNormalizer(GasTransmissionCompressorDesignConstraint(observation_noise[1]))
            constraint.set_normalization_by_sampling()
            mu, scale = constraint.get_normalization()
            constraint.set_normalization_manually(0.0, scale)
        else:
            main = GasTransmissionCompressorDesignMain(observation_noise[0])
            constraint = GasTransmissionCompressorDesignConstraint(observation_noise[1])
        super().__init__(main, constraint)



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    oracle = GasTransmissionCompressorDesign(0.01)
    X, Y, Z = oracle.get_random_data(10000, noisy=True)
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    print( f'[{Y.min()}, {Y.max()}]' )
    print( f'mean {Y.mean()}, std {np.sqrt(Y.var())}' )
    print( f'[{Z.min()}, {Z.max()}]' )
    print( f'mean {Z.mean()}, std {np.sqrt(Z.var())}' )
    print( 100*(Z >= 0).sum() / Z.shape[0], '%' )
    split_factor = 3
    fig, axs = plt.subplots(split_factor, split_factor, sharex='all', sharey='all')
    for x2 in range(split_factor):
        mask2 = np.logical_and(X[:, 2] >= x2/split_factor, X[:, 2] <= (x2+1)/split_factor) # x2 in [0, 1/3] or [1/3, 2/3] or [2/3, 1]
        for x3 in range(split_factor):
            mask3 = np.logical_and(X[:, 3] >= x3/split_factor, X[:, 3] <= (x3+1)/split_factor) # x3 in [0, 1/3] or [1/3, 2/3] or [2/3, 1]
            for x0 in range(split_factor):
                mask0 = np.logical_and(X[:, 0] >= x0/split_factor, X[:, 0] <= (x0+1)/split_factor) # x0 in [0, 1/3] or [1/3, 2/3] or [2/3, 1]
                for x1 in range(split_factor):
                    mask1 = np.logical_and(X[:, 1] >= x1/split_factor, X[:, 1] <= (x1+1)/split_factor) # x1 in [0, 1/3] or [1/3, 2/3] or [2/3, 1]
                    mask = np.logical_and(np.logical_and(mask2, mask3), np.logical_and(mask0, mask1))
                    safe_ratio = np.sum(Z[mask] >= 0)
                    axs[x3, x2].text(
                        (x0+0.5)/split_factor,
                        (x1+0.5)/split_factor,
                        f'{safe_ratio}/{mask.sum()}',
                        ha="center", va="center", color="k"
                    )
            for x0 in range(1, split_factor):
                axs[x3, x2].plot([x0/split_factor, x0/split_factor], [0, 1], 'k--')
            for x1 in range(1, split_factor):
                axs[x3, x2].plot([0, 1], [x1/split_factor, x1/split_factor], 'k--')
            axs[x3, x2].set_title(f'x2 (dv): [{x2/split_factor:.2f}, {(x2+1)/split_factor:.2f}]\nx3 (v): [{x3/split_factor:.2f}, {(x3+1)/split_factor:.2f}]')
            axs[x3, x2].set_xlabel('x0 (dn)')
            axs[x3, x2].set_ylabel('x1 (n)')
    plt.tight_layout()
    plt.show()


    X, Y, Z = oracle.get_random_data_in_box(5000, 0.4, 0.2, noisy=True)
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    print( f'[{Y.min()}, {Y.max()}]' )
    print( f'mean {Y.mean()}, std {np.sqrt(Y.var())}' )
    print( f'[{Z.min()}, {Z.max()}]' )
    print( f'mean {Z.mean()}, std {np.sqrt(Z.var())}' )
    print( 100*(Z >= 0).sum() / Z.shape[0], '%' )
