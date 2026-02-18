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
import tensorflow as tf
import gpflow
from gpflow.mean_functions import MeanFunction
from gpflow.base import Parameter, TensorType
from gpflow.utilities import positive
from gpflow.config import default_float, default_int

from alef.configs.base_parameters import INPUT_DOMAIN

class QuadraticMean(MeanFunction):
    def __init__(self, input_size, **kwargs):
        """
        m(x) = c * ( b + 1/d sum_d w_d(x_d - center_d)**2 )
        """
        super().__init__()
        center = (INPUT_DOMAIN[1] + INPUT_DOMAIN[0]) / 2 * np.ones([input_size], dtype=default_float())
        self.center = Parameter(center)

        weights = 3*np.ones([input_size, 1], dtype=default_float())
        self.weights = Parameter(weights, transform=positive())

        self.scale = Parameter(-10*np.ones(1, dtype=default_float()))
        self.bias = Parameter(-1/2*np.ones(1, dtype=default_float()))

    def __call__(self, X: TensorType) -> tf.Tensor:
        # input is [*B, N, D], center / weights is [*B, D], up to broadcast permit
        shifted_input = 1/self.center.shape[-1] * (X - self.center)**2
        
        q = tf.einsum('...ND,Dm->...Nm', shifted_input, self.weights)
        return self.scale * ( q + self.bias )

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    D = 2
    B = 3
    m = QuadraticMean(D)
    x = np.random.rand(4, B, 10, D)
    print(m(x).shape)
    x = np.random.rand(1000, D)
    
    fig, axs = plt.subplots(1, 1)
    p = axs.tricontourf(
        x[:,0], x[:,1], m(x)[..., 0],
        levels=np.linspace(-5, 5, 100),
        cmap='seismic',
        alpha=1.0
    )
    fig.colorbar(p, ax=axs)
    plt.show()
    