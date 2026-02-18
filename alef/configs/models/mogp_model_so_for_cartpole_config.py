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
from typing import Union, Sequence
from alef.configs.models.mogp_model_so_config import BasicSOMOGPModelConfig
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.kernels.multi_output_kernels.coregionalization_Platent_kernel_configs import (
    BasicCoregionalizationPLConfig,
)
from alef.enums.global_model_enums import PredictionQuantity

_P = 4
_kernel_pars = {
    'kernels[0].kernels[0].lengthscales':  np.array([ 0.49, 23.18, 31.53, 16.37,  0.16]) ,
    'kernels[0].kernels[1].W':  np.array([[ 0.1 , -0.05,  0.04,  0.02],
       [ 0.13, -0.07,  0.05,  0.03],
       [ 0.13, -0.07,  0.05,  0.03],
       [ 0.13, -0.07,  0.05,  0.03]]) ,
    'kernels[0].kernels[1].kappa':  np.array([1.21e-05, 7.55e-06, 4.66e-06, 3.14e-07]) ,
    'kernels[1].kernels[0].lengthscales':  np.array([29.83, 19.68, 25.72, 61.91, 18.4 ]) ,
    'kernels[1].kernels[1].W':  np.array([[0.31, 0.38, 0.04, 0.16],
       [0.36, 0.44, 0.05, 0.18],
       [0.37, 0.46, 0.05, 0.19],
       [0.36, 0.44, 0.05, 0.18]]) ,
    'kernels[1].kernels[1].kappa':  np.array([5.02e-08, 2.37e-11, 3.36e-06, 5.08e-06]) ,
    'kernels[2].kernels[0].lengthscales':  np.array([ 0.81,  0.46, 38.72,  0.31, 36.91]) ,
    'kernels[2].kernels[1].W':  np.array([[0.24, 0.03, 0.05, 0.1 ],
       [0.31, 0.03, 0.07, 0.12],
       [0.32, 0.04, 0.07, 0.13],
       [0.3 , 0.03, 0.07, 0.12]]) ,
    'kernels[2].kernels[1].kappa':  np.array([6.37e-08, 2.96e-06, 1.02e-06, 2.64e-13]) ,
    'kernels[3].kernels[0].lengthscales':  np.array([1.03e-03, 6.06e+00, 6.33e+00, 1.37e-06, 4.87e+00]) ,
    'kernels[3].kernels[1].W':  np.array([[-0.14,  0.04, -0.08,  0.03],
       [-0.06,  0.03, -0.03, -0.01],
       [-0.04,  0.05, -0.02, -0.03],
       [-0.04,  0.07, -0.03, -0.06]]) ,
    'kernels[3].kernels[1].kappa':  np.array([8.58e-05, 4.61e-16, 5.62e-22, 1.47e-10]) ,
}
_safe_kernel_pars = {
    'kernels[0].kernels[0].lengthscales':  np.array([1.18e+04, 3.68e-01, 2.83e+04, 2.75e-01, 8.08e+03]) ,
    'kernels[0].kernels[1].W':  np.array([[-0.06,  0.14, -0.76, -1.12],
       [-0.18,  0.13, -0.76, -0.99],
       [-0.17,  0.14, -0.81, -1.08],
       [-0.16,  0.12, -0.71, -0.93]]) ,
    'kernels[0].kernels[1].kappa':  np.array([6.32e-06, 3.51e-16, 4.75e-10, 1.54e-07]) ,
    'kernels[1].kernels[0].lengthscales':  np.array([ 0.54, 21.33, 21.39, 16.35, 12.54]) ,
    'kernels[1].kernels[1].W':  np.array([[-0.11,  0.01, -0.3 , -0.26],
       [-0.11,  0.01, -0.29, -0.25],
       [-0.13,  0.01, -0.34, -0.3 ],
       [-0.09,  0.01, -0.25, -0.22]]) ,
    'kernels[1].kernels[1].kappa':  np.array([4.23e-34, 5.15e-11, 1.05e-36, 2.42e-22]) ,
    'kernels[2].kernels[0].lengthscales':  np.array([5.06e-02, 7.28e-02, 2.37e+01, 1.85e-02, 7.74e+00]) ,
    'kernels[2].kernels[1].W':  np.array([[ 0.26, -0.05,  0.42,  0.6 ],
       [ 0.1 , -0.08,  0.25,  0.62],
       [ 0.13, -0.02,  0.29,  0.67],
       [ 0.15,  0.15,  0.28,  0.5 ]]) ,
    'kernels[2].kernels[1].kappa':  np.array([1.38e-09, 1.49e-10, 3.51e-08, 5.53e-07]) ,
    'kernels[3].kernels[0].lengthscales':  np.array([27.53, 20.32, 28.21, 22.31, 31.38]) ,
    'kernels[3].kernels[1].W':  np.array([[-0.67,  0.16,  0.08,  0.49],
       [-0.58,  0.14,  0.07,  0.43],
       [-0.74,  0.18,  0.09,  0.54],
       [-0.49,  0.12,  0.06,  0.36]]) ,
    'kernels[3].kernels[1].kappa':  np.array([2.25e-23, 1.58e-13, 2.24e-24, 1.11e-18]) ,
}
class CartpoleMOGPModelConfig(BasicSOMOGPModelConfig):
    optimize_hps: bool = True
    observation_noise: Union[float, Sequence[float]] = 0.1
    train_likelihood_variance: bool = False
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=5,
        output_dimension=_P,
        W_rank=_P,
        fix_kernel=True,
        assign_values=True,
        parameter_values=_kernel_pars,
    )
    name = "CartpoleMOGPModel"


class CartpoleMOGPSafetyModelConfig(CartpoleMOGPModelConfig):
    observation_noise: Union[float, Sequence[float]] = 0.1
    classification: bool = False
    kernel_config: BaseKernelConfig = BasicCoregionalizationPLConfig(
        input_dimension=5,
        output_dimension=_P,
        W_rank=_P,
        fix_kernel=True,
        assign_values=True,
        parameter_values=_safe_kernel_pars,
    )
    name = "CartpoleMOGPSafetyModel"


if __name__ == "__main__":
    from alef.models.model_factory import ModelFactory
    from gpflow.utilities import print_summary

    m_config = CartpoleMOGPModelConfig()
    print(m_config.kernel_config)
    print(m_config.kernel_config.base_lengthscale)
    model = ModelFactory.build(m_config)

    print_summary(model.kernel)
