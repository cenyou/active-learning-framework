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
from alef.configs.models.mogp_model_transfer_config import BasicTransferGPModelConfig
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.kernels.multi_output_kernels.multi_source_additive_kernel_configs import BasicMIAdditiveConfig
from alef.enums.global_model_enums import PredictionQuantity

_P = 4
_kernel_pars = {
    'kernels[0].variance':  0.23649467678874786 ,
    'kernels[0].lengthscales':  np.array([0.56, 0.57, 4.06, 0.37, 0.4 ]) ,
    'kernels[1].kernels[1].variance':  0.004929804663401428 ,
    'kernels[1].kernels[1].lengthscales':  np.array([1.90e-01, 8.92e-04, 1.32e+01, 8.89e+00, 9.27e+00]) ,
    'kernels[2].kernels[1].variance':  0.0035680088918280803 ,
    'kernels[2].kernels[1].lengthscales':  np.array([0.19, 9.72, 0.12, 0.54, 6.24]) ,
    'kernels[3].kernels[1].variance':  0.002139376675908325 ,
    'kernels[3].kernels[1].lengthscales':  np.array([1.33e+01, 2.09e+01, 1.43e-02, 8.96e+00, 4.78e+00]) ,
}
_safe_kernel_pars = {
    'kernels[0].variance':  1.8059076317171032 ,
    'kernels[0].lengthscales':  np.array([8.79e-02, 2.07e-01, 1.83e+03, 1.64e-01, 2.42e+00]) ,
    'kernels[1].kernels[1].variance':  0.048790947981458825 ,
    'kernels[1].kernels[1].lengthscales':  np.array([2.98e-04, 7.34e+00, 1.04e-02, 9.89e+00, 9.47e-04]) ,
    'kernels[2].kernels[1].variance':  9.999999999999988e-10 ,
    'kernels[2].kernels[1].lengthscales':  np.array([5.07e+00, 1.82e+00, 2.40e+00, 2.09e-04, 2.01e-04]) ,
    'kernels[3].kernels[1].variance':  0.05169994315917596 ,
    'kernels[3].kernels[1].lengthscales':  np.array([2.37e+00, 6.84e-04, 1.01e-02, 1.12e-04, 3.36e-03]) ,
}
class CartpoleTransferGPModelConfig(BasicTransferGPModelConfig):
    optimize_hps: bool = True
    observation_noise: Union[float, Sequence[float]] = 0.1
    train_likelihood_variance: bool = False
    kernel_config: BaseKernelConfig = BasicMIAdditiveConfig(
        input_dimension=5,
        output_dimension=_P,
        fix_kernel=True,
        assign_values=True,
        parameter_values=_kernel_pars
    )
    name = "CartpoleTransferGPModel"

class CartpoleTransferGPSafetyModelConfig(CartpoleTransferGPModelConfig):
    observation_noise: Union[float, Sequence[float]] = 0.1
    classification: bool = False
    kernel_config: BaseKernelConfig = BasicMIAdditiveConfig(
        input_dimension=5,
        output_dimension=_P,
        fix_kernel=True,
        assign_values=True,
        parameter_values=_safe_kernel_pars
    )
    name = "CartpoleTransferGPSafetyModel"

if __name__ == "__main__":
    from alef.models.model_factory import ModelFactory
    from gpflow.utilities import print_summary
    m_config = CartpoleTransferGPModelConfig()
    print(m_config.kernel_config)
    print(m_config.kernel_config.base_lengthscale)
    model = ModelFactory.build(m_config)
    
    print_summary(model.kernel)

    