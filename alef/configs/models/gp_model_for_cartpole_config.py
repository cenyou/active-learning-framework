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
from alef.configs.models.base_model_config import BaseModelConfig
from alef.configs.models.gp_model_config import BasicGPModelConfig
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.kernels.matern52_for_cartpole_configs import CartpoleMatern52Config, CartpoleMatern52SafetyConfig
from alef.enums.global_model_enums import PredictionQuantity
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE

class CartpoleGPModelConfig(BasicGPModelConfig):
    observation_noise: float = 0.1
    kernel_config: BaseKernelConfig = CartpoleMatern52Config()
    optimize_hps: bool = True
    train_likelihood_variance: bool = False
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name = "CartpoleGPModel"

class CartpoleGPSafetyModelConfig(CartpoleGPModelConfig):
    observation_noise: float = 0.1
    kernel_config: BaseKernelConfig = CartpoleMatern52SafetyConfig()
    classification: bool = False
    name = "CartpoleGPSafetyModel"

if __name__ == "__main__":
    from alef.models.model_factory import ModelFactory
    from gpflow.utilities import print_summary
    m_config = CartpoleGPModelConfig()
    model = ModelFactory.build(m_config)
    
    print_summary(model.kernel)

    