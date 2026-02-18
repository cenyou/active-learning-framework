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

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.configs.kernels.multi_output_kernels.fpacoh_kernel_config import BasicFPACOHKernelConfig
from alef.configs.models.metagp_model_config import BasicMetaGPModelConfig
from alef.configs.paths import EXPERIMENT_PATH

class CartpoleMetaGPModelConfig(BasicMetaGPModelConfig):
    observation_noise: float=0.1
    kernel_config: BaseKernelConfig=BasicFPACOHKernelConfig(
        base_lengthscale=0.65,
        base_variance=0.92,
        input_dimension=5,
        output_dimension=4
    )
    load_model: Dict={
        'nn_kernel_map': (EXPERIMENT_PATH / 'safe_transfer_AL_metagp' / 'model_nn_kernel_map.pth'),
        'nn_mean_fn': (EXPERIMENT_PATH / 'safe_transfer_AL_metagp' / 'model_nn_mean_fn.pth'),
        'covar_module': (EXPERIMENT_PATH / 'safe_transfer_AL_metagp' / 'model_covar_module.pth'),
        #'mean_module': (EXPERIMENT_PATH / 'safe_transfer_AL_metagp' / 'model_mean_module.pth'),
        'normalization_stats_dict': {'x_mean': np.array([0.44, 0.48, 0.49, 0.53, 0.5 ]), 'y_mean': np.array([0.37]), 'x_std': np.array([0.27, 0.27, 0.29, 0.26, 0.29]), 'y_std': np.array([0.3])},
    }
    name : str = "CartpoleMetaGPModel"

class CartpoleMetaGPSafetyModelConfig(CartpoleMetaGPModelConfig):
    observation_noise: float=0.1
    kernel_config: BaseKernelConfig=BasicFPACOHKernelConfig(
        base_lengthscale=0.1,
        base_variance=2.0,
        input_dimension=5,
        output_dimension=4
    )
    load_model: Dict={
        'nn_kernel_map': (EXPERIMENT_PATH / 'safe_transfer_AL_metagp' / 'safe_model_nn_kernel_map.pth'),
        'nn_mean_fn': (EXPERIMENT_PATH / 'safe_transfer_AL_metagp' / 'safe_model_nn_mean_fn.pth'),
        'covar_module': (EXPERIMENT_PATH / 'safe_transfer_AL_metagp' / 'safe_model_covar_module.pth'),
        #'mean_module': (EXPERIMENT_PATH / 'safe_transfer_AL_metagp' / 'safe_model_mean_module.pth'),
        'normalization_stats_dict': {'x_mean': np.array([0.44, 0.48, 0.49, 0.53, 0.5 ]), 'y_mean': np.array([0.68]), 'x_std': np.array([0.27, 0.27, 0.29, 0.26, 0.29]), 'y_std': np.array([1.04])},
    }
    name : str = "CartpoleMetaGPSafetyModel"

if __name__ == '__main__':
    pass