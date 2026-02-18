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

from typing import List, Union
from pydantic_settings import BaseSettings
from .base_loss_configs import (
    BasicAmortizedPolicyLossConfig,
    BasicNonMyopicLossConfig,
    BasicMyopicLossConfig,
    #
    BasicSafetyAwarePolicyLossConfig,
    BasicLossCurriculumConfig,
)

_myopic_loss_list = [
    'GPMyopicEntropy1LossConfig',
    'GPMyopicEntropy2LossConfig',
    'GPMyopicMI1LossConfig',
    'GPMyopicMI2LossConfig',
]
_nonmyopic_loss_list = [
    'DADLossConfig',
    'GPEntropy1LossConfig',
    'GPEntropy2LossConfig',
    'GPMI1LossConfig',
    'GPMI2LossConfig',
    #'GPPCELossConfig', # !!! this loss is under development
    #'GPMI_Entropy1LossConfig',
    #'GPMI_Entropy2LossConfig',
]

__all__ = _myopic_loss_list + _nonmyopic_loss_list

############
# DAD loss
#
class DADLossConfig(BasicNonMyopicLossConfig):
    batch_size: int = 10
    num_kernels: int = 10
    num_functions_per_kernel: int = 200
    name: str='dad_loss'

class _ScoreDADLossConfig(BasicNonMyopicLossConfig):
    batch_size: int = 25
    num_kernels: int = 10
    num_functions_per_kernel: int = 200
    name: str='dad_reinforce_loss'

############
# GP Entropy loss
#
class GPEntropy1LossConfig(BasicNonMyopicLossConfig):
    batch_size: int = 10
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='E[  -H(y|y_init)  ]'

class GPEntropy2LossConfig(BasicNonMyopicLossConfig):
    batch_size: int = 10
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='E[  log p(y|y_init)  ]'

############
class GPMyopicEntropy1LossConfig(BasicMyopicLossConfig):
    batch_size: int = 10
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_splits: int = 5 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200 # this loss need pretraining, not just long training
    name: str='myopic E[  -H(y|y_init)  ]'

class GPMyopicEntropy2LossConfig(BasicMyopicLossConfig):
    batch_size: int = 10
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_splits: int = 5 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200 # this loss need pretraining, not just long training
    name: str='myopic E[  log p(y|y_init)  ]'

############
# GP Mutual Information loss
#
class GPMI1LossConfig(BasicNonMyopicLossConfig):
    batch_size: int = 10
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_grid_points: int = 500
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='E[  -H(y|y_init) + H(y|y_grid, y_init)  ]'

class GPMI2LossConfig(BasicNonMyopicLossConfig):
    batch_size: int = 10
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_grid_points: int = 500
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='E[  log p(y|y_init) - log p(y|y_grid, y_init)  ]'

class GPPCELossConfig(BasicNonMyopicLossConfig):
    batch_size: int = 25
    num_kernels: int = 10
    num_functions_per_kernel: int = 25
    name: str='gp_mi_pce_loss'

############
class GPMyopicMI1LossConfig(BasicMyopicLossConfig):
    batch_size: int = 10
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_grid_points: int = 100
    num_splits: int = 5 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='myopic E[  -H(y|y_init) + H(y|y_grid, y_init)  ]'

class GPMyopicMI2LossConfig(BasicMyopicLossConfig):
    batch_size: int = 10
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_grid_points: int = 100
    num_splits: int = 5 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='myopic E[  log p(y|y_init) - log p(y|y_grid, y_init)  ]'

############
# Curriculum of losses
#
class GPMI_Entropy1LossConfig(BasicLossCurriculumConfig):
    loss_config_list: List[Union[BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig]] = [
        GPMI1LossConfig(num_epochs = 75),
        GPEntropy1LossConfig(num_epochs = 125),
    ]
    name: str='mi_and_entropy_curriculum_version2'

class GPMI_Entropy2LossConfig(BasicLossCurriculumConfig):
    loss_config_list: List[Union[BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig]] = [
        GPMI2LossConfig(num_epochs = 75),
        GPEntropy2LossConfig(num_epochs = 125),
    ]
    name: str='mi_and_entropy_curriculum'



