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

from typing import List, Tuple, Union
from pydantic import BaseSettings
from .base_loss_configs import (
    BasicAmortizedPolicyLossConfig,
    #
    BasicSafetyAwarePolicyLossConfig,
    BasicNonMyopicSafetyLossConfig,
    BasicMyopicSafetyLossConfig,
    #
    BasicLossCurriculumConfig,
)
from .information_loss_configs import (
    GPEntropy1LossConfig,
    GPEntropy2LossConfig,
    GPMI1LossConfig,
    GPMI2LossConfig,
    #GPPCELossConfig,
)
from alef.enums.active_learner_amortized_policy_enums import SafetyProbability, SafetyProbabilityWrapper
_myopic_loss_list = [
    'SafeGPMyopicEntropy1LossConfig', 'SafeProductWrapGPMyopicEntropy1LossConfig', 'MinUnsafeGPMyopicEntropy1LossConfig',
    'TrivialSafeGPMyopicEntropy1LossConfig',
    'SafeGPMyopicEntropy2LossConfig', 'SafeProductWrapGPMyopicEntropy2LossConfig', 'MinUnsafeGPMyopicEntropy2LossConfig',
    'TrivialSafeGPMyopicEntropy2LossConfig',
    'SafeGPMyopicMI1LossConfig', 'SafeProductWrapGPMyopicMI1LossConfig', 'MinUnsafeGPMyopicMI1LossConfig',
    'TrivialSafeGPMyopicMI1LossConfig',
    'SafeGPMyopicMI2LossConfig', 'SafeProductWrapGPMyopicMI2LossConfig', 'MinUnsafeGPMyopicMI2LossConfig',
    'TrivialSafeGPMyopicMI2LossConfig',
]
_nonmyopic_loss_list = [
    'SafeGPEntropy1LossConfig', 'SafeProductWrapGPEntropy1LossConfig', 'MinUnsafeGPEntropy1LossConfig',
    'TrivialSafeGPEntropy1LossConfig',
    'SafeGPEntropy2LossConfig', 'SafeProductWrapGPEntropy2LossConfig', 'MinUnsafeGPEntropy2LossConfig',
    'TrivialSafeGPEntropy2LossConfig',
    'SafeGPMI1LossConfig', 'SafeProductWrapGPMI1LossConfig', 'MinUnsafeGPMI1LossConfig',
    'TrivialSafeGPMI1LossConfig',
    'SafeGPMI2LossConfig', 'SafeProductWrapGPMI2LossConfig', 'MinUnsafeGPMI2LossConfig',
    'TrivialSafeGPMI2LossConfig',
    #'SafeGPPCELossConfig', # !!! this loss is under development
    #'TrivialSafeGPPCELossConfig', # !!! this loss is under development
    #'SafeMI_SafeEntropy1LossConfig',
    #'SafeMI_SafeEntropy2LossConfig',
    #'GPMI_SafeMI_SafeEntropy1LossConfig',
    #'GPMI_SafeMI_SafeEntropy2LossConfig',
]

__all__ = _myopic_loss_list + _nonmyopic_loss_list
"""
Trivial* should compute the same loss as without safety constraint, but using safe AL simulation pipeline.
"""
############
# GP Entropy loss
#

class SafeGPEntropy1LossConfig(BasicNonMyopicSafetyLossConfig):
    batch_size: int = 1
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='E[  -H(y|y_init)  +  neg_log_safe_prob  ]'

class SafeProductWrapGPEntropy1LossConfig(SafeGPEntropy1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT
    safety_discount_ratio: float = 1.0
    name: str='safe_product_wrap_entropy_version2'

class MinUnsafeGPEntropy1LossConfig(SafeGPEntropy1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.JOINTPROBABILITY
    safety_discount_ratio: float = 1.0
    name: str='E[  -H(y|y_init)  +  log_unsafe_prob  ]'

class TrivialSafeGPEntropy1LossConfig(SafeGPEntropy1LossConfig):
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    name: str='E[  -H(y|y_init)  ]'

class SafeGPEntropy2LossConfig(BasicNonMyopicSafetyLossConfig):
    batch_size: int = 1
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='E[  log p(y|y_init)  +  neg_log_safe_prob  ]'

class SafeProductWrapGPEntropy2LossConfig(SafeGPEntropy2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT
    safety_discount_ratio: float = 1.0
    name: str='safe_product_wrap_entropy'

class MinUnsafeGPEntropy2LossConfig(SafeGPEntropy2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.JOINTPROBABILITY
    safety_discount_ratio: float = 1.0
    name: str='E[  log p(y|y_init)  +  log_unsafe_prob  ]'

class TrivialSafeGPEntropy2LossConfig(SafeGPEntropy2LossConfig):
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    name: str='E[  log p(y|y_init)  ]'

############
class SafeGPMyopicEntropy1LossConfig(BasicMyopicSafetyLossConfig):
    batch_size: int = 1
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_splits: int = 20 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200 # this loss need pretraining, not just long training
    name: str='myopic E[  -H(y|y_init)  +  neg_log_safe_prob  ]'

class SafeProductWrapGPMyopicEntropy1LossConfig(SafeGPMyopicEntropy1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT
    safety_discount_ratio: float = 1.0
    name: str='safe_product_wrap_myopic_entropy_version2'

class MinUnsafeGPMyopicEntropy1LossConfig(SafeGPMyopicEntropy1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.JOINTPROBABILITY
    safety_discount_ratio: float = 1.0
    name: str='myopic E[  -H(y|y_init)  +  log_unsafe_prob  ]'

class TrivialSafeGPMyopicEntropy1LossConfig(SafeGPMyopicEntropy1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.NONE
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    num_splits: int = 20 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    name: str='myopic E[  -H(y|y_init)  ]'

class SafeGPMyopicEntropy2LossConfig(BasicMyopicSafetyLossConfig):
    batch_size: int = 1
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_splits: int = 20 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200 # this loss need pretraining, not just long training
    name: str='myopic E[  log p(y|y_init)  +  neg_log_safe_prob  ]'

class SafeProductWrapGPMyopicEntropy2LossConfig(SafeGPMyopicEntropy2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT
    safety_discount_ratio: float = 1.0
    name: str='safe_product_wrap_myopic_entropy'

class MinUnsafeGPMyopicEntropy2LossConfig(SafeGPMyopicEntropy2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.JOINTPROBABILITY
    safety_discount_ratio: float = 1.0
    name: str='myopic E[  log p(y|y_init)  +  log_unsafe_prob  ]'

class TrivialSafeGPMyopicEntropy2LossConfig(SafeGPMyopicEntropy2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.NONE
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    num_splits: int = 20 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    name: str='myopic E[  log p(y|y_init)  ]'

############
# GP Mutual Information loss
#
class SafeGPMI1LossConfig(BasicNonMyopicSafetyLossConfig):
    batch_size: int = 1
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_grid_points: int = 200
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='E[  -H(y|y_init) + H(y|y_grid, y_init)  +  neg_log_safe_prob  ]'

class SafeProductWrapGPMI1LossConfig(SafeGPMI1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT
    safety_discount_ratio: float = 1.0
    name: str='safe_product_wrap_regularized_entropy_version2'

class MinUnsafeGPMI1LossConfig(SafeGPMI1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.JOINTPROBABILITY
    safety_discount_ratio: float = 1.0
    name: str='E[  -H(y|y_init) + H(y|y_grid, y_init)  +  log_unsafe_prob  ]'

class TrivialSafeGPMI1LossConfig(SafeGPMI1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.NONE
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    name: str='E[  -H(y|y_init) + H(y|y_grid, y_init)  ]'

class SafeGPMI2LossConfig(BasicNonMyopicSafetyLossConfig):
    batch_size: int = 1
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_grid_points: int = 200
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='E[  log p(y|y_init) - log p(y|y_grid, y_init)  +  neg_log_safe_prob  ]'

class SafeProductWrapGPMI2LossConfig(SafeGPMI2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT
    safety_discount_ratio: float = 1.0
    name: str='safe_product_wrap_regularized_entropy'

class MinUnsafeGPMI2LossConfig(SafeGPMI2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.JOINTPROBABILITY
    safety_discount_ratio: float = 1.0
    name: str='E[  log p(y|y_init) - log p(y|y_grid, y_init)  +  log_unsafe_prob  ]'

class TrivialSafeGPMI2LossConfig(SafeGPMI2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.NONE
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    name: str='E[  log p(y|y_init) - log p(y|y_grid, y_init)  ]'

class SafeGPPCELossConfig(BasicNonMyopicSafetyLossConfig):
    batch_size: int = 1
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 400
    name: str='safe_gp_mi_pce_loss'

class TrivialSafeGPPCELossConfig(SafeGPPCELossConfig):
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    name: str='trivial_safe_gp_mi_pce_loss'

############
class SafeGPMyopicMI1LossConfig(BasicMyopicSafetyLossConfig):
    batch_size: int = 1
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_grid_points: int = 200
    num_splits: int = 20 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='myopic E[  -H(y|y_init) + H(y|y_grid, y_init)  +  neg_log_safe_prob  ]'

class SafeProductWrapGPMyopicMI1LossConfig(SafeGPMyopicMI1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT
    safety_discount_ratio: float = 1.0
    name: str='safe_product_wrap_myopic_regularized_entropy_version2'

class MinUnsafeGPMyopicMI1LossConfig(SafeGPMyopicMI1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.JOINTPROBABILITY
    safety_discount_ratio: float = 1.0
    name: str='myopic E[  -H(y|y_init) + H(y|y_grid, y_init)  +  log_unsafe_prob  ]'

class TrivialSafeGPMyopicMI1LossConfig(SafeGPMyopicMI1LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.NONE
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    num_splits: int = 20 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    name: str='myopic E[  -H(y|y_init) + H(y|y_grid, y_init)  ]'

class SafeGPMyopicMI2LossConfig(BasicMyopicSafetyLossConfig):
    batch_size: int = 1
    num_kernels: int = 10
    num_functions_per_kernel: int = 5
    num_grid_points: int = 200
    num_splits: int = 20 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 200
    name: str='myopic E[  log p(y|y_init) - log p(y|y_grid, y_init)  +  neg_log_safe_prob  ]'

class SafeProductWrapGPMyopicMI2LossConfig(SafeGPMyopicMI2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.PRODUCT
    safety_discount_ratio: float = 1.0
    name: str='safe_product_wrap_myopic_regularized_entropy'

class MinUnsafeGPMyopicMI2LossConfig(SafeGPMyopicMI2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.JOINTPROBABILITY
    safety_discount_ratio: float = 1.0
    name: str='myopic E[  log p(y|y_init) - log p(y|y_grid, y_init)  +  log_unsafe_prob  ]'

class TrivialSafeGPMyopicMI2LossConfig(SafeGPMyopicMI2LossConfig):
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.NONE
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    num_splits: int = 20 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    name: str='myopic E[  log p(y|y_init) - log p(y|y_grid, y_init)  ]'

############
# Curriculum of losses
#
class SafeMI_SafeEntropy1LossConfig(BasicLossCurriculumConfig):
    loss_config_list: List[Union[BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig]] = [
        SafeGPMI1LossConfig(num_epochs = 125),
        SafeGPEntropy1LossConfig(num_epochs = 200),
    ]
    name: str='safe_mi_and_safe_entropy_curriculum_version2'

class SafeMI_SafeEntropy2LossConfig(BasicLossCurriculumConfig):
    loss_config_list: List[Union[BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig]] = [
        SafeGPMI2LossConfig(num_epochs = 125),
        SafeGPEntropy2LossConfig(num_epochs = 200),
    ]
    name: str='safe_mi_and_safe_entropy_curriculum'

class GPMI_SafeMI_SafeEntropy1LossConfig(BasicLossCurriculumConfig):
    loss_config_list: List[Union[BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig]] = [
        TrivialSafeGPMI1LossConfig(num_epochs = 75),
        SafeGPMI1LossConfig(num_epochs = 125),
        SafeGPEntropy1LossConfig(num_epochs = 200),
    ]
    name: str='mi_safe_mi_and_safe_entropy_curriculum_version2'

class GPMI_SafeMI_SafeEntropy2LossConfig(BasicLossCurriculumConfig):
    loss_config_list: List[Union[BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig]] = [
        TrivialSafeGPMI2LossConfig(num_epochs = 75),
        SafeGPMI2LossConfig(num_epochs = 125),
        SafeGPEntropy2LossConfig(num_epochs = 200),
    ]
    name: str='mi_safe_mi_and_safe_entropy_curriculum'



