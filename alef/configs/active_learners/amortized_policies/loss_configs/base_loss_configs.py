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

from typing import List, Union, Tuple
from pydantic_settings import BaseSettings
from alef.enums.active_learner_amortized_policy_enums import SafetyProbability, SafetyProbabilityWrapper

__all__ = [
    'BasicAmortizedPolicyLossConfig',
    'BasicNonMyopicLossConfig',
    'BasicMyopicLossConfig',
    'BasicSafetyAwarePolicyLossConfig',
    'BasicNonMyopicSafetyLossConfig',
    'BasicMyopicSafetyLossConfig',
    'BasicLossCurriculumConfig',
]

class BasicAmortizedPolicyLossConfig(BaseSettings):
    batch_size: int
    num_kernels: int
    num_functions_per_kernel: int
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 400
    epochs_size: int = 50
    name: str='basic_amortized_policy_loss'

class BasicNonMyopicLossConfig(BasicAmortizedPolicyLossConfig):
    name: str='nonmyopic_loss'

class BasicMyopicLossConfig(BasicAmortizedPolicyLossConfig):
    name: str='myopic_loss'


############
class BasicSafetyAwarePolicyLossConfig(BaseSettings):
    probability_function: SafetyProbability = SafetyProbability.TRIVIAL
    probability_function_args: Tuple[float, float] = (0.05, -0.05) # (alpha, half_prob_loc): p(0)=1-alpha, p(half_prob_loc)=0.5
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.LOGCONDITION
    safety_discount_ratio: float = 1.0 # if LOGCONDITION, loss = information_loss - ratio * log p(z)
    batch_size: int
    num_kernels: int
    num_functions_per_kernel: int
    num_splits: int = 1 # this parameter control the number of runs we do in 'sequence' (split kernel batch into multiple chunks)
    num_epochs: int = 400
    epochs_size: int = 50
    name: str='basic_amortized_policy_safety_loss'

safety_prob_args = {
    'SIGMOID': (0.95, 0.5), # (alpha, half_prob_loc): p(0)= 1 - alpha, p(half_prob_loc)=0.5
    'SIGMOID_SOFTPLUS': (3.0, 3.0),
    'GP_POSTERIOR': (0.05, 0.000), # (alpha, useless_value), p(z>=0) < 1-alpha is unsafe
}
class BasicNonMyopicSafetyLossConfig(BasicSafetyAwarePolicyLossConfig):
    probability_function: SafetyProbability = SafetyProbability.GP_POSTERIOR
    probability_function_args: Tuple[float, float] = safety_prob_args['GP_POSTERIOR']
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.LOGCONDITION
    safety_discount_ratio: float = 1.0 # if LOGCONDITION, loss = information_loss - ratio * log p(z)
    name: str='nonmyopic_safety_loss'

class BasicMyopicSafetyLossConfig(BasicSafetyAwarePolicyLossConfig):
    probability_function: SafetyProbability = SafetyProbability.GP_POSTERIOR
    probability_function_args: Tuple[float, float] = safety_prob_args['GP_POSTERIOR']
    probability_wrap_mode: SafetyProbabilityWrapper = SafetyProbabilityWrapper.LOGCONDITION
    safety_discount_ratio: float = 1.0 # if LOGCONDITION, loss = information_loss - ratio * log p(z)
    name: str='myopic_safety_loss'


############
class BasicLossCurriculumConfig(BaseSettings):
    loss_config_list: List[Union[BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig]]
    name: str='basic_amortized_policy_curriculum'
