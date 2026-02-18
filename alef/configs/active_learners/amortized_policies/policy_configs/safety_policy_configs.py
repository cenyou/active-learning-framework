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

from typing import Tuple, Sequence, Union
from pydantic import BaseSettings


from alef.configs.base_parameters import INPUT_DOMAIN
from alef.enums.active_learner_amortized_policy_enums import DomainWarpperType

from .base_policy_configs import BaseAmortizedPolicyConfig


class SafetyAwareContinuousGPPolicyConfig(BaseAmortizedPolicyConfig):
    input_dim: int
    observation_dim: int = 1
    safety_dim: int = 1
    hidden_dim_encoder: Sequence[int] = [512]
    encoding_dim: int = 128
    hidden_dim_emitter: Sequence[int] = [512]
    input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN
    num_self_attention_layer: int=2
    domain_warpper: DomainWarpperType=DomainWarpperType.TANH
    device: str = 'cpu'
    name: str = 'basic_continuous_gp_safe_al_policy_config'

class SafetyAwareContinuousGPFlexDimPolicyConfig(BaseAmortizedPolicyConfig):
    encoding_dim: int = 32
    num_self_attention_layer: int=2
    attend_sequence_first: bool=True
    hidden_dim_budget_encoder: Sequence[int] = [512]
    hidden_dim_emitter: Sequence[int] = [512]
    input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN
    domain_warpper: DomainWarpperType=DomainWarpperType.TANH
    device: str = 'cpu'
    name: str = 'continuous_gp_safe_al_flexible_dimension_policy_config'

