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

from typing import List, Dict, Union, Optional
from pydantic_settings import BaseSettings

import json
import torch

from .loss_configs import BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig, BasicLossCurriculumConfig
from .policy_configs import BaseAmortizedPolicyConfig
from alef.configs.means.pytorch_means import BaseMeanPytorchConfig, BasicZeroMeanPytorchConfig
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.enums.active_learner_amortized_policy_enums import LengthscaleDistribution

__all__ = [
    'AmortizedContinuousFixGPPolicyTrainingConfig',
    'AmortizedContinuousRandomGPPolicyTrainingConfig',
]

class BaseAmortizedPolicyTrainingConfig(BaseSettings):
    optimizer: torch.optim.Optimizer = torch.optim.RAdam #torch.optim.Adam
    optim_args: Dict= {"lr": 1e-4} #{"lr": 1e-5, "betas": [0.9, 0.999], "weight_decay": 0,}
    gamma: float = 0.98
    policy_config: BaseAmortizedPolicyConfig
    loss_config: Union[BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig, BasicLossCurriculumConfig]

class AmortizedContinuousFixGPPolicyTrainingConfig(BaseAmortizedPolicyTrainingConfig):
    kernel_config: BaseKernelPytorchConfig
    mean_config: Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]] = [BasicZeroMeanPytorchConfig(batch_shape=[])]
    safety_kernel_config: Optional[BaseKernelPytorchConfig] = None
    safety_mean_config: Optional[Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]]] = None
    n_initial_min: int
    n_initial_max: Optional[int] = None
    n_steps_min: Optional[int] = None
    n_steps_max: int
    sample_gp_prior: bool = False
    lengthscale_distribution: LengthscaleDistribution = LengthscaleDistribution.GAMMA
    random_subsequence: bool = False
    split_subsequence: bool = False
    optim_args: Dict= {"lr": 1e-4} #{"lr": 1e-5, "betas": [0.9, 0.999], "weight_decay": 0,}
    gamma: float = 0.98

class AmortizedContinuousRandomGPPolicyTrainingConfig(BaseAmortizedPolicyTrainingConfig):
    kernel_config: BaseKernelPytorchConfig
    mean_config: Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]] = [BasicZeroMeanPytorchConfig(batch_shape=[])]
    safety_kernel_config: Optional[BaseKernelPytorchConfig] = None
    safety_mean_config: Optional[Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]]] = None
    n_initial_min: int
    n_initial_max: Optional[int] = None
    n_steps_min: Optional[int] = None
    n_steps_max: int
    sample_gp_prior: bool = True
    lengthscale_distribution: LengthscaleDistribution = LengthscaleDistribution.GAMMA
    random_subsequence: bool = False
    split_subsequence: bool = False
    optim_args: Dict= {"lr": 1e-4}
    gamma: float = 0.98


