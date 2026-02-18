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

import torch
from typing import Union, Sequence, Tuple, Optional
from .base_mean_pytorch_config import BaseMeanPytorchConfig
from alef.configs.base_parameters import INPUT_DOMAIN
from alef.enums.gpytorch_enums import GPytorchPriorEnum

class BasicSechMeanPytorchConfig(BaseMeanPytorchConfig):
    input_dimension: int
    center: Union[float, Sequence[float]] = (INPUT_DOMAIN[1] - INPUT_DOMAIN[0]) / 2
    weights: Union[float, Sequence[float]] = 20.0
    scale: float = 3.2
    bias: float = -0.47
    center_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = None #(GPytorchPriorEnum.UNIFORM, [0.35, 0.65]) #(GPytorchPriorEnum.CATEGORICAL, [ [0.35, 0.5, 0.65], [1,1,1] ])
    weights_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = (GPytorchPriorEnum.UNIFORM, [5, 40] ) # UniformPrior(3, 10)
    scale_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = None
    bias_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = None
    name: str = "SechMean"

class BasicSechRotatedMeanPytorchConfig(BaseMeanPytorchConfig):
    input_dimension: int
    center: Union[float, Sequence[float]] = (INPUT_DOMAIN[1] - INPUT_DOMAIN[0]) / 2
    weights: Union[float, Sequence[float]] = 20.0
    scale: float = 3.2
    bias: float = -0.47
    center_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = None #(GPytorchPriorEnum.UNIFORM, [0.35, 0.65]) #(GPytorchPriorEnum.CATEGORICAL, [ [0.35, 0.5, 0.65], [1,1,1] ])
    weights_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = (GPytorchPriorEnum.UNIFORM, [5, 40] ) # UniformPrior(3, 10)
    axis_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = (GPytorchPriorEnum.UNIFORM, [-1, 1] )
    scale_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = None
    bias_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = None
    name: str = "SechMean"
