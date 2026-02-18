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

from typing import Union, Sequence, Tuple, Optional
from .base_mean_pytorch_config import BaseMeanPytorchConfig
from alef.enums.gpytorch_enums import GPytorchPriorEnum

class BasicLinearMeanPytorchConfig(BaseMeanPytorchConfig):
    input_dimension: int
    weights: Union[float, Sequence[float]] = 1.0
    scale: float = 2.5
    bias: float = -1.25
    weights_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = None # (prior_type, [*args, **kwargs])
    scale_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = None
    bias_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = None
    name: str = "LinearMean"