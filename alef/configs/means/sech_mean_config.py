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

from typing import Union, Sequence, Tuple
from .base_mean_config import BaseMeanConfig
from alef.configs.base_parameters import INPUT_DOMAIN

class BasicSechMeanConfig(BaseMeanConfig):
    input_dimension: int
    center: Union[float, Sequence[float]] = (INPUT_DOMAIN[1] - INPUT_DOMAIN[0]) / 2
    weights: Union[float, Sequence[float]] = 20.0
    scale: float = 3.2
    bias: float = -0.47
    name: str = "SechMean"