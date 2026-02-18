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
from alef.configs.means.pytorch_means.sech_mean_pytorch_config import BasicSechRotatedMeanPytorchConfig
from alef.enums.gpytorch_enums import GPytorchPriorEnum

class HighPressureFluidSystemSechRotatedMeanPytorchConfig(BasicSechRotatedMeanPytorchConfig):
    input_dimension: int = 7
    center: Union[float, Sequence[float]] = 0.5
    weights_prior: Optional[Tuple[GPytorchPriorEnum, Sequence]] = (
        GPytorchPriorEnum.UNIFORM,
        [[10.42, 1.57, 6.02, 8.33, 16.66, 1.47, 11.36],
        [41.66, 6.29, 24.1, 33.34, 66.67, 5.89, 45.45]]
    )
    name: str = "HPFSSechMean"