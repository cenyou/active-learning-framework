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

from typing import Tuple, Union, Sequence
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig


class HighPressureFluidSystemRBFPytorchConfig(RBFWithPriorPytorchConfig):
    input_dimension: int = 7
    base_lengthscale: Union[float, Sequence[float]] = [0.48, 3.18, 0.83, 0.6, 0.3, 3.4, 0.44]
    base_variance: float = 1.0
    name = "HighPressureFluidSystemRBF"

class HighPressureFluidSystemReduceRBFPytorchConfig(RBFWithPriorPytorchConfig):
    input_dimension: int = 4
    base_lengthscale: Union[float, Sequence[float]] = [0.48, 0.6, 0.3, 0.44] # [0.48, 0.83, 0.6, 0.3, 0.44]
    base_variance: float = 1.0
    name = "HighPressureFluidSystemReduceRBF"


