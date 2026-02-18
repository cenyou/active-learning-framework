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


class CartpoleRBFPytorchConfig(RBFWithPriorPytorchConfig):
    input_dimension: int = 5
    base_lengthscale: Union[float, Sequence[float]] = [0.95, 0.78, 4.80e+04, 0.38, 0.66]
    base_variance: float = 1.0
    name = "CartpoleRBF"
class CartpoleReduceRBFPytorchConfig(CartpoleRBFPytorchConfig):
    input_dimension: int = 4
    base_lengthscale: Union[float, Sequence[float]] = [0.95, 0.78, 4.80e+04, 0.38]
    name = "CartpoleReduceRBF"

class CartpoleRBFSafetyPytorchConfig(RBFWithPriorPytorchConfig):
    input_dimension: int = 5
    base_lengthscale: Union[float, Sequence[float]] = [0.18, 0.27, 1.5 , 0.03, 0.98]
    base_variance: float = 1.0
    name = "CartpoleRBFSafety"
class CartpoleReduceRBFSafetyPytorchConfig(CartpoleRBFSafetyPytorchConfig):
    input_dimension: int = 4
    base_lengthscale: Union[float, Sequence[float]] = [0.18, 0.27, 1.5 , 0.03]
    name = "CartpoleReduceRBFSafety"


