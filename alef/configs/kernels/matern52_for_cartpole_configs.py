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
from alef.configs.kernels.matern52_configs import Matern52WithPriorConfig


class CartpoleMatern52Config(Matern52WithPriorConfig):
    input_dimension: int = 5
    base_lengthscale: Union[float, Sequence[float]] = [1.47, 0.88, 3.76, 0.66, 0.56]
    base_variance: float = 1.0
    fix_lengthscale: bool = True
    fix_variance: bool = True
    add_prior: bool = False
    name = "CartpoleMatern52"
class CartpoleReduceMatern52Config(CartpoleMatern52Config):
    input_dimension: int = 4
    base_lengthscale: Union[float, Sequence[float]] = [1.47, 0.88, 3.76, 0.66]
    name = "CartpoleReduceMatern52"


class CartpoleMatern52SafetyConfig(Matern52WithPriorConfig):
    input_dimension: int = 5
    base_lengthscale: Union[float, Sequence[float]] = [1.18e+04, 5.25e-02, 2.83e+04, 2.06e-01, 8.08e+03]
    base_variance: float = 1.0
    fix_lengthscale: bool = True
    fix_variance: bool = True
    add_prior: bool = False
    name = "CartpoleMatern52Safety"
class CartpoleReduceMatern52SafetyConfig(CartpoleMatern52SafetyConfig):
    input_dimension: int = 4
    base_lengthscale: Union[float, Sequence[float]] = [1.18e+04, 5.25e-02, 2.83e+04, 2.06e-01]
    name = "CartpoleReduceMatern52Safety"


