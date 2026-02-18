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

import numpy as np
from typing import Union, Sequence
from alef.configs.acquisition_functions.safe_acquisition_functions.base_safe_acquisition_function_config import (
    BaseSafeAcquisitionFunctionConfig,
)


class BasicMinUnsafePredEntropyConfig(BaseSafeAcquisitionFunctionConfig):
    safety_thresholds_lower: Union[float, Sequence[float]]
    safety_thresholds_upper: Union[float, Sequence[float]]
    alpha: float = 0.02275
    lagrange_multiplier: float = 1.0
    name: str = "min_unsafe_pred_entropy"

class MinUnsafePredEntropyLambda01Config(BasicMinUnsafePredEntropyConfig):
    lagrange_multiplier: float = 0.1
    name: str = "min_unsafe_pred_entropy_lambda_0.1"

class MinUnsafePredEntropyLambda05Config(BasicMinUnsafePredEntropyConfig):
    lagrange_multiplier: float = 0.5
    name: str = "min_unsafe_pred_entropy_lambda_0.5"

class MinUnsafePredEntropyLambda09Config(BasicMinUnsafePredEntropyConfig):
    lagrange_multiplier: float = 0.9
    name: str = "min_unsafe_pred_entropy_lambda_0.9"

class MinUnsafePredEntropyLambda2Config(BasicMinUnsafePredEntropyConfig):
    lagrange_multiplier: float = 2.0
    name: str = "min_unsafe_pred_entropy_lambda_2"

class MinUnsafePredEntropyLambda3Config(BasicMinUnsafePredEntropyConfig):
    lagrange_multiplier: float = 3.0
    name: str = "min_unsafe_pred_entropy_lambda_3"

class MinUnsafePredEntropyLambda4Config(BasicMinUnsafePredEntropyConfig):
    lagrange_multiplier: float = 4.0
    name: str = "min_unsafe_pred_entropy_lambda_4"

class MinUnsafePredEntropyLambda5Config(BasicMinUnsafePredEntropyConfig):
    lagrange_multiplier: float = 5.0
    name: str = "min_unsafe_pred_entropy_lambda_5"

class MinUnsafePredEntropyLambda10Config(BasicMinUnsafePredEntropyConfig):
    lagrange_multiplier: float = 10.0
    name: str = "min_unsafe_pred_entropy_lambda_10"

class MinUnsafePredEntropyLambda100Config(BasicMinUnsafePredEntropyConfig):
    lagrange_multiplier: float = 100.0
    name: str = "min_unsafe_pred_entropy_lambda_100"

##############################################################################
class BasicMinUnsafePredEntropyAllConfig(BaseSafeAcquisitionFunctionConfig):
    safety_thresholds_lower: Union[float, Sequence[float]]
    safety_thresholds_upper: Union[float, Sequence[float]]
    alpha: float = 0.02275
    lagrange_multiplier: float = 1.0
    name: str = "min_unsafe_pred_entropy_all"
