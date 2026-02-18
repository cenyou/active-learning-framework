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

import math
import numpy as np

NUMERICAL_POSITIVE_LOWER_BOUND = 1e-5
NUMERICAL_POSITIVE_LOWER_BOUND_IN_LOG = math.log(NUMERICAL_POSITIVE_LOWER_BOUND)

INPUT_DOMAIN = (0.0, 1.0)
CENTRAL_DOMAIN = (0.6*INPUT_DOMAIN[0] + 0.4*INPUT_DOMAIN[1], 0.4*INPUT_DOMAIN[0] + 0.6*INPUT_DOMAIN[1])

BASE_KERNEL_VARIANCE = 1.0
BASE_KERNEL_LENGTHSCALE = 0.15
BASE_RQ_KERNEL_ALPHA = 1.0
BASE_LINEAR_KERNEL_OFFSET = 1.0
BASE_KERNEL_PERIOD = 1.0

NOISE_VARIANCE_LOWER_BOUND = 1e-4
