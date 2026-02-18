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
import gpflow
from gpflow.config import default_float
from alef.configs.means.base_mean_config import BaseMeanConfig
from alef.configs.means.zero_mean_config import BasicZeroMeanConfig
from alef.configs.means.linear_mean_config import BasicLinearMeanConfig
from alef.configs.means.quadratic_mean_config import BasicQuadraticMeanConfig
from alef.means.quadratic_mean import QuadraticMean
from alef.configs.means.periodic_mean_config import BasicPeriodicMeanConfig
from alef.means.periodic_mean import PeriodicMean
from alef.configs.means.sech_mean_config import BasicSechMeanConfig
from alef.means.sech_mean import SechMean

class MeanFactory:
    @staticmethod
    def build(mean_config: BaseMeanConfig):
        if isinstance(mean_config, BasicZeroMeanConfig):
            return gpflow.mean_functions.Zero()
        if isinstance(mean_config, BasicLinearMeanConfig):
            D = mean_config.input_dimension
            A = np.array(mean_config.scale) * np.ones([D], dtype=default_float())
            b = np.array(mean_config.bias) * np.ones(1, dtype=default_float())
            m = gpflow.mean_functions.Linear(A[..., None], b)
            return m
        elif isinstance(mean_config, BasicQuadraticMeanConfig):
            D = mean_config.input_dimension
            m = QuadraticMean(input_size=D)
            m.scale.assign(
                np.array(mean_config.scale) * np.ones(1, dtype=default_float())
            )
            m.bias.assign( np.array(mean_config.bias) * np.ones(1, dtype=default_float()) )
            m.center.assign( np.array(mean_config.center) * np.ones([D], dtype=default_float()) )
            m.weights.assign( (np.array(mean_config.weights) * np.ones([D], dtype=default_float()))[..., None] )
            return m
        elif isinstance(mean_config, BasicPeriodicMeanConfig):
            D = mean_config.input_dimension
            m = PeriodicMean(input_size=D)
            m.scale.assign(
                np.array(mean_config.scale) * np.ones(1, dtype=default_float())
            )
            m.bias.assign( np.array(mean_config.bias) * np.ones(1, dtype=default_float()) )
            m.center.assign( np.array(mean_config.center) * np.ones([D], dtype=default_float()) )
            m.weights.assign( (np.array(mean_config.weights) * np.ones([D], dtype=default_float()))[..., None] )
            return m
        elif isinstance(mean_config, BasicSechMeanConfig):
            D = mean_config.input_dimension
            m = SechMean(input_size=D)
            m.scale.assign(
                np.array(mean_config.scale) * np.ones(1, dtype=default_float())
            )
            m.bias.assign( np.array(mean_config.bias) * np.ones(1, dtype=default_float()) )
            m.center.assign( np.array(mean_config.center) * np.ones([D], dtype=default_float()) )
            m.weights.assign( (np.array(mean_config.weights) * np.ones([D], dtype=default_float()))[..., None] )
            return m
        else:
            raise NotImplementedError(f"invalid mean config {mean_config}")


