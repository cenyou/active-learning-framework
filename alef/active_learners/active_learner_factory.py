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

from typing import Union
from alef.acquisition_functions.acquisition_function_factory import AcquisitionFunctionFactory
from alef.active_learners.pool_active_learner import PoolActiveLearner
from alef.active_learners.oracle_active_learner import OracleActiveLearner

from alef.configs.active_learners.pool_active_learner_configs import BasicPoolActiveLearnerConfig
from alef.configs.active_learners.oracle_active_learner_configs import BasicOracleActiveLearnerConfig


class ActiveLearnerFactory:
    @staticmethod
    def build(active_learner_config: Union[BasicPoolActiveLearnerConfig, BasicOracleActiveLearnerConfig]):
        if isinstance(active_learner_config, BasicPoolActiveLearnerConfig):
            acquisition_function_config = active_learner_config.acquisition_function_config
            acquisition_function = AcquisitionFunctionFactory.build(acquisition_function_config)
            return PoolActiveLearner(acquisition_function=acquisition_function, **active_learner_config.dict())
        elif isinstance(active_learner_config, BasicOracleActiveLearnerConfig):
            acquisition_function_config = active_learner_config.acquisition_function_config
            acquisition_function = AcquisitionFunctionFactory.build(acquisition_function_config)
            return OracleActiveLearner(acquisition_function=acquisition_function, **active_learner_config.dict())
        else:
            raise NotImplementedError("Invalid config")
