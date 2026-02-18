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

from typing import List, Optional, Union
from alef.configs.acquisition_functions.al_acquisition_functions.base_al_acquisition_function_config import BaseALAcquisitionFunctionConfig
from alef.configs.acquisition_functions.al_acquisition_functions.pred_sigma_config import BasicPredSigmaConfig
from alef.configs.acquisition_functions.al_acquisition_functions.pred_variance_config import BasicPredVarianceConfig
from alef.configs.acquisition_functions.al_acquisition_functions.pred_entropy_config import BasicPredEntropyConfig
from alef.configs.acquisition_functions.al_acquisition_functions.acq_random_config import BasicRandomConfig
from alef.enums.active_learner_enums import ValidationType
from pydantic import BaseSettings
import json


class BasicPoolActiveLearnerConfig(BaseSettings):
    acquisition_function_config: BaseALAcquisitionFunctionConfig
    validation_type: Union[ValidationType, List[ValidationType]] = [ValidationType.MAE, ValidationType.RMSE, ValidationType.NEG_LOG_LIKELI]
    validation_at: Optional[List[int]] = None
    use_smaller_acquistion_set: bool = False
    smaller_set_size: int = 0


class PredVarPoolActiveLearnerConfig(BasicPoolActiveLearnerConfig):
    acquisition_function_config: BaseALAcquisitionFunctionConfig = BasicPredVarianceConfig()


class PredSigmaPoolActiveLearnerConfig(BasicPoolActiveLearnerConfig):
    acquisition_function_config: BaseALAcquisitionFunctionConfig = BasicPredSigmaConfig()


class PredEntropyPoolActiveLearnerConfig(BasicPoolActiveLearnerConfig):
    acquisition_function_config: BaseALAcquisitionFunctionConfig = BasicPredEntropyConfig()
    use_smaller_acquistion_set: bool = True
    smaller_set_size: int = 200


class RandomPoolActiveLearnerConfig(BasicPoolActiveLearnerConfig):
    acquisition_function_config: BaseALAcquisitionFunctionConfig = BasicRandomConfig()


if __name__ == "__main__":
    print(type(json.loads(PredEntropyPoolActiveLearnerConfig().json())))
