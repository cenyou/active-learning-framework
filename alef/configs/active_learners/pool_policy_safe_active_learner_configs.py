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
from alef.enums.active_learner_enums import ValidationType
from pydantic_settings import BaseSettings
import json

class BasicPoolPolicySafeActiveLearnerConfig(BaseSettings):
    constraint_on_y: bool
    validation_type: Union[ValidationType, List[ValidationType]] = [ValidationType.MAE, ValidationType.RMSE, ValidationType.NEG_LOG_LIKELI]
    validation_at: Optional[List[int]] = None
    train_with_safe_data_only: List[bool]=[False]
    policy_path: str = ''

class PytestPoolPolicySafeActiveLearnerConfig(BasicPoolPolicySafeActiveLearnerConfig):
    pytest: bool = True
    policy_dimension: int

if __name__ == "__main__":
    pass
