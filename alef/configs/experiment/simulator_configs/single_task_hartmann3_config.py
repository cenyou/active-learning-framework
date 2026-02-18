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

from typing import List, Union, Sequence
from alef.configs.experiment.simulator_configs.base_simulator_config import BaseSimulatorConfig
from alef.enums.simulator_enums import InitialDataGenerationMethod

class SingleTaskHartmann3Config(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool=False
    oracle_type: str="hartmann3_0"
    
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    
    observation_noise: float=0.01
    seed: int=1234
    name: str='single_task_hartmann3'

class SingleTaskHartmann3_0Config(SingleTaskHartmann3Config):
    oracle_type: str="hartmann3_0" # only affect initial data in this case
    name: str='single_task_hartmann3_0'

class SingleTaskHartmann3_1Config(SingleTaskHartmann3Config):
    oracle_type: str="hartmann3_1" # only affect initial data in this case
    name: str='single_task_hartmann3_1'

class SingleTaskHartmann3_2Config(SingleTaskHartmann3Config):
    oracle_type: str="hartmann3_2" # only affect initial data in this case
    name: str='single_task_hartmann3_2'

class SingleTaskHartmann3_3Config(SingleTaskHartmann3Config):
    oracle_type: str="hartmann3_3" # only affect initial data in this case
    name: str='single_task_hartmann3_3'

class SingleTaskHartmann3_4Config(SingleTaskHartmann3Config):
    oracle_type: str="hartmann3_4" # only affect initial data in this case
    name: str='single_task_hartmann3_4'
