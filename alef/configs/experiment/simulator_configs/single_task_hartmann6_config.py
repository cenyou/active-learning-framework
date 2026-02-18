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

class SingleTaskHartmann6Config(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool=False
    oracle_type: str="hartmann6_0"
    
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    
    observation_noise: float=0.01
    seed: int=1234
    name: str='single_task_hartmann6'

class SingleTaskHartmann6_0Config(SingleTaskHartmann6Config):
    oracle_type: str="hartmann6_0" # only affect initial data in this case
    name: str='single_task_hartmann6_0'

class SingleTaskHartmann6_1Config(SingleTaskHartmann6Config):
    oracle_type: str="hartmann6_1" # only affect initial data in this case
    name: str='single_task_hartmann6_1'

class SingleTaskHartmann6_2Config(SingleTaskHartmann6Config):
    oracle_type: str="hartmann6_2" # only affect initial data in this case
    name: str='single_task_hartmann6_2'

class SingleTaskHartmann6_3Config(SingleTaskHartmann6Config):
    oracle_type: str="hartmann6_3" # only affect initial data in this case
    name: str='single_task_hartmann6_3'

class SingleTaskHartmann6_4Config(SingleTaskHartmann6Config):
    oracle_type: str="hartmann6_4" # only affect initial data in this case
    name: str='single_task_hartmann6_4'
