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

class SingleTaskMOGP5DzBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool=True
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float=0.01
    seed: int=1234
    name: str='single_task_mogp'

class SingleTaskMOGP5Dz0Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz0"
    name: str='single_task_mogp5dz0'

class SingleTaskMOGP5Dz1Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz1"
    name: str='single_task_mogp5dz1'

class SingleTaskMOGP5Dz2Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz2"
    name: str='single_task_mogp5dz2'

class SingleTaskMOGP5Dz3Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz3"
    name: str='single_task_mogp5dz3'

class SingleTaskMOGP5Dz4Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz4"
    name: str='single_task_mogp5dz4'

class SingleTaskMOGP5Dz5Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz5"
    name: str='single_task_mogp5dz5'

class SingleTaskMOGP5Dz6Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz6"
    name: str='single_task_mogp5dz6'

class SingleTaskMOGP5Dz7Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz7"
    name: str='single_task_mogp5dz7'

class SingleTaskMOGP5Dz8Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz8"
    name: str='single_task_mogp5dz8'

class SingleTaskMOGP5Dz9Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz9"
    name: str='single_task_mogp5dz9'

class SingleTaskMOGP5Dz10Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz10"
    name: str='single_task_mogp5dz10'

class SingleTaskMOGP5Dz11Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz11"
    name: str='single_task_mogp5dz11'

class SingleTaskMOGP5Dz12Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz12"
    name: str='single_task_mogp5dz12'

class SingleTaskMOGP5Dz13Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz13"
    name: str='single_task_mogp5dz13'

class SingleTaskMOGP5Dz14Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz14"
    name: str='single_task_mogp5dz14'

class SingleTaskMOGP5Dz15Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz15"
    name: str='single_task_mogp5dz15'

class SingleTaskMOGP5Dz16Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz16"
    name: str='single_task_mogp5dz16'

class SingleTaskMOGP5Dz17Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz17"
    name: str='single_task_mogp5dz17'

class SingleTaskMOGP5Dz18Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz18"
    name: str='single_task_mogp5dz18'

class SingleTaskMOGP5Dz19Config(SingleTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz19"
    name: str='single_task_mogp5dz19'
