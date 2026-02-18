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

class SingleTaskMOGP4DzBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool=True
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float=0.01
    seed: int=1234
    name: str='single_task_mogp'

class SingleTaskMOGP4Dz0Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz0"
    name: str='single_task_mogp4dz0'

class SingleTaskMOGP4Dz1Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz1"
    name: str='single_task_mogp4dz1'

class SingleTaskMOGP4Dz2Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz2"
    name: str='single_task_mogp4dz2'

class SingleTaskMOGP4Dz3Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz3"
    name: str='single_task_mogp4dz3'

class SingleTaskMOGP4Dz4Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz4"
    name: str='single_task_mogp4dz4'

class SingleTaskMOGP4Dz5Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz5"
    name: str='single_task_mogp4dz5'

class SingleTaskMOGP4Dz6Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz6"
    name: str='single_task_mogp4dz6'

class SingleTaskMOGP4Dz7Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz7"
    name: str='single_task_mogp4dz7'

class SingleTaskMOGP4Dz8Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz8"
    name: str='single_task_mogp4dz8'

class SingleTaskMOGP4Dz9Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz9"
    name: str='single_task_mogp4dz9'

class SingleTaskMOGP4Dz10Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz10"
    name: str='single_task_mogp4dz10'

class SingleTaskMOGP4Dz11Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz11"
    name: str='single_task_mogp4dz11'

class SingleTaskMOGP4Dz12Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz12"
    name: str='single_task_mogp4dz12'

class SingleTaskMOGP4Dz13Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz13"
    name: str='single_task_mogp4dz13'

class SingleTaskMOGP4Dz14Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz14"
    name: str='single_task_mogp4dz14'

class SingleTaskMOGP4Dz15Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz15"
    name: str='single_task_mogp4dz15'

class SingleTaskMOGP4Dz16Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz16"
    name: str='single_task_mogp4dz16'

class SingleTaskMOGP4Dz17Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz17"
    name: str='single_task_mogp4dz17'

class SingleTaskMOGP4Dz18Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz18"
    name: str='single_task_mogp4dz18'

class SingleTaskMOGP4Dz19Config(SingleTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz19"
    name: str='single_task_mogp4dz19'
