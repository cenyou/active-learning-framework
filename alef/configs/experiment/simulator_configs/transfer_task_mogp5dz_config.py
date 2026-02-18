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

class TransferTaskMOGP5DzBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool=True
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float=0.01
    seed: int=1234
    name: str='transfer_task_mogp'

class TransferTaskMOGP5Dz0Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz0"
    name: str='transfer_task_mogp5dz0'

class TransferTaskMOGP5Dz1Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz1"
    name: str='transfer_task_mogp5dz1'

class TransferTaskMOGP5Dz2Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz2"
    name: str='transfer_task_mogp5dz2'

class TransferTaskMOGP5Dz3Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz3"
    name: str='transfer_task_mogp5dz3'

class TransferTaskMOGP5Dz4Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz4"
    name: str='transfer_task_mogp5dz4'

class TransferTaskMOGP5Dz5Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz5"
    name: str='transfer_task_mogp5dz5'

class TransferTaskMOGP5Dz6Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz6"
    name: str='transfer_task_mogp5dz6'

class TransferTaskMOGP5Dz7Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz7"
    name: str='transfer_task_mogp5dz7'

class TransferTaskMOGP5Dz8Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz8"
    name: str='transfer_task_mogp5dz8'

class TransferTaskMOGP5Dz9Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz9"
    name: str='transfer_task_mogp5dz9'

class TransferTaskMOGP5Dz10Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz10"
    name: str='transfer_task_mogp5dz10'

class TransferTaskMOGP5Dz11Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz11"
    name: str='transfer_task_mogp5dz11'

class TransferTaskMOGP5Dz12Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz12"
    name: str='transfer_task_mogp5dz12'

class TransferTaskMOGP5Dz13Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz13"
    name: str='transfer_task_mogp5dz13'

class TransferTaskMOGP5Dz14Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz14"
    name: str='transfer_task_mogp5dz14'

class TransferTaskMOGP5Dz15Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz15"
    name: str='transfer_task_mogp5dz15'

class TransferTaskMOGP5Dz16Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz16"
    name: str='transfer_task_mogp5dz16'

class TransferTaskMOGP5Dz17Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz17"
    name: str='transfer_task_mogp5dz17'

class TransferTaskMOGP5Dz18Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz18"
    name: str='transfer_task_mogp5dz18'

class TransferTaskMOGP5Dz19Config(TransferTaskMOGP5DzBaseConfig):
    oracle_type: str="mogp5dz19"
    name: str='transfer_task_mogp5dz19'
