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

class TransferTaskMOGP4DzBaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool=True
    oracle_type: str
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float=0.01
    seed: int=1234
    name: str='transfer_task_mogp'

class TransferTaskMOGP4Dz0Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz0"
    name: str='transfer_task_mogp4dz0'

class TransferTaskMOGP4Dz1Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz1"
    name: str='transfer_task_mogp4dz1'

class TransferTaskMOGP4Dz2Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz2"
    name: str='transfer_task_mogp4dz2'

class TransferTaskMOGP4Dz3Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz3"
    name: str='transfer_task_mogp4dz3'

class TransferTaskMOGP4Dz4Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz4"
    name: str='transfer_task_mogp4dz4'

class TransferTaskMOGP4Dz5Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz5"
    name: str='transfer_task_mogp4dz5'

class TransferTaskMOGP4Dz6Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz6"
    name: str='transfer_task_mogp4dz6'

class TransferTaskMOGP4Dz7Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz7"
    name: str='transfer_task_mogp4dz7'

class TransferTaskMOGP4Dz8Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz8"
    name: str='transfer_task_mogp4dz8'

class TransferTaskMOGP4Dz9Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz9"
    name: str='transfer_task_mogp4dz9'

class TransferTaskMOGP4Dz10Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz10"
    name: str='transfer_task_mogp4dz10'

class TransferTaskMOGP4Dz11Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz11"
    name: str='transfer_task_mogp4dz11'

class TransferTaskMOGP4Dz12Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz12"
    name: str='transfer_task_mogp4dz12'

class TransferTaskMOGP4Dz13Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz13"
    name: str='transfer_task_mogp4dz13'

class TransferTaskMOGP4Dz14Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz14"
    name: str='transfer_task_mogp4dz14'

class TransferTaskMOGP4Dz15Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz15"
    name: str='transfer_task_mogp4dz15'

class TransferTaskMOGP4Dz16Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz16"
    name: str='transfer_task_mogp4dz16'

class TransferTaskMOGP4Dz17Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz17"
    name: str='transfer_task_mogp4dz17'

class TransferTaskMOGP4Dz18Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz18"
    name: str='transfer_task_mogp4dz18'

class TransferTaskMOGP4Dz19Config(TransferTaskMOGP4DzBaseConfig):
    oracle_type: str="mogp4dz19"
    name: str='transfer_task_mogp4dz19'
