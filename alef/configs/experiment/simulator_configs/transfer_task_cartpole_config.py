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

class TransferTaskCartPoleConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool=True
    dim_s: int=3
    oracle_type: str="cartpole"
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float=0.01
    seed: int=1234
    name: str='transfer_task_cartpole'

class TransferTaskCartPole0Config(TransferTaskCartPoleConfig):
    oracle_type: str="cartpole0"
    name: str='transfer_task_cartpole0'

class TransferTaskCartPole1Config(TransferTaskCartPoleConfig):
    oracle_type: str="cartpole1"
    name: str='transfer_task_cartpole1'

class TransferTaskCartPole2Config(TransferTaskCartPoleConfig):
    oracle_type: str="cartpole2"
    name: str='transfer_task_cartpole2'

class TransferTaskCartPole3Config(TransferTaskCartPoleConfig):
    oracle_type: str="cartpole3"
    name: str='transfer_task_cartpole3'

class TransferTaskCartPole4Config(TransferTaskCartPoleConfig):
    oracle_type: str="cartpole4"
    name: str='transfer_task_cartpole4'
