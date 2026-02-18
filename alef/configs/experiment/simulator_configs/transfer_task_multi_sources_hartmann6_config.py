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

class TransferTaskMultiSourcesHartmann6BaseConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool=False
    dim_s: int=5
    oracle_type: str="multi_sources_hartmann6"
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.FILE
    observation_noise: float=0.01
    seed: int=1234
    name: str='transfer_task_multi_sources_hartmann6'

class TransferTaskMultiSourcesHartmann6_0Config(TransferTaskMultiSourcesHartmann6BaseConfig):
    oracle_type: str="multi_sources_hartmann6_0"
    name: str='transfer_task_multi_sources_hartmann6_0'

class TransferTaskMultiSourcesHartmann6_1Config(TransferTaskMultiSourcesHartmann6BaseConfig):
    oracle_type: str="multi_sources_hartmann6_1"
    name: str='transfer_task_multi_sources_hartmann6_1'

class TransferTaskMultiSourcesHartmann6_2Config(TransferTaskMultiSourcesHartmann6BaseConfig):
    oracle_type: str="multi_sources_hartmann6_2"
    name: str='transfer_task_multi_sources_hartmann6_2'

class TransferTaskMultiSourcesHartmann6_3Config(TransferTaskMultiSourcesHartmann6BaseConfig):
    oracle_type: str="multi_sources_hartmann6_3"
    name: str='transfer_task_multi_sources_hartmann6_3'

class TransferTaskMultiSourcesHartmann6_4Config(TransferTaskMultiSourcesHartmann6BaseConfig):
    oracle_type: str="multi_sources_hartmann6_4"
    name: str='transfer_task_multi_sources_hartmann6_4'
