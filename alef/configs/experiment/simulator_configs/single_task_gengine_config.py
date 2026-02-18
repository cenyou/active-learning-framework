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


class SingleTaskGEngineConfig(BaseSimulatorConfig):
    n_pool: int
    data_path: str
    additional_safety: bool = True
    data_set: str = 'gengine1'
    initial_data_source: InitialDataGenerationMethod = InitialDataGenerationMethod.POOL_SELECT
    target_data_idx: List[int] = [100623, 258800, 182283, 258861, 258860, 176828, 176829, 258866,
       311254, 258886, 181599, 183062, 258833, 258841, 181602, 186826,
        89822, 258858, 293823, 181601, 258798,  99476, 311247, 258825,
       181610, 258862, 176840, 186828, 258884, 182282,  99495, 258839,
       100936, 302267, 182284, 302263,  94705, 100621,  94701, 258831,
       258792, 186819,  94700, 181591,  99486, 258796, 186827,  89820,
       302264, 258863, 311263, 176831, 182752, 100624, 181605,  94703,
       182744, 311261,  99489,  99488, 258848, 258790, 182750,  99485,
       100939, 302260, 181615, 176835, 258797, 302265, 258801, 258840,
       100161, 258850, 258874, 258803, 100162, 183069, 311252, 186820,
       258870,  94704, 258830, 258854, 258838, 258865, 186821, 311257,
       100163,  94696, 258783,  99487, 311253, 176839, 100625, 182285,
       302266, 258885, 171591, 181593]
    seed: int=1234
    name: str='single_task_gengine'


class SingleTaskGEngineTestConfig(SingleTaskGEngineConfig):
    name: str='single_task_gengine_test'
