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

from typing import Union, Sequence, List
import numpy as np
from alef.pools.pool_with_safety_from_data import PoolWithSafetyFromData
from alef.data_sets.base_data_set import StandardDataSet

class PoolWithSafetyFromDataSet(PoolWithSafetyFromData):
    def __init__(
        self,
        dataset: StandardDataSet,
        input_idx: List[Union[int, bool]],
        output_idx: List[Union[int, bool]]=[0],
        safety_idx: List[Union[int, bool]]=[0],
        data_is_noisy: bool=True,
        observation_noise: float = None,
        safety_noise: Sequence[float] = None,
        seed:int=123,
        set_seed:bool=False
    ):
        x, y = dataset.get_complete_dataset()
        super().__init__(
            x[..., input_idx],
            y[..., output_idx],
            y[..., safety_idx],
            data_is_noisy=data_is_noisy,
            observation_noise=observation_noise,
            safety_noise=safety_noise,
            seed=seed,
            set_seed=set_seed
        )
        
    
if __name__ == '__main__':
    import pandas as pd
    import os
    from alef.data_sets.pytest_set import PytestMOSet
    
    dataset = PytestMOSet()
    dataset.load_data_set()

    pool = PoolWithSafetyFromDataSet(dataset, [0,1,2], [0], [1])
    
    xx, yy, zz = pool.get_full_data()
    print(xx.shape)
    print(yy.shape)
    print(zz.shape)