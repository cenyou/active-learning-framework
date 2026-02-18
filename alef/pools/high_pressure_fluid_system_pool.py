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

from alef.pools.pool_with_safety_from_constrained_oracle import PoolWithSafetyFromConstrainedOracle
from alef.oracles import HighPressureFluidSystem

class HighPressureFluidSystemPool(PoolWithSafetyFromConstrainedOracle):
    def __init__(self, observation_noise: float, seed: int=123, set_seed: bool=False, trajectory_sampler: bool=True):
        super().__init__(HighPressureFluidSystem(observation_noise, normalized=True, trajectory_sampler=trajectory_sampler), seed=seed, set_seed=set_seed)

    def discretize_random(self,n : int):
        r"""
        set x randomly from the space defined in the oracle (get discretized input space from the oracle)
        """
        if self.oracle.trajectory_sampler:
            X = self.oracle.get_trajectory(num_trajectories=n)
            self.set_data(X)
        else:
            super().discretize_random(n)

if __name__ == "__main__":
    pool = HighPressureFluidSystemPool(observation_noise=0.1, seed=123, set_seed=True)
    pool.discretize_random(10)
    print(pool.possible_queries())  # Print the discretized input space
    pool.get_random_data(5)
    pool.get_random_data_in_box(5, a=0, box_width=0.5)
    pool.get_random_constrained_data_in_box(5, a=0, box_width=0.5, constraint_lower=0)