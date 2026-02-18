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

from .pool_with_safety_from_oracle import PoolWithSafetyFromOracle
from alef.oracles import OracleNormalizer, SimionescuMain, SimionescuConstraint

class SimionescuPool(PoolWithSafetyFromOracle):
    def __init__(self, observation_noise: float, seed: int=123, set_seed: bool=False):
        oracle = OracleNormalizer(SimionescuMain(observation_noise))
        oracle.set_normalization_by_sampling()

        safety_oracle = OracleNormalizer(SimionescuConstraint(observation_noise))
        safety_oracle.set_normalization_by_sampling()
        mu, scale = safety_oracle.get_normalization()
        safety_oracle.set_normalization_manually(0.0, scale)

        super().__init__(oracle, safety_oracle, seed, set_seed)