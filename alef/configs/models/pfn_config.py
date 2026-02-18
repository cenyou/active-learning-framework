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

from typing import List, Dict, Union, Optional
from pydantic_settings import BaseSettings

class PFNTorchConfig(BaseSettings):
    input_dimension: int
    output_dimension: int=1
    d_model: int=128
    dim_feedforward: int=256
    nhead: int=4
    dropout: float=0.0
    num_layers: int=6
    head_num_buckets: int=100
