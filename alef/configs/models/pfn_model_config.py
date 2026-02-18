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

from typing import Optional
from alef.configs.models.base_model_config import BaseModelConfig
from alef.enums.global_model_enums import PredictionQuantity
from alef.configs.models.pfn_config import PFNTorchConfig
# config for PFN model of our API

class BasicPFNModelConfig(BaseModelConfig):
    pfn_backend_config: PFNTorchConfig
    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    name = "PFNModel"

class PFNModelGPUConfig(BasicPFNModelConfig):
    pfn_backend_config: PFNTorchConfig
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
    name = "PFNModel_CUDA"

