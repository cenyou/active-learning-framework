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

from alef.configs.models.base_model_config import BaseModelConfig
from alef.configs.models.tabpfn_model_config import BasicTabPFNModelConfig
from alef.models.tabpfn_model import TabPFNModel

class ModelFactory:
    @staticmethod
    def build(model_config: BaseModelConfig):
        if isinstance(model_config, BasicTabPFNModelConfig):
            model = TabPFNModel(**model_config.dict())
            return model
        else:
            raise NotImplementedError(f"Invalid config: {model_config.__class__.__name__}")


