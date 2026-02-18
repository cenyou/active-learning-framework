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

from typing import Tuple, Union, Optional
from pydantic_settings import BaseSettings


from alef.configs.base_parameters import INPUT_DOMAIN

__all__ = [
    'BaseAmortizedPolicyConfig',
]


class BaseAmortizedPolicyConfig(BaseSettings):
    input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN
    resume_policy_path: Optional[str] = None
    forward_with_budget: bool = False # if True, we wish the NN forward like (T, x1, y1, ...)
    name: str = 'basic_al_policy_config'
