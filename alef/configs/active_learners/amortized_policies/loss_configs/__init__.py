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

from . import (
    base_loss_configs,
    information_loss_configs,
    safety_loss_configs,
)
from .base_loss_configs import *
from .information_loss_configs import _ScoreDADLossConfig
from .information_loss_configs import *
from .safety_loss_configs import *

_myopic_loss_list = information_loss_configs._myopic_loss_list + safety_loss_configs._myopic_loss_list
_nonmyopic_loss_list = information_loss_configs._nonmyopic_loss_list + safety_loss_configs._nonmyopic_loss_list

__all__ = base_loss_configs.__all__ + information_loss_configs.__all__ + safety_loss_configs.__all__
