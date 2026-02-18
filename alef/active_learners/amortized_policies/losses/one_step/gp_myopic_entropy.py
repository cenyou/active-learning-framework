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

from alef.active_learners.amortized_policies.losses.multiple_steps.gp_entropy import BaseEntropyLoss, _GPEntropyComputer1, _GPEntropyComputer2

class GPMyopicEntropy1Loss(BaseEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPEntropyComputer1(myopic=True)

class GPMyopicEntropy2Loss(BaseEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_computer = _GPEntropyComputer2(myopic=True)


