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

import numpy as np

class PoolContextualHelper:
    def _return_variable_idx(self):
        idx_dim = np.ones(self.x_data.shape[1], dtype=bool)
        
        dim = self.pool.get_dimension()
        var_dim = self.pool.get_variable_dimension()
        if var_dim < dim:
            _, idx = self.pool.get_context_status(return_idx=True)
            idx_dim[idx] = False
                
        return idx_dim

    def _get_variable_input(self, x: np.ndarray):
        idx_dim = self._return_variable_idx()
        return x[:, idx_dim]

class OracleContextualHelper:
    def _return_variable_idx(self):
        idx_dim = np.ones(self.x_data.shape[1], dtype=bool)
        
        dim = self.oracle.get_dimension()
        var_dim = self.oracle.get_variable_dimension()
        if var_dim < dim:
            _, idx = self.oracle.get_context_status(return_idx=True)
            idx_dim[idx] = False
                
        return idx_dim

    def _get_variable_input(self, x: np.ndarray):
        idx_dim = self._return_variable_idx()
        return x[:, idx_dim]
