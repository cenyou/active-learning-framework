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
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, Union, List, Sequence
from gpflow.utilities import print_summary, set_trainable

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float
from tensorflow_probability import distributions as tfd

from alef.utils.gpflow_addon.kronecker_delta_kernel import Delta, Delta_t
from alef.kernels.multi_output_kernels.base_transfer_kernel import BaseTransferKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel

class FPACOHKernel(BaseTransferKernel):
    """
    X: [N, D+1], the last column is binary
    D: input_dimension
    """
    def __init__(
        self,
        base_variance: float,
        base_lengthscale: float,
        input_dimension: int,
        output_dimension: int,
        latent_kernel: LatentKernel,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs
    ):
        P = output_dimension
        super().__init__(
            input_dimension,
            output_dimension,
            active_on_single_dimension,
            active_dimension,
            name,
            **kwargs,
        )
        self.base_variance = base_variance
        self.base_lengthscale = base_lengthscale
        self.kernel_type = latent_kernel
    
    @property
    def num_latent_gps(self):
        return 1
    
    @property
    def latent_kernels(self):
        return [self.kernel.kernels[0], self.kernel.kernels[1].kernels[1]]
    
    def get_source_parameters_trainable(self):
        return self.kernel.kernels[0].lengthscales.trainable

    def set_source_parameters_trainable(self, source_trainable: bool):
        for p in range(self.output_dimension - 1):
            set_trainable(self.kernel.kernels[p], source_trainable)

    def get_target_parameters_trainable(self):
        return self.kernel.kernels[1].kernels[1].lengthscales.trainable

    def set_target_parameters_trainable(self, target_trainable: bool):
        P = self.output_dimension
        set_trainable(self.kernel.kernels[1].kernels[1], target_trainable)

if __name__ == "__main__":
    pass