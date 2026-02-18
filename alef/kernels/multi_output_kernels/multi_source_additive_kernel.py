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
from typing import Tuple, Union, List, Sequence, Dict
from gpflow.utilities import print_summary, set_trainable

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float
from tensorflow_probability import distributions as tfd

from alef.utils.gpflow_addon.kronecker_delta_kernel import Delta, Delta_t
from alef.kernels.multi_output_kernels.base_transfer_kernel import BaseTransferKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel

"""
The kernel in

Alonso Marco, Felix Berkenkamp, Philipp Hennig, Angela P. Schoellig, Andreas Krause, Stefan Schaal and Sebastian Trimpe,
ICRA 2017, Virtual vs. Real: Trading Off Simulations and Physical Experiments in Reinforcement Learning with Bayesian Optimization


The kernel is similar to this one:
Matthias Poloczek, Jialei Wang and Peter Frazier, NeurIPS 2017, Multi-Information Source Optimization

"""
class FilterKernel(gpflow.kernels.Coregion):
    
    def __init__(
        self,
        output_dim: int,
        start_dim: int,
        *,
        active_dims = None,
        name = None,
    ):
        # this kernel should be [1(i>=start_dim, j>=start_dim)]_ij
        super(gpflow.kernels.Coregion, self).__init__(active_dims=active_dims, name=name)

        self.output_dim = output_dim
        self.start_dim = start_dim
        B = np.zeros([output_dim, output_dim])
        B[start_dim-1:, start_dim-1:] = 1
        self.B = f64(B)

    def output_covariance(self):
        return self.B

    def output_variance(self) -> tf.Tensor:
        return tf.linalg.diag_part(self.B)

class MIAdditiveKernel(BaseTransferKernel):
    """
    X: [N, D+1], the last column is binary
    D: input_dimension
    """
    def __init__(
        self,
        base_variance: float,
        base_lengthscale: Union[float, Sequence[float]],
        input_dimension: int,
        output_dimension: int,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        latent_kernel: LatentKernel,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        fix_kernel: bool=False,
        assign_values: bool=False,
        parameter_values: Dict={},
        **kwargs
    ):
        P = output_dimension
        super().__init__(input_dimension, P, active_on_single_dimension, active_dimension, name, **kwargs,)

        var = f64(base_variance)
        if hasattr(base_lengthscale, '__len__'):
            if len(base_lengthscale) == 1:
                lg = f64(np.repeat(base_lengthscale[0], self.num_active_dimensions))
            else:
                assert len(base_lengthscale) == self.num_active_dimensions
                lg = f64(base_lengthscale)
        else:
            lg = f64(np.repeat(base_lengthscale, self.num_active_dimensions))

        k_object = self.pick_kernel_object(latent_kernel)
        
        k_source = k_object(
            variance=var,
            lengthscales=lg,
            active_dims=tf.range(self.num_active_dimensions),
            name='kernel_0'
        )
        
        for p in range(1, P):
            k_p = k_object(
                variance=var,
                lengthscales=lg,
                active_dims=tf.range(self.num_active_dimensions),
                name=f'kernel_residual_{p}'
            )

            dt = FilterKernel(P, p+1, active_dims=[self.num_active_dimensions])
            k_source = k_source + dt * k_p
        self.kernel = k_source
        if add_prior:
            self.assign_prior(lengthscale_prior_parameters, variance_prior_parameters)

        if assign_values:
            self.assign_parameters(parameter_values)
        if fix_kernel:
            set_trainable(self.kernel, False)

    @property
    def num_latent_gps(self):
        return self.output_dimension - 1
    
    @property
    def latent_kernels(self):
        out_list = [self.kernel.kernels[0]]
        for p in range(1, self.output_dimension):
            out_list.append(self.kernel.kernels[p].kernels[1])
        return out_list
    
    def assign_prior(self, lengthscale_prior_parameters, variance_prior_parameters):
        a_lengthscale, b_lengthscale = lengthscale_prior_parameters
        loc, std = variance_prior_parameters

        for k in self.latent_kernels:
            k.variance.prior = tfd.TruncatedNormal(
                loc = f64(loc), scale = f64(std),
                low = f64(1e-6), high = f64(30),
                name='kernel_var_prior_TruncatedNormal')
            k.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.num_active_dimensions)),
                f64(np.repeat(b_lengthscale, self.num_active_dimensions)),
                name='kernel_len_prior_Gamma')

    def get_source_parameters_trainable(self):
        return self.kernel.kernels[0].lengthscales.trainable

    def set_source_parameters_trainable(self, source_trainable: bool):
        for p in range(self.output_dimension - 1):
            set_trainable(self.kernel.kernels[p], source_trainable)

    def get_target_parameters_trainable(self):
        return self.kernel.kernels[1].kernels[1].lengthscales.trainable

    def set_target_parameters_trainable(self, target_trainable: bool):
        P = self.output_dimension
        set_trainable(self.kernel.kernels[P-1].kernels[1], target_trainable)

if __name__ == "__main__":
    D = 2
    P = 3
    k = MIAdditiveKernel(1.0, 1.0, D, P, False, None, None, LatentKernel.MATERN52, False, None, 'name_hello')
    X = np.hstack((
        np.random.uniform(size=(P, D)),
        np.arange(P).reshape([P,1])
    ))

    print(k(X))
    print_summary(k)
    print(k.latent_kernels)
    k.set_source_parameters_trainable(False)
    print_summary(k)
    k.set_source_parameters_trainable(True)
    print_summary(k)
    k.set_target_parameters_trainable(False)
    print_summary(k)
    k.set_source_parameters_trainable(False)
    k.set_target_parameters_trainable(True)
    print_summary(k)
