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

from typing import Optional, Tuple, Union, Sequence
import numpy as np
import tensorflow as tf
import gpflow

from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.kernels import Kernel
from gpflow.kullback_leiblers import gauss_kl
from gpflow.likelihoods import Bernoulli, Gaussian, SwitchedLikelihood
from gpflow.likelihoods.utils import inv_probit
from gpflow.logdensities import bernoulli, gaussian
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.utilities import triangular, triangular_size


gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float
"""
Some pieces of the following class SOMOGPR is adapted from GPflow
(https://github.com/GPflow/GPflow/blob/develop/gpflow/models/vgp.py
https://github.com/GPflow/GPflow/blob/develop/gpflow/likelihoods/base.py
Copyright 2016-2020 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""


class SingleTaskGaussianBernoulli(SwitchedLikelihood):
    def __init__(self, *args, **kwargs):
        base_lik = Gaussian(*args, **kwargs)
        super().__init__([base_lik, Bernoulli()])

    @property
    def variance(self):
        return self.likelihoods[0].variance


class GPRC_Binary(gpflow.models.VGP):
    """
    This is a GP regression where the training data
    may contain both regression observations and binary classification labels.
    
    data (X, Y) should be prepared as follows:
    
    X: [N, D]
    
    Y: [N, 2], where the last dim is class index (0: regression observation, 1: class label)
    
    e.g. Y[:, -1] is in [0, 1] and Y[Y[:,-1]==1] are 0 or 1
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        likelihood = SingleTaskGaussianBernoulli(f64(noise_variance))
        super().__init__(data=data, kernel=kernel, likelihood=likelihood, mean_function=mean_function)

    def predict_y(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        f_mean, f_var = self.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        return self.likelihood.likelihoods[0].predict_mean_and_var(f_mean, f_var)



class _GPRC_Binary_Laplace(gpflow.models.GPR):
    """
    This is a GP regression where the training data
    may contain both regression observations and binary classification labels.
    
    data (X, Y) should be prepared as follows:
    
    X: [N, D]
    
    Y: [N, 2], where the last dim is class index (0: regression observation, 1: class label)
    
    e.g. Y[:, -1] is in [0, 1] and Y[Y[:,-1]==1] are 0 or 1
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        X_data, Y_data = data
        self.class_mask = tf.cast(Y_data[:, -1], bool)
        super().__init__((X_data, Y_data[:, 0, None]), kernel=kernel, mean_function=mean_function, noise_variance=f64(noise_variance))
        num_data = Y_data.shape[-2]
        self.mode = Parameter(
            tf.zeros((num_data, self.num_latent_gps)),
            shape=(num_data, self.num_latent_gps),
        )

    @property
    def W(self):
        X, Y = self.data
        labels = tf.where(tf.expand_dims(self.class_mask, -1), Y, tf.zeros_like(Y)) # [N, 1]
        mean = self.mean_function(X)
        F = self.mode + mean
        
        class_prob = tf.exp( bernoulli( labels, inv_probit(F) ) )
        gaussian_f = tf.exp( gaussian(F, f64(0.0), f64(1.0)) )
        W_diag = - tf.square(gaussian_f / class_prob) - tf.multiply(tf.multiply(Y, F), gaussian_f) / class_prob
        W = tf.linalg.diag( tf.reshape(W_diag, [-1]) )
        return W

    def maximum_log_likelihood_objective(self):
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self):
        """
        Laplace Approximation
        sec 3.2 of Nickisch & Rasmussen 2008
        """
        X, Y = self.data
        labels = tf.where(tf.expand_dims(self.class_mask, -1), Y, tf.zeros_like(Y)) # [N, 1]
        
        mean = self.mean_function(X)
        F = self.mode + mean
        
        class_log_prob = bernoulli( labels, inv_probit(F) )
        regres_log_prob = self.likelihood.log_prob( F, Y )
        log_prob = tf.reduce_sum( tf.where(self.class_mask, class_log_prob, regres_log_prob) )
        
        K = self.kernel(X) + tf.eye(Y.shape[0], dtype=default_float()) * default_jitter()
        L = tf.linalg.cholesky(K)
        Lm = tf.linalg.triangular_solve(L, self.mode, lower=True, adjoint=False)
        log_prob += -1/2 * tf.reduce_sum(tf.square(Lm))
        
        regularizer = tf.eye(Y.shape[0], dtype=default_float()) + tf.linalg.matmul(K, self.W)
        regularizer = 1/2 * tf.linalg.det(regularizer)

        return log_prob - regularizer

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        X, Y = self.data

        kmm = self.kernel(X) + tf.eye(Y.shape[0], dtype=default_float()) * default_jitter()
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X, Xnew)
        W = self.W
        B = tf.eye(Y.shape[0], dtype=default_float()) + tf.linalg.matmul(W, kmm)
        # f_mean_zero = kmn.T @ K^-1 @ self.mode
        L = tf.linalg.cholesky(kmm)
        Lm = tf.linalg.triangular_solve(L, self.mode, lower=True, adjoint=False)
        Lk = tf.linalg.triangular_solve(L, kmn, lower=True, adjoint=False)
        f_mean_zero = tf.linalg.matmul(Lk, Lm, transpose_a=True)
        # f_var = knn - kmn.T W_sqrt B^-1 @_sqrt kmn
        L = tf.linalg.cholesky(B)
        Lk = tf.linalg.triangular_solve(L, tf.linalg.matmul(tf.math.sqrt(W), kmn), lower=True, adjoint=False)

        if full_cov:
            cov_reg = tf.matmul(Lk, Lk, transpose_a=True)
            f_var = knn - cov_reg
        else:
            cov_reg = tf.reduce_sum( tf.square(Lk), axis=-2 )
            f_var = knn - cov_reg

        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var
