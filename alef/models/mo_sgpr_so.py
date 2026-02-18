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

from collections import namedtuple
from typing import NamedTuple, Optional, Tuple, Union, Sequence
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow

from gpflow import posteriors
from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.inducing_variables import InducingVariables, InducingPoints
from gpflow.inducing_variables.multioutput import MultioutputInducingVariables
from gpflow.kernels import Kernel
from gpflow.kullback_leiblers import gauss_kl
from gpflow.likelihoods import Bernoulli, Gaussian, SwitchedLikelihood
from gpflow.logdensities import multivariate_normal
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)

f64 = gpflow.utilities.to_default_float

"""
Some pieces of the following class SOMOSGPR is adapted from GPflow
(https://github.com/GPflow/GPflow/blob/develop/gpflow/models/sgpr.py
Copyright 2016-2020 The GPflow Contributors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""

class FlattenedSeparateIndependentInducingVariables(InducingPoints):
    # gpflow methods handle subclasses of InducingPoints
    # don't change the parent to InducingVariables
    def __init__(self, inducing_variable_list, task_index_list: Sequence[int], name: Optional[str] = None):
        assert len(inducing_variable_list)==len(task_index_list)
        InducingVariables.__init__(self, name=name)
        self.inducing_variable_list = inducing_variable_list
        self.task_index_list = [p*tf.ones([iv.num_inducing, 1], dtype=iv.Z.dtype) for p, iv in zip(task_index_list, inducing_variable_list)]

    @property
    def Z(self):
        Zs = [iv.Z for iv in self.inducing_variable_list]
        return tf.concat([
            tf.concat(Zs, axis=0),
            tf.concat(self.task_index_list, axis=0)
        ], axis=-1)

    @property
    def num_inducing(self):
        return sum([iv.num_inducing for iv in self.inducing_variable_list])






