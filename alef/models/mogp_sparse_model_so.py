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
from typing import Union, Sequence, List, Tuple, Optional
from tensorflow_probability import distributions as tfd
import gpflow
from gpflow.inducing_variables import InducingPoints
from gpflow.models import SGPR
from gpflow.utilities import print_summary, set_trainable

from alef.utils.utils import normal_entropy, k_means
from alef.utils.gp_paramater_cache import GPParameterCache
from alef.models.base_model import BaseModel
from alef.models.mogp_model_so import SOMOGPModel
from alef.models.mo_sgpr_so import FlattenedSeparateIndependentInducingVariables
from alef.kernels.multi_output_kernels.base_multioutput_flattened_kernel import BaseMultioutputFlattenedKernel
from alef.kernels.regularized_kernel_interface import RegularizedKernelInterface
from alef.enums.global_model_enums import InitializationType, PredictionQuantity, InitialParameters
import logging
from alef.utils.custom_logging import getLogger

logger = getLogger(__name__)

class SOMOSparseGPModel(SOMOGPModel):
    """
    Class that implements multi-output Gaussian process regression with Type-2 ML for kernel hyperparameter infernece. It forms mainly a wrapper around
    the MOGPR object

    Attributes:
        kernel: kernel that is used inside the Gaussian process - needs to be child of gpflow.kernels.MultioutputKernel
        model: holds the MOGPR instance
        optimize_hps: bool if kernel parameters are trained
        train_likelihood_variance: bool if likelihood variance is trained
        observation_noise: observation noise level - is either set fixed to that value or acts as initial starting value for optimization
        pertube_parameters_at_start: bool if parameters of the kernels should be pertubed before optimization
        set_prior_on_observation_noise: bool if prior should be applied to obvservation noise (Exponential prior with expected value self.observation_noise)
    """

    def __init__(
        self,
        kernel: BaseMultioutputFlattenedKernel,
        observation_noise: float,
        expected_observation_noise: float,
        num_inducing_points_per_dim: int,
        optimize_hps: bool,
        train_likelihood_variance: bool,
        sample_initial_parameters_at_start: bool,
        initial_parameter_strategy: InitialParameters,
        perturbation_for_multistart_opt: float = 0.5,
        perturbation_for_singlestart_opt: float = 0.1,
        perform_multi_start_optimization=False,
        n_starts_for_multistart_opt: int = 5,
        set_prior_on_observation_noise=False,
        prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_F,
        classification: bool = False,
        **kwargs
    ):
        self.kernel = kernel
        self.kernel_copy = gpflow.utilities.deepcopy(kernel)
        self.observation_noise = observation_noise
        self.expected_observation_noise = expected_observation_noise
        self.num_inducing_points_per_dim = num_inducing_points_per_dim
        self.model = None
        self.optimize_hps = optimize_hps
        self.train_likelihood_variance = train_likelihood_variance
        self.use_mean_function = False
        self.last_opt_time = 0
        self.last_multi_start_opt_time = 0
        self.sample_initial_parameters_at_start = sample_initial_parameters_at_start
        self.initial_parameter_strategy = initial_parameter_strategy
        self.perform_multi_start_optimization = perform_multi_start_optimization

        if self.perform_multi_start_optimization:
            self.perturbation_factor = perturbation_for_multistart_opt
        else:
            self.perturbation_factor = perturbation_for_singlestart_opt

        self.n_starts_for_multistart_opt = n_starts_for_multistart_opt
        self.set_prior_on_observation_noise = set_prior_on_observation_noise
        self.prediction_quantity = prediction_quantity
        self.classification = classification
        self.print_summaries = False
        if isinstance(kernel, RegularizedKernelInterface):
            self.add_kernel_hp_regularizer = True
        else:
            self.add_kernel_hp_regularizer = False

    def assign_likelihood_variance(self):
        new_value = np.power(self.observation_noise, 2.0)
        self.model.likelihood.variance.assign(new_value)

    def reset_model(self):
        """
        resets the model to the initial values - kernel parameters and observation noise are reset to initial values - gpflow model is deleted
        """
        if self.model is not None:
            self.kernel = gpflow.utilities.deepcopy(self.kernel_copy)
            del self.model

    def set_model_data(self, x_data: np.array, y_data: np.array, class_mask: np.array=None):
        assert hasattr(self, 'model')
        assert isinstance(self.model, SGPR)
        self.model.data = gpflow.models.util.data_input_to_tensor((x_data, y_data))

    def infer(self, x_data: np.array, y_data: np.array, class_mask: np.array=None):
        """
        Main entrance method for learning the model - training methods are called if hp should be done otherwise only gpflow model is build

        Arguments:
            x_data: Input array with shape (n,d+1) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
            class_mask [n] bool array, y_data[class_mask] is class label while y_data[~class_mask] is regression observation
        """
        yy = y_data
        model = SGPR
        D = self.kernel.input_dimension
        P = self.kernel.output_dimension
        iv_list = []
        task_id_list = []
        for p in range(P):
            xx = x_data[x_data[:, -1]==p, :D]
            if xx.shape[0] <= 0:
                continue
            x_repr = xx if xx.shape[0] <= self.num_inducing_points_per_dim else k_means(self.num_inducing_points_per_dim, xx)
            iv_list.append( InducingPoints(x_repr) )
            task_id_list.append(p)
        iv = FlattenedSeparateIndependentInducingVariables(iv_list, task_id_list)
        
        if self.use_mean_function:
            self.model = model(data=(x_data, yy), kernel=self.kernel, inducing_variable=iv, mean_function=self.mean_function, noise_variance=np.power(self.observation_noise, 2.0))
            set_trainable(self.model.mean_function.c, False)
        else:
            self.model = model(data=(x_data, yy), kernel=self.kernel, inducing_variable=iv, mean_function=None, noise_variance=np.power(self.observation_noise, 2.0))
        
        set_trainable(self.model.inducing_variable, False)
        set_trainable(self.model.likelihood.variance, self.train_likelihood_variance)
        if self.set_prior_on_observation_noise:
            self.model.likelihood.variance.prior = tfd.Exponential(1 / np.power(self.expected_observation_noise, 2.0))

        if self.optimize_hps:
            if self.perform_multi_start_optimization:
                self.multi_start_optimization(self.n_starts_for_multistart_opt)
            else:
                self.optimize()

    def print_model_summary(self):
        if logger.isEnabledFor(logging.DEBUG):
            print_summary(self.model)

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None, class_mask: Optional[np.array] = None) -> float:
        """
        Estimates the model evidence - always retrieves marg likelihood, also when HPs are provided with prior!!

        Arguments:
            x_data: (only necessary if infer was not yet called) Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: (only necessary if infer was not yet called) Label array with shape (n,1) where n is the number of training points
            class_mask: (n,) bool array, y_data[class_mask] is binary class (0 or 1) and y_data[~class_mask] is regression value

        Returns:
            marginal likelihood value of infered model
        """
        if self.model is None and x_data is not None and y_data is not None:
            self.infer(x_data, y_data, class_mask)
        return self.model.elbo().numpy()

