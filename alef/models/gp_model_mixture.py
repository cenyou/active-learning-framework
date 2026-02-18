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

import logging
import gpflow
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from typing import List, Tuple, Optional
import numpy as np
from gpflow.utilities import print_summary, set_trainable
from alef.kernels.input_initialized_kernel_interface import InputInitializedKernelInterface
from alef.kernels.regularized_kernel_interface import RegularizedKernelInterface
from alef.models.base_model import BaseModel
from alef.models.bayesian_ensemble_interface import BayesianEnsembleInterface
from alef.utils.gaussian_mixture_density import EntropyApproximation, GaussianMixtureDensity
from alef.utils.gaussian_mixture_density_nd import GaussianMixtureDensityNd
from alef.models.batch_model_interface import BatchModelInterace
from alef.enums.global_model_enums import PredictionQuantity

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
f64 = gpflow.utilities.to_default_float

from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)


class GPModelMixture(BaseModel, BatchModelInterace, BayesianEnsembleInterface):
    """
    Class for Mixture GP Regression. It uses gpflow.models.GPR
    The prediction is done on a set of GP models while each is trained as standard GPR model.
    see 
    Christoffer Riis et al. 2023, Mixture of Gaussian Processes for Bayesian Active Learning

    Attributes:
        kernel: an instance of a subclass of gpflow.kernels.Kernel - the kernel parameters need to be equipped with priors!
        model: holds the gpflow.models.GPR instance
        optimize_hps: bool if hyperparameters are fine-tuned
        train_likelihood_variance: bool if likelihood variance is trained
        observation_noise: observation noise level - is either set fixed to that value or acts as initial starting value for optimization
        num_samples: number of models
        num_finessing_steps: number of steps fitted to the model after the initial sampling
        initialization_type: InitializationType Enum that specifies how the starting point of the HMC chain should be generated
        prediction_quantity: PredictionQuantity Emum that specifies if P(y|x,D) or P(f|x,D) should be approximated for prediction
        samples: list of posterior_draws for the unconstrained parameters (called variables in this context - as they are the tensorflow variables)
        parameter_samples: list of posterior draws transformend to the constrained parameters
    """

    def __init__(
        self,
        kernel: gpflow.kernels.Kernel,
        observation_noise: float,
        expected_observation_noise: float,
        optimize_hps: bool,
        train_likelihood_variance: bool,
        num_samples: int = 30,
        num_finessing_steps: int = 30,
        retrain_when_failed: bool = True,
        prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y,
        entropy_approximation: EntropyApproximation = EntropyApproximation.MOMENT_MATCHED_GAUSSIAN,
        **kwargs
    ):
        self.observation_noise = observation_noise
        self.expected_observation_noise = expected_observation_noise
        self.model = None
        self.num_samples = num_samples
        self.num_finessing_steps = num_finessing_steps
        self.retrain_when_failed = retrain_when_failed
        self.kernel = gpflow.utilities.deepcopy(kernel)
        assert not isinstance(self.kernel, InputInitializedKernelInterface)  # not yet implemented
        self.optimize_hps = optimize_hps
        self.train_likelihood_variance = train_likelihood_variance
        self.prediction_quantity = prediction_quantity
        self.entropy_approximation = entropy_approximation
        self.target_acceptance_prob = 0.75
        self.use_mean_function = False
        self.samples = []
        self.parameter_samples = []
        if isinstance(kernel, RegularizedKernelInterface):
            self.add_additional_kernel_hp_prior = True
        else:
            self.add_additional_kernel_hp_prior = False

    def reset_model(self):
        pass

    def set_number_of_samples(self, num_samples: int):
        """
        Setter method for the number of posterior samples that should be drawn

        Arguments:
            num_samples: number of posterior samples
        """
        self.num_samples = num_samples

    def set_mean_function(self, constant: float):
        """
        Setter method, to set the mean function to a constant

        Arguments:
            constant: mean function constant
        """
        self.use_mean_function = True
        self.mean_function = gpflow.mean_functions.Constant(c=constant)

    def build_model(self, x_data: np.array, y_data: np.array):
        """
        Method that builds the initial gpflow.model.GPR model

        Arguments:
            x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
        """
        self.model_list = []
        for i in range(self.num_samples):
            if self.use_mean_function:
                model = gpflow.models.GPR(
                    data=(x_data, y_data),
                    kernel=gpflow.utilities.deepcopy(self.kernel),
                    mean_function=gpflow.utilities.deepcopy(self.mean_function),
                    noise_variance=np.power(self.observation_noise, 2.0),
                )
                set_trainable(model.mean_function.c, False)
            else:
                model = gpflow.models.GPR(
                    data=(x_data, y_data),
                    kernel=gpflow.utilities.deepcopy(self.kernel),
                    mean_function=None,
                    noise_variance=np.power(self.observation_noise, 2.0),
                )

            if self.train_likelihood_variance:
                model.likelihood.variance.prior = tfd.Exponential(1.0 / np.power(self.expected_observation_noise, 2.0))
            else:
                set_trainable(model.likelihood.variance, False)

            self.model_list.append(model)
            self.model_weights = np.array([1.0 / self.num_samples] * self.num_samples)

    def optimize_hyperparameters_with_adam(self):
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        likelihood_list = [] # use this later to assign weights
        for i, model in enumerate(self.model_list):
            def loss_closure():
                return self.training_loss(model)
            for step in range(self.num_finessing_steps):
                try:
                    optimizer.minimize(loss_closure, model.trainable_variables)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    logger.error(f"Model {i}-th: error in optimization - {e}")
                logger.debug(f"Model {i}-th: step {step} - loss: {loss_closure().numpy()}")
            # store marginal likelihood for this model
            likelihood_list.append(
                np.exp( self.log_posterior_density(model) )
            )
        logger.debug("-Optimization done")
        self.model_weights = np.array(likelihood_list) / np.sum(likelihood_list)

    def optimize_hyperparameters(self):
        """
        Method for performing Type-2 ML infernence - optimization is repeated if convergence was not succesfull or cholesky was not possible
        pertubation of initial values is applied in this case.
        If kernel parameters have prior this method automatically turns to MAP estimation!!
        """
        optimizer = gpflow.optimizers.Scipy()
        likelihood_list = [] # use this later to assign weights
        for i, model in enumerate(self.model_list):
            def loss_closure():
                return self.training_loss(model)

            optimization_success = False
            while not optimization_success:
                try:
                    opt_res = optimizer.minimize(loss_closure, model.trainable_variables, options={'maxiter': self.num_finessing_steps})
                    optimization_success = opt_res.success
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    logger.error(f"Model {i}-th: error in optimization - try again")
                    self.draw_from_hyperparameter_prior(model)
                if not optimization_success and self.retrain_when_failed:
                    logger.warning(f"Model {i}-th: not converged - try again")
                    self.draw_from_hyperparameter_prior(model)
                else:
                    logger.debug(f"Model {i}-th: optimization succesful - learned parameters:")
                    if logger.isEnabledFor(logging.DEBUG):
                        print_summary(model)
            # store marginal likelihood for this model
            likelihood_list.append(
                np.exp( self.log_posterior_density(model) )
            )
        logger.debug("-Optimization done")
        self.model_weights = np.array(likelihood_list) / np.sum(likelihood_list)

    def draw_from_hyperparameter_prior(self, model: gpflow.models.GPR, show_draw: bool = True):
        """
        Method for drawing a sample from the prior and setting the kernel parameters to this sample.

        Arguments:
            model: gpflow.models.GPR model that should be sampled
            show_draw: bool if the drawn parameters should be printed
        """
        logger.info("-Draw from hyperparameter prior")

        for parameter in model.trainable_parameters:
            try:
                new_value = parameter.prior.sample()
                parameter.assign(new_value)
            except:
                logger.warning(f"{parameter.prior.name} give invalid sample {new_value}, skip initial drawing")

        if show_draw and logger.isEnabledFor(logging.DEBUG):
            print_summary(model)

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Main training method. First builds a gpflow.model.GPR model and initializes the HP's (either to MAP or prior draw). It then start the HMC procedure
        and collects self.num_samples posterior draws of the HP's given the data and stores the samples in self.parameter_samples

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        y_data: Label array with shape (n,1) where n is the number of training points
        """
        self.build_model(x_data, y_data)
        for model in self.model_list:
            self.draw_from_hyperparameter_prior(model)
        if self.optimize_hps:
            #self.optimize_hyperparameters()
            self.optimize_hyperparameters_with_adam()

    def log_posterior_density(self, model: gpflow.models.GPR) -> tf.Tensor:
        if self.add_additional_kernel_hp_prior:
            return model.log_posterior_density() - model.kernel.regularization_loss(model.data[0])
        else:
            return model.log_posterior_density()

    def training_loss(self, model: gpflow.models.GPR) -> tf.Tensor:
        return -1 * self.log_posterior_density(model)

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> float:
        """
        Arguments:
            x_data: (only necessary if infer was not yet called) Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: (only necessary if infer was not yet called) Label array with shape (n,1) where n is the number of training points

        Returns:
            marginal likelihood value of infered model
        """
        if self.model is None and x_data is not None and y_data is not None:
            self.infer(x_data, y_data)
        marginal_likelihoods = []
        for i, model in enumerate(self.model_list):
            w = self.model_weights[i]
            ll = self.log_posterior_density(model)
            marginal_likelihoods.append(w * np.exp(ll))
        return np.mean(marginal_likelihoods)

    def predict(self, x_test: np.array, prediction_quantity: PredictionQuantity) -> Tuple[np.array, np.array]:
        """
        Inner method for getting predictive distributions associated with each model sample - collected in summarized mean and sigma arrays

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (self.num_samples,n)
        sigma array with shape (self.num_samples,n)
        """
        pred_mus_over_samples = []
        pred_sigmas_over_samples = []
        for i, model in enumerate(self.model_list):
            if prediction_quantity == PredictionQuantity.PREDICT_F:
                pred_mus, pred_vars = model.predict_f(x_test)
            elif prediction_quantity == PredictionQuantity.PREDICT_Y:
                pred_mus, pred_vars = model.predict_y(x_test)
            pred_sigmas = np.sqrt(pred_vars)
            pred_mus_over_samples.append(pred_mus)
            pred_sigmas_over_samples.append(pred_sigmas)
        pred_mus_complete = np.array(pred_mus_over_samples)
        pred_sigmas_complete = np.array(pred_sigmas_over_samples)
        return pred_mus_complete, pred_sigmas_complete

    def predict_full_cov(self, x_test: np.array, prediction_quantity: PredictionQuantity) -> Tuple[np.array, np.array]:
        """
        Inner method for getting full predictive distribution over all test points associated with each HMC sample - collected in summarized mean and covariance arrays

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (self.num_samples,n)
        covariance array with shape (self.num_samples,n,n)
        """

        pred_mus_over_samples = []
        pred_covs_over_samples = []
        for i, model in enumerate(self.model_list):
            if prediction_quantity == PredictionQuantity.PREDICT_F:
                pred_mus, pred_covs = model.predict_f(x_test, full_cov=True)
            elif prediction_quantity == PredictionQuantity.PREDICT_Y:
                pred_mus, pred_covs = model.predict_y(x_test, full_cov=True)
            pred_mus_over_samples.append(np.squeeze(pred_mus))
            pred_covs_over_samples.append(np.squeeze(pred_covs))
        pred_mus_complete = np.array(pred_mus_over_samples)
        pred_covs_complete = np.array(pred_covs_over_samples)
        return pred_mus_complete, pred_covs_complete

    def predictive_dist(self, x_test: np.array, best_mode: bool=True) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points
        best_mode: bool - if True then only the dominant model is used for prediction, otherwise all models are used
        
        Returns:
        mean array with shape (n,)
        sigma array with shape (n,)
        """
        logger.info("Predict")

        if best_mode:
            best_model_index = np.argmax(self.model_weights)
            if self.prediction_quantity == PredictionQuantity.PREDICT_F:
                pred_mus, pred_vars = self.model_list[best_model_index].predict_f(x_test)
            elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
                pred_mus, pred_vars = self.model_list[best_model_index].predict_y(x_test)
            pred_sigmas = np.sqrt(pred_vars)
            return np.squeeze(pred_mus), np.squeeze(pred_sigmas)
        else:
            pred_mus_complete, pred_sigmas_complete = self.predict(x_test, self.prediction_quantity)
            n = x_test.shape[0]
            mus_over_inputs = []
            sigmas_over_inputs = []
            weights = self.num_samples * self.model_weights
            for i in range(0, n):
                weighted_mus = np.multiply(pred_mus_complete[:, i], weights)
                vars = np.power(pred_sigmas_complete[:, i], 2.0)
                weighted_vars = np.multiply(vars, weights)
                mu = np.mean(weighted_mus)
                var = np.mean(
                    weighted_vars + \
                    np.power(pred_mus_complete[:, i], 2.0) - \
                    np.power(mu, 2.0)
                )
                mus_over_inputs.append(mu)
                sigmas_over_inputs.append(np.sqrt(var))
            return np.array(mus_over_inputs), np.array(sigmas_over_inputs)

    def get_predictive_distributions(self, x_test: np.array) -> List[Tuple[np.array, np.array]]:
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test, self.prediction_quantity)
        pred_dists = [
            (   self.model_weights[j],
                np.squeeze(pred_mus_complete[j, :]),
                np.squeeze(pred_sigmas_complete[j, :])
            ) for j in range(0, self.num_samples)]
        return pred_dists

    def entropy_predictive_dist_full_cov(self, x_test: np.array) -> float:
        """
        Estimates entropy of full predictive distribution over all test points (batch mode) - predictive distribution is an n dim mixture of Gaussian
        - entropy is approximated with that of a gaussian distribution with same covariance matrix as the mixture

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        float - entropy of the full predictive dist over test points
        """
        pred_mus, pred_sigmas = self.predict_full_cov(x_test, self.prediction_quantity)
        n_samples = len(pred_mus)
        assert n_samples == self.num_samples
        gmm = GaussianMixtureDensityNd(self.model_weights, pred_mus, pred_sigmas)
        entropy = gmm.entropy_gaussian_approx()
        return entropy

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Calculates entropies of the predictive distributions at the test points - is the entropy of a mixture of gaussian - entropy is approximated via quadrature

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        np.array with shape (n,) containing the entropies for each test point
        """
        logger.info("Calculate entropy")
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test, self.prediction_quantity)
        n = x_test.shape[0]
        m_posterior_draws = len(pred_mus_complete[:, 0])
        assert m_posterior_draws == self.num_samples
        entropies = []
        for i in range(0, n):
            gmm_at_test_point = GaussianMixtureDensity(self.model_weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            gmm_at_test_point.set_entropy_approx_type(self.entropy_approximation)
            entropy = gmm_at_test_point.entropy()
            logger.debug(str(i) + "/" + str(n) + ":")
            logger.debug(entropy)
            entropies.append(entropy)
        return np.array(entropies)

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        """
        Method for calculating the log likelihood value of the the predictive distribution at the test input points (evaluated at the output values)
        - method is therefore for validation purposes only

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points
        y_test: Array of test output points with shape (n,1)

        Returns:
        array of shape (n,) with log liklihood values
        """
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test, PredictionQuantity.PREDICT_Y)
        n = x_test.shape[0]
        m_posterior_draws = len(pred_mus_complete[:, 0])
        assert m_posterior_draws == self.num_samples
        log_likelis = []
        for i in range(0, n):
            gmm_at_test_point = GaussianMixtureDensity(self.model_weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            log_likeli = gmm_at_test_point.log_likelihood(np.squeeze(y_test[i]))
            log_likelis.append(log_likeli)
        return np.squeeze(np.array(log_likelis))
