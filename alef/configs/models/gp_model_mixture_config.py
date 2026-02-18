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

from alef.configs.models.base_model_config import BaseModelConfig
from alef.configs.kernels.base_kernel_config import BaseKernelConfig
from alef.models.gp_model_marginalized import PredictionQuantity
from alef.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE
from alef.utils.gaussian_mixture_density import EntropyApproximation


class BasicGPModelMixtureConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.01
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    optimize_hps: bool = True
    train_likelihood_variance: bool = True
    num_samples: int = 30
    num_finessing_steps: int = 30
    retrain_when_failed: bool = False
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    entropy_approximation: EntropyApproximation = EntropyApproximation.MOMENT_MATCHED_GAUSSIAN
    name = "GPModelMixture"
