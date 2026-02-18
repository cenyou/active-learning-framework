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

from alef.configs.acquisition_functions import (
    BasicRandomConfig,
    BasicNosafeRandomConfig,
    BasicSafeRandomConfig,
    BasicPredVarianceConfig,
    BasicPredEntropyConfig,
    BasicPredSigmaConfig,
    BasicSafePredEntropyConfig,
    BasicSafePredEntropyAllConfig,
    BasicSafeDiscountPredEntropyConfig,
    BasicSafeDiscountPredEntropyAllConfig,
    BasicMinUnsafePredEntropyConfig,
    BasicMinUnsafePredEntropyAllConfig,
    BasicSafeOptConfig,
    BasicSafeGPUCBConfig,
    BasicEIConfig,
    BasicSafeEIConfig,
)
from alef.configs.acquisition_functions.bo_acquisition_functions.gp_ucb_config import BasicGPUCBConfig
from alef.configs.active_learners.oracle_active_learner_configs import (
    PredEntropyOracleActiveLearnerConfig,
    PredVarOracleActiveLearnerConfig,
    PredSigmaOracleActiveLearnerConfig,
    RandomOracleActiveLearnerConfig,
)
from alef.configs.active_learners.oracle_policy_active_learner_configs import BasicOraclePolicyActiveLearnerConfig
from alef.configs.active_learners.pool_policy_active_learner_configs import BasicPoolPolicyActiveLearnerConfig
from alef.configs.active_learners.oracle_policy_safe_active_learner_configs import BasicOraclePolicySafeActiveLearnerConfig
from alef.configs.active_learners.pool_policy_safe_active_learner_configs import BasicPoolPolicySafeActiveLearnerConfig
from alef.configs.active_learners.pool_active_learner_configs import (
    PredEntropyPoolActiveLearnerConfig,
    PredVarPoolActiveLearnerConfig,
    PredSigmaPoolActiveLearnerConfig,
    RandomPoolActiveLearnerConfig,
)

class ConfigPicker:

    acquisition_function_configs_dict = {
        c.__name__: c
        for c in [
            BasicRandomConfig,
            BasicNosafeRandomConfig,
            BasicSafeRandomConfig,
            BasicPredVarianceConfig,
            BasicPredSigmaConfig,
            BasicPredEntropyConfig,
            BasicSafePredEntropyConfig,
            BasicSafePredEntropyAllConfig,
            BasicSafeDiscountPredEntropyConfig,
            BasicSafeDiscountPredEntropyAllConfig,
            BasicMinUnsafePredEntropyConfig,
            BasicMinUnsafePredEntropyAllConfig,
            BasicSafeOptConfig,
            BasicSafeGPUCBConfig,
            BasicEIConfig,
            BasicGPUCBConfig,
            BasicSafeEIConfig,
        ]
    }

    active_learner_configs_dict = {
        c.__name__: c
        for c in [
            PredEntropyPoolActiveLearnerConfig,
            PredVarPoolActiveLearnerConfig,
            PredSigmaPoolActiveLearnerConfig,
            RandomPoolActiveLearnerConfig,
            PredEntropyOracleActiveLearnerConfig,
            PredVarOracleActiveLearnerConfig,
            PredSigmaOracleActiveLearnerConfig,
            RandomOracleActiveLearnerConfig,
            BasicOraclePolicyActiveLearnerConfig,
            BasicPoolPolicyActiveLearnerConfig,
            BasicOraclePolicySafeActiveLearnerConfig,
            BasicPoolPolicySafeActiveLearnerConfig,
        ]
    }

    @staticmethod
    def pick_acquisition_function_config(config_class_name):
        return ConfigPicker.acquisition_function_configs_dict[config_class_name]

    @staticmethod
    def pick_active_learner_config(config_class_name):
        return ConfigPicker.active_learner_configs_dict[config_class_name]
