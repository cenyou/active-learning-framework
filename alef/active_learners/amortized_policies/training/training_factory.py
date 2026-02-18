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

from typing import Union
from functools import partial
import pyro

from alef.configs.active_learners.amortized_policies.loss_configs import (
    # basic configs
    BasicAmortizedPolicyLossConfig,
    BasicSafetyAwarePolicyLossConfig,
    BasicLossCurriculumConfig,
    BasicNonMyopicLossConfig,
    BasicMyopicLossConfig,
    BasicNonMyopicSafetyLossConfig,
    BasicMyopicSafetyLossConfig,
    # non myopic loss configs
    DADLossConfig,
    _ScoreDADLossConfig,
    GPEntropy1LossConfig,
    GPEntropy2LossConfig,
    GPMI1LossConfig,
    GPMI2LossConfig,
    #GPPCELossConfig,
    # myopic loss configs,
    GPMyopicEntropy1LossConfig,
    GPMyopicEntropy2LossConfig,
    GPMyopicMI1LossConfig,
    GPMyopicMI2LossConfig,
    # non myopic safe loss configs
    SafeGPEntropy1LossConfig,
    SafeGPEntropy2LossConfig,
    SafeGPMI1LossConfig,
    SafeGPMI2LossConfig,
    #SafeGPPCELossConfig,
    # myopic safe
    SafeGPMyopicEntropy1LossConfig,
    SafeGPMyopicEntropy2LossConfig,
    SafeGPMyopicMI1LossConfig,
    SafeGPMyopicMI2LossConfig,
)
from alef.active_learners.amortized_policies.losses import(
    TrivialLossCurriculum,
    LossCurriculum,
    # non myopic losses
    PriorContrastiveEstimation,
    PriorContrastiveEstimationScoreGradient,
    GPEntropy1Loss,
    GPEntropy2Loss,
    GPMutualInformation1Loss,
    GPMutualInformation2Loss,
    #GPMutualInformationPCELoss,
    # myopic losses
    GPMyopicEntropy1Loss,
    GPMyopicEntropy2Loss,
    GPMyopicMutualInformation1Loss,
    GPMyopicMutualInformation2Loss,
    # safety loss wrappers
    GPSafetyEntropyWrapLoss,
    GPSafetyMIWrapLoss,
)
from alef.active_learners.amortized_policies.simulated_processes import (
    SequentialGaussianProcessContinuousDomain,
    SequentialSafeGaussianProcessContinuousDomain,
)
from alef.active_learners.amortized_policies.amortized_policy_factory import AmortizedPolicyFactory
from alef.active_learners.amortized_policies.training.dad_oed import OED
#from alef.active_learners.amortized_policies.training.idad_oed import OED
from alef.configs.active_learners.amortized_policies.training_configs import (
    BaseAmortizedPolicyTrainingConfig,
    AmortizedContinuousFixGPPolicyTrainingConfig,
    AmortizedContinuousRandomGPPolicyTrainingConfig,
)

class _LossPicker:
    @staticmethod
    def pick_information_loss(loss_config: BasicAmortizedPolicyLossConfig):
        if isinstance(loss_config, BasicNonMyopicLossConfig):
            myopic_flag = False
            if isinstance(loss_config, DADLossConfig):
                loss = PriorContrastiveEstimation(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            elif isinstance(loss_config, _ScoreDADLossConfig):
                loss = PriorContrastiveEstimationScoreGradient(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            elif isinstance(loss_config, GPEntropy1LossConfig):
                loss = GPEntropy1Loss(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            elif isinstance(loss_config, GPEntropy2LossConfig):
                loss = GPEntropy2Loss(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            elif isinstance(loss_config, GPMI1LossConfig):
                loss = GPMutualInformation1Loss(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            elif isinstance(loss_config, GPMI2LossConfig):
                loss = GPMutualInformation2Loss(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            #elif isinstance(loss_config, GPPCELossConfig):
            #    loss = GPMutualInformationPCELoss(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            else:
                raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")
            return loss, myopic_flag
        elif isinstance(loss_config, BasicMyopicLossConfig):
            myopic_flag = True
            if isinstance(loss_config, GPMyopicEntropy1LossConfig):
                loss = GPMyopicEntropy1Loss(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            elif isinstance(loss_config, GPMyopicEntropy2LossConfig):
                loss = GPEntropy2Loss(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            elif isinstance(loss_config, GPMyopicMI1LossConfig):
                loss = GPMyopicMutualInformation1Loss(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            elif isinstance(loss_config, GPMyopicMI2LossConfig):
                loss = GPMyopicMutualInformation2Loss(**loss_config.dict(exclude={'num_epochs', 'epochs_size'}))
            else:
                raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")
            return loss, myopic_flag
        else:
            raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")

    @staticmethod
    def pick_safety_loss(loss_config: BasicSafetyAwarePolicyLossConfig):
        safety_relevant_keys = ['probability_function', 'probability_function_args', 'probability_wrap_mode', 'safety_discount_ratio']
        epoch_relevant_keys = ['num_epochs', 'epochs_size']
        if isinstance(loss_config, BasicNonMyopicSafetyLossConfig):
            myopic_flag = False
            if isinstance(loss_config, SafeGPEntropy1LossConfig):
                information_loss = GPEntropy1Loss(**loss_config.dict(exclude={*safety_relevant_keys, *epoch_relevant_keys}))
                loss = GPSafetyEntropyWrapLoss(
                    information_loss,
                    name = loss_config.name,
                    **loss_config.dict(include={*safety_relevant_keys}),
                )
            elif isinstance(loss_config, SafeGPEntropy2LossConfig):
                information_loss = GPEntropy2Loss(**loss_config.dict(exclude={*safety_relevant_keys, *epoch_relevant_keys}))
                loss = GPSafetyEntropyWrapLoss(
                    information_loss,
                    name = loss_config.name,
                    **loss_config.dict(include={*safety_relevant_keys}),
                )
            elif isinstance(loss_config, SafeGPMI1LossConfig):
                information_loss = GPMutualInformation1Loss(**loss_config.dict(exclude={*safety_relevant_keys, *epoch_relevant_keys}))
                loss = GPSafetyMIWrapLoss(
                    information_loss,
                    name = loss_config.name,
                    **loss_config.dict(include={*safety_relevant_keys}),
                )
            elif isinstance(loss_config, SafeGPMI2LossConfig):
                information_loss = GPMutualInformation2Loss(**loss_config.dict(exclude={*safety_relevant_keys, *epoch_relevant_keys}))
                loss = GPSafetyMIWrapLoss(
                    information_loss,
                    name = loss_config.name,
                    **loss_config.dict(include={*safety_relevant_keys}),
                )
            #elif isinstance(loss_config, SafeGPPCELossConfig):
            #    information_loss = GPMutualInformationPCELoss(**loss_config.dict(exclude={*safety_relevant_keys, *epoch_relevant_keys}))
            #    loss = GPSafetyEntropyWrapLoss(
            #        information_loss,
            #        name = loss_config.name,
            #        **loss_config.dict(include={*safety_relevant_keys}),
            #    )
            else:
                raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")
            return loss, myopic_flag
        elif isinstance(loss_config, BasicMyopicSafetyLossConfig):
            myopic_flag = True
            if isinstance(loss_config, SafeGPMyopicEntropy1LossConfig):
                information_loss = GPMyopicEntropy1Loss(**loss_config.dict(exclude={*safety_relevant_keys, *epoch_relevant_keys}))
                loss = GPSafetyEntropyWrapLoss(
                    information_loss,
                    name = loss_config.name,
                    **loss_config.dict(include={*safety_relevant_keys}),
                )
            elif isinstance(loss_config, SafeGPMyopicEntropy2LossConfig):
                information_loss = GPMyopicEntropy2Loss(**loss_config.dict(exclude={*safety_relevant_keys, *epoch_relevant_keys}))
                loss = GPSafetyEntropyWrapLoss(
                    information_loss,
                    name = loss_config.name,
                    **loss_config.dict(include={*safety_relevant_keys}),
                )
            elif isinstance(loss_config, SafeGPMyopicMI1LossConfig):
                information_loss = GPMyopicMutualInformation1Loss(**loss_config.dict(exclude={*safety_relevant_keys, *epoch_relevant_keys}))
                loss = GPSafetyMIWrapLoss(
                    information_loss,
                    name = loss_config.name,
                    **loss_config.dict(include={*safety_relevant_keys}),
                )
            elif isinstance(loss_config, SafeGPMyopicMI2LossConfig):
                information_loss = GPMyopicMutualInformation2Loss(**loss_config.dict(exclude={*safety_relevant_keys, *epoch_relevant_keys}))
                loss = GPSafetyMIWrapLoss(
                    information_loss,
                    name = loss_config.name,
                    **loss_config.dict(include={*safety_relevant_keys}),
                )
            else:
                raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")
            return loss, myopic_flag
        else:
            raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")

    @staticmethod
    def pick_loss(loss_config: Union[BasicAmortizedPolicyLossConfig, BasicSafetyAwarePolicyLossConfig]):
        r"""
        :param loss_config: Union[BasicAmortizedPolicyLossConfig, BasicSafetyWrapLossConfig]
        :return: loss, myopic_flag, safety_flag
            loss: the loss function
            myopic_flag: bool, whether a myopic pipeline is required
            safety_flag: bool, whether the safety pipeline is required
        """
        if isinstance(loss_config, BasicAmortizedPolicyLossConfig):
            safety_flag = False
            loss, myopic_flag = _LossPicker.pick_information_loss(loss_config)
        elif isinstance(loss_config, BasicSafetyAwarePolicyLossConfig):
            safety_flag = True
            loss, myopic_flag = _LossPicker.pick_safety_loss(loss_config)
        else:
            raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")
        return loss, myopic_flag, safety_flag

class _CurriculumSetter:
    @staticmethod
    def get_loss(
        loss_config: Union[
            BasicAmortizedPolicyLossConfig,
            BasicSafetyAwarePolicyLossConfig,
            BasicLossCurriculumConfig,
        ]
    ):
        r"""
        :param loss_config: Union[BasicAmortizedPolicyLossConfig, BasicSafetyWrapLossConfig, BasicLossCurriculumConfig]
        :return: BaseCurriculum, myopic_flat (bool), safety_flag (bool)
        """
        if isinstance(
            loss_config,
            (
                BasicAmortizedPolicyLossConfig,
                BasicSafetyAwarePolicyLossConfig
            )
        ):
            loss, myopic_flag, safety_flag = _LossPicker.pick_loss(loss_config)
            return TrivialLossCurriculum(
                loss,
                **loss_config.dict(include={'num_epochs', 'epochs_size'})
            ), myopic_flag, safety_flag
        elif isinstance(loss_config, BasicLossCurriculumConfig):
            loss_list = []
            num_epochs_list = []
            epochs_size_list = []
            myopic_flag = False
            safety_flag = False
            for individual_loss_config in loss_config.loss_config_list:
                loss, myopic_flag_i, safety_flag_i = _LossPicker.pick_loss(individual_loss_config)
                if myopic_flag_i:
                    myopic_flag = True
                if safety_flag_i:
                    safety_flag = True
                loss_list.append(loss)
                num_epochs_list.append(individual_loss_config.num_epochs)
                epochs_size_list.append(individual_loss_config.epochs_size)

            return LossCurriculum(loss_list, num_epochs_list, epochs_size_list), myopic_flag, safety_flag
        else:
            raise NotImplementedError(f"Invalid config: {loss_config.__class__.__name__}")

class AmortizedLearnerTrainingFactory:
    @staticmethod
    def build(training_config: BaseAmortizedPolicyTrainingConfig):
        if isinstance(training_config, (AmortizedContinuousFixGPPolicyTrainingConfig, AmortizedContinuousRandomGPPolicyTrainingConfig,) ):
            optimizer = training_config.optimizer
            scheduler = pyro.optim.ExponentialLR(
                {
                    "optimizer": optimizer,
                    "optim_args": training_config.optim_args,
                    "gamma": training_config.gamma,
                }
            )
            loss_function, myopic_flag, safety_flag = _CurriculumSetter.get_loss(training_config.loss_config)
            process_class = SequentialSafeGaussianProcessContinuousDomain if safety_flag else SequentialGaussianProcessContinuousDomain
            
            design_net = AmortizedPolicyFactory.build(training_config.policy_config)
            process = process_class(
                design_net,
                kernel_config = training_config.kernel_config,
                mean_config = training_config.mean_config,
                n_initial_min = training_config.n_initial_min,
                n_initial_max = training_config.n_initial_max + training_config.n_steps_max - 1 if myopic_flag else training_config.n_initial_max,
                n_steps_min = training_config.n_steps_min,
                n_steps_max = 1 if myopic_flag else training_config.n_steps_max,
                sample_gp_prior = training_config.sample_gp_prior,
                lengthscale_distribution = training_config.lengthscale_distribution,
                random_subsequence = training_config.random_subsequence,
                split_subsequence = training_config.split_subsequence,
                device = training_config.policy_config.device,
                safety_kernel_config = training_config.safety_kernel_config, # maybe None
                safety_mean_config = training_config.safety_mean_config, # maybe None
            )

            return OED(process, scheduler, loss_function)
        else:
            raise NotImplementedError(f"Invalid config: {training_config.__class__.__name__}")
