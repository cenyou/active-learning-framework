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

import re
import json
import torch
from typing import Union
from pathlib import Path
import logging
from alef.utils.custom_logging import getLogger

from alef.configs.active_learners.amortized_policies import policy_configs
from alef.configs.active_learners.amortized_policies.policy_configs import (
    BaseAmortizedPolicyConfig,
    ContinuousGPPolicyConfig,
    SafetyAwareContinuousGPPolicyConfig,
    ContinuousGPFlexDimPolicyConfig,
    SafetyAwareContinuousGPFlexDimPolicyConfig,
)
from alef.active_learners.amortized_policies.nn.policies import (
    ContinuousGPPolicy,
    ContinuousGPBudgetParsedPolicy,
    SafetyAwareContinuousGPPolicy,
    SafetyAwareContinuousGPBudgetParsedPolicy,
)
from alef.active_learners.amortized_policies.nn.policies_flex_dim import (
    ContinuousGPFlexDimPolicy,
    ContinuousGPFlexDimBudgetParsedPolicy,
    SafetyAwareContinuousGPFlexDimPolicy,
    SafetyAwareContinuousGPFlexDimBudgetParsedPolicy,
)

logger = getLogger(__name__)

def _load_model_dict(model_dict_path, device):
    model_path = Path(model_dict_path)
    logger.info(f'Load model: {model_path.as_posix()}')
    assert model_path.exists()
    model_state_dict = torch.load(model_path, map_location=device)
    return model_state_dict


class AmortizedPolicyFactory:
    @staticmethod
    def load_config(path):
        """
        these should exist:
            path \\ params \\ policy_config_{config_name}.json
            path \\ result \\ model_checkpoint_final.pth
        
        """
        root_path = Path(path)
        config_path = list( (root_path / 'params' ).glob('policy_config_*.json') )
        assert len(config_path)==1
        config_path = config_path[0]
        config_name = re.match('policy_config_(.*).json', config_path.name).group(1)
        # load json file to python dict
        with open(config_path, mode='r') as p:
            config_dict = json.load(p)
        if isinstance(config_dict, str):
            config_dict = json.loads(config_dict)
        # remove saved training device and add policy path
        del config_dict['device']
        policy_path = root_path / 'result' / 'model_checkpoint_final.pth'
        config_dict['resume_policy_path'] = policy_path.as_posix()

        # return config
        config_class = getattr(policy_configs, config_name)
        config = config_class(**config_dict)
        # print summary
        config_dict_print = json.dumps(config_dict)#, indent=4)
        logger.info(f'Load config:\n{config_name}({config_dict_print})')
        return config

    @staticmethod
    def build(policy_config: BaseAmortizedPolicyConfig):
        if isinstance(policy_config, ContinuousGPPolicyConfig):
            policy_class = ContinuousGPBudgetParsedPolicy if policy_config.forward_with_budget else ContinuousGPPolicy
        elif isinstance(policy_config, SafetyAwareContinuousGPPolicyConfig):
            policy_class = SafetyAwareContinuousGPBudgetParsedPolicy if policy_config.forward_with_budget else SafetyAwareContinuousGPPolicy
        elif isinstance(policy_config, ContinuousGPFlexDimPolicyConfig):
            policy_class = ContinuousGPFlexDimBudgetParsedPolicy if policy_config.forward_with_budget else ContinuousGPFlexDimPolicy
        elif isinstance(policy_config, SafetyAwareContinuousGPFlexDimPolicyConfig):
            policy_class = SafetyAwareContinuousGPFlexDimBudgetParsedPolicy if policy_config.forward_with_budget else SafetyAwareContinuousGPFlexDimPolicy
        else:
            raise NotImplementedError(f"Invalid config: {policy_config.__class__.__name__}")

        if policy_config.forward_with_budget:
            logger.info('Policy forward with budget')
        policy = policy_class(**policy_config.dict()).to(policy_config.device)
        if not policy_config.resume_policy_path is None and policy_config.resume_policy_path.endswith('.pth'):
            model_state_dict = _load_model_dict(policy_config.resume_policy_path, policy_config.device)
            policy.load_state_dict(model_state_dict)
        return policy
