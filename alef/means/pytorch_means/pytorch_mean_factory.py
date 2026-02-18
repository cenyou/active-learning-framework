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

import torch
import gpytorch
from alef.enums.gpytorch_enums import GPytorchPriorEnum
from alef.utils.gpytorch_priors import UniformPrior, CategoricalWithValues

from alef.configs.means.pytorch_means.base_mean_pytorch_config import BaseMeanPytorchConfig
from alef.configs.means.pytorch_means.zero_mean_pytorch_config import BasicZeroMeanPytorchConfig
from .pytorch_zero_mean import ZeroPytorchMean
from alef.configs.means.pytorch_means.linear_mean_pytorch_config import BasicLinearMeanPytorchConfig
from .pytorch_linear_mean import LinearPytorchMean
from alef.configs.means.pytorch_means.quadratic_mean_pytorch_config import BasicQuadraticMeanPytorchConfig
from .pytorch_quadratic_mean import QuadraticPytorchMean
from alef.configs.means.pytorch_means.periodic_mean_pytorch_config import BasicPeriodicMeanPytorchConfig
from .pytorch_periodic_mean import PeriodicPytorchMean
from alef.configs.means.pytorch_means.sech_mean_pytorch_config import BasicSechMeanPytorchConfig, BasicSechRotatedMeanPytorchConfig
from .pytorch_sech_mean import SechPytorchMean, SechRotatedPytorchMean

def _gpytorch_prior_wrapper(prior_type: GPytorchPriorEnum, *args, **kwargs) -> gpytorch.priors.Prior:
    if prior_type==GPytorchPriorEnum.NONE:
        return None
    elif prior_type==GPytorchPriorEnum.UNIFORM:
        pass_kwargs = {'device': kwargs['device']} if kwargs.get('device') else {}
        decor_args = [torch.tensor(a, **pass_kwargs) if hasattr(a, '__len__') else a for a in args]
        return UniformPrior(*decor_args, **kwargs)
    elif prior_type==GPytorchPriorEnum.CATEGORICAL:
        pass_kwargs = {'device': kwargs['device']} if kwargs.get('device') else {}
        decor_args = [torch.tensor(a, **pass_kwargs) if hasattr(a, '__len__') else a for a in args]
        return CategoricalWithValues(*decor_args, **kwargs)
    else:
        raise NotImplementedError

def _config2dict_with_prior_wrap(config):
    config_dict = config.dict()
    pass_kwargs = {'device': config_dict['device']} if config_dict.get('device') else {}
    for name, v in config.dict().items():
        if name.endswith('_prior') and not v is None:
            prior_type, prior_args = v
            prior = _gpytorch_prior_wrapper(prior_type, *prior_args, **pass_kwargs)
            config_dict[name] = prior
    return config_dict

class PytorchMeanFactory:
    @staticmethod
    def build(mean_config: BaseMeanPytorchConfig):
        if isinstance(mean_config, BasicZeroMeanPytorchConfig):
            return ZeroPytorchMean(**mean_config.dict())
        if isinstance(mean_config, BasicLinearMeanPytorchConfig):
            D = mean_config.input_dimension
            m = LinearPytorchMean(input_size=D, **_config2dict_with_prior_wrap(mean_config))
            m._set_scale( torch.tensor(mean_config.scale, device=m.scale.device) )
            m._set_bias( torch.tensor(mean_config.bias, device=m.bias.device) )
            weights = torch.tensor(mean_config.weights, device=m.weights.device).expand(D)
            m._set_weights( weights.expand(m.weights.shape) )
            return m
        elif isinstance(mean_config, BasicQuadraticMeanPytorchConfig):
            D = mean_config.input_dimension
            m = QuadraticPytorchMean(input_size=D, **_config2dict_with_prior_wrap(mean_config))
            m._set_scale( torch.tensor(mean_config.scale, device=m.scale.device) )
            m._set_bias( torch.tensor(mean_config.bias, device=m.bias.device) )
            center = torch.tensor(mean_config.center, device=m.center.device).expand(D)
            m._set_center( center.expand(m.center.shape) )
            weights = torch.tensor(mean_config.weights, device=m.weights.device).expand(D)
            m._set_weights( weights.expand(m.weights.shape) )
            return m
        elif isinstance(mean_config, BasicPeriodicMeanPytorchConfig):
            D = mean_config.input_dimension
            m = PeriodicPytorchMean(input_size=D, **_config2dict_with_prior_wrap(mean_config))
            m._set_scale( torch.tensor(mean_config.scale, device=m.scale.device) )
            m._set_bias( torch.tensor(mean_config.bias, device=m.bias.device) )
            center = torch.tensor(mean_config.center, device=m.center.device).expand(D)
            m._set_center( center.expand(m.center.shape) )
            weights = torch.tensor(mean_config.weights, device=m.weights.device).expand(D)
            m._set_weights( weights.expand(m.weights.shape) )
            return m
        elif isinstance(mean_config, (BasicSechMeanPytorchConfig, BasicSechRotatedMeanPytorchConfig)):
            D = mean_config.input_dimension
            mean_class = SechPytorchMean if isinstance(mean_config, BasicSechMeanPytorchConfig) else SechRotatedPytorchMean
            m = mean_class(input_size=D, **_config2dict_with_prior_wrap(mean_config))
            m._set_scale( torch.tensor(mean_config.scale, device=m.scale.device) )
            m._set_bias( torch.tensor(mean_config.bias, device=m.bias.device) )
            center = torch.tensor(mean_config.center, device=m.center.device).expand(D)
            m._set_center( center.expand(m.center.shape) )
            weights = torch.tensor(mean_config.weights, device=m.weights.device).expand(D)
            m._set_weights( weights.expand(m.weights.shape) )
            return m
        else:
            raise NotImplementedError(f"invalid mean config {mean_config}")


