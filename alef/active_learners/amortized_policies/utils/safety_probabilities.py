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

import math
import torch
from torch import nn
from alef.configs.base_parameters import NUMERICAL_POSITIVE_LOWER_BOUND

class BaseSafetyProbability(nn.Module):
    def __init__(self, alpha: float, beta: float):
        r"""
        the idea here is to have functions used for safety loss discount, i.e. loss = information_loss * discount(Z)
        During the training, we have ground truth of Z, thus we here design the discount probability function
        based on the following principles:
        1) the function scale as a probability density func:
                the discount function is bijective onto [0, 1] (which is why it is called SafetyProbability)
        2) the function is differentiable:
                this function is differentiable and has easy to access close form
        3) we control the probability function shape:
                we should be able to hyper parameters (alpha, beta) to
                manipulate the discount function (sharpness & location), assuming a safety threshold 0
        """
        super().__init__()
        self.set_hyper_parameters(alpha, beta)

    def set_hyper_parameters(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta


class TrivialSafetyProbability(BaseSafetyProbability):
    def forward(self, z: torch.Tensor):
        return torch.ones_like(z)


class SigmoidSafetyProbability(BaseSafetyProbability):
    """
    Sigmoid safety probability function
    p(z) = sigmoid(az + b) = 1 / (1 + exp(-az - b))
    here we can simply control the function shape by setting
    p(0) = 1 - alpha
    p(beta) = 0.5
    """
    def set_hyper_parameters(self, alpha: float, beta: float):
        assert alpha >= NUMERICAL_POSITIVE_LOWER_BOUND
        assert alpha <= 1 - NUMERICAL_POSITIVE_LOWER_BOUND
        assert alpha >= 0.5 + NUMERICAL_POSITIVE_LOWER_BOUND or \
            alpha <= 0.5 - NUMERICAL_POSITIVE_LOWER_BOUND,\
            f"This is ill-defined, alpha cannot be 0.5 (the following interval is numerically considered as 0.5: ({0.5 - NUMERICAL_POSITIVE_LOWER_BOUND}, {0.5 + NUMERICAL_POSITIVE_LOWER_BOUND}))."
        assert beta >= NUMERICAL_POSITIVE_LOWER_BOUND or \
            beta <= - NUMERICAL_POSITIVE_LOWER_BOUND,\
            f"This is ill-defined, beta must be non-zero (the following interval is numerically considered as zero: (-{NUMERICAL_POSITIVE_LOWER_BOUND}, {NUMERICAL_POSITIVE_LOWER_BOUND}))."
        super().set_hyper_parameters(alpha, beta)
        """
        b:
        solve p(0) = sigmoid(b) = 1 / (1 + exp(-b)) = 1 - alpha,
        we get
        b = log( (1 - alpha) / alpha )
        
        a:
        denote c = beta
        solve p(c) = 0.5 <=> sigmoid(ac + b) = 1 / ( 1 + exp(-ac-b) ) = 0.5
        <=> exp(-ac-b) = 1 <=> a = -b / c
        """
        self.__b = math.log(1 - self.alpha) - math.log(self.alpha)
        self.__a = - self.b / self.beta
        self.layer = nn.Sigmoid()

    def function_expression(self):
        return 'p(z) = sigmoid(%.2f z + %.2f)'%(self.a, self.b)

    @property
    def b(self):
        return self.__b

    @property
    def a(self):
        return self.__a

    def forward(self, z: torch.Tensor):
        """
        return sigmoid(az + b)
        """
        sigmoid = self.layer(self.a * z + self.b)
        return sigmoid
        #return (1 - NUMERICAL_POSITIVE_LOWER_BOUND) * sigmoid + NUMERICAL_POSITIVE_LOWER_BOUND



class _rescale_softplus(nn.Softplus):
    def __init__(self, slope: float, upper_bound: float, sharpness: float):
        super().__init__(beta=sharpness, threshold=50/sharpness)
        self._sharpness = sharpness
        self.slope = slope
        self.upper_bound = upper_bound

    def function_expression(self):
        return ' - softplus(beta=%.2f)(- %.2fz ) + %.2f)'%(self._sharpness, self.slope, self.upper_bound)

    def forward(self, x: torch.Tensor):
        return - super().forward(- self.slope * x ) + self.upper_bound

class SigmoidSoftplusSafetyProbability(BaseSafetyProbability):
    """
    Sigmoid safety probability function
    p(z) = sigmoid( - softplus_(2)(- alpha z) + beta)
        - softplus(- alpha z) ~~ - alpha z for z >> 0
        beta should be the upper bound of sensitive domain (normal distribution upper bound to 3 is fine)
    """
    def set_hyper_parameters(self, alpha: float, beta: float):
        #assert alpha >= NUMERICAL_POSITIVE_LOWER_BOUND
        #assert alpha <= 1 - NUMERICAL_POSITIVE_LOWER_BOUND
        super().set_hyper_parameters(alpha, beta)
        self.layer1 = _rescale_softplus(slope=self.alpha, upper_bound=self.beta, sharpness=2)
        # threshold is not important to us, just make sure it is large enough
        self.layer2 = nn.Sigmoid()

    def function_expression(self):
        return 'p(z) = sigmoid(%s)'%(self.layer1.function_expression())

    def forward(self, z: torch.Tensor):
        sigmoid = self.layer2( self.layer1(z) )
        return sigmoid
        #return (1 - NUMERICAL_POSITIVE_LOWER_BOUND) * sigmoid + NUMERICAL_POSITIVE_LOWER_BOUND
