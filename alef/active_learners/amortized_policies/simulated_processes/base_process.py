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

from enum import Enum
import numpy as np
import torch
from torch import nn
import pyro
import pyro.distributions as dist
from typing import Tuple, Optional, Sequence
from abc import abstractmethod

import logging
from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)

class XPDF(Enum):
    UNIFORM = 1
    BETA = 2

class BaseSimulatedProcess(nn.Module):

    def __init__(
        self,
        design_net: nn.Module,
        n_initial_min: int,
        n_initial_max: Optional[int] = None,
        n_steps_min: Optional[int] = None,
        n_steps_max: int = 1
    ):
        """
        :param design_net: NN to run AL
        :param n_initial_min: int, min num of initial observations in the simulation
        :param n_initial_max: int, max num of initial observations in the simulation
        :param n_steps_min: int, min num of NN queries in the simulation 
        :param n_steps_max: int, max num of NN queries in the simulation 
        """
        super().__init__()
        self.design_net = design_net
        self.set_n_initial(n_initial_min, n_initial_max)
        self.set_n_steps(n_steps_min, n_steps_max)

    def set_n_initial(self, n_initial_min: int=1, n_initial_max: Optional[int]=None):
        assert n_initial_min >= 0, 'n_initial (lower, upper), lower must be non negative'
        self.n_initial_min = n_initial_min
        if n_initial_max is None or n_initial_max < n_initial_min:
            logger.warning(
                f'n_initial=[lower, upper]=[{n_initial_min}, {n_initial_max}] is invalid'
            )
            logger.warning(
                f'set to n_initial = [{n_initial_min}, {n_initial_min}]'
            )
            self.n_initial_max = n_initial_min
        else:
            self.n_initial_max = n_initial_max

    def set_n_steps(self, n_steps_min: int=None, n_steps_max: int=1):
        assert n_steps_max > 0, 'n_steps_max must be positive'
        self.n_steps_max = n_steps_max
        if n_steps_min is None or n_steps_min <= 0 or n_steps_min > n_steps_max:
            logger.warning(
                f'n_steps_min={n_steps_min} is invalid, must be in (0, n_steps_max]'
            )
            logger.warning(
                f'set to n_steps_min = n_steps_max = {n_steps_max} instead'
            )
            self.n_steps_min = n_steps_max
        else:
            self.n_steps_min = n_steps_min

    @property
    def input_domain(self):
        return self.design_net.input_domain

    @property
    def flexible_dimension(self):
        return self.design_net.flexible_dimension

    def set_device(self, device: torch.device):
        self.to(device)

    @abstractmethod
    def global_variables_in_dict(self):
        r"""
        return global variables in dictionary
        """
        raise NotImplementedError

    @abstractmethod
    def process(self):
        """
        simulated experiment process for policy training.
        """
        raise NotImplementedError

    @abstractmethod
    def validation(self):
        """
        simulated experiment process for policy evaluation.
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def _get_x_pdf(
        self,
        pdf_type: XPDF,
        batch_sizes: Sequence[int],
        event_sizes: Sequence[int],
        input_domain: Sequence,
    ):
        if pdf_type == XPDF.UNIFORM:
            if len(event_sizes) > 0:
                x_pdf = dist.Uniform(*input_domain).expand(batch_sizes + event_sizes).to_event(len(event_sizes))
            else:
                x_pdf = dist.Uniform(*input_domain).expand(batch_sizes)
        elif pdf_type == XPDF.BETA:
            if len(event_sizes) > 0:
                x_pdf = dist.TransformedDistribution(
                    dist.Beta(0.5, 0.5*torch.ones(batch_sizes+event_sizes, device=self.device)),
                    [torch.distributions.AffineTransform(input_domain[0], input_domain[1] - input_domain[0])]
                ).to_event(2)
            else:
                x_pdf = dist.TransformedDistribution(
                    dist.Beta(0.5, 0.5*torch.ones(batch_sizes, device=self.device)),
                    [torch.distributions.AffineTransform(input_domain[0], input_domain[1] - input_domain[0])]
                )
        else:
            raise NotImplementedError
        return x_pdf

if __name__=='__main__':
    print(BaseSimulatedProcess)
    print(BaseSimulatedProcess.__class__)
    print(isinstance(BaseSimulatedProcess, BaseSimulatedProcess.__class__))