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

from torch import nn
from typing import Tuple, Optional
import torch
from torch.distributions import AffineTransform
from pyro.distributions import Categorical, TransformedDistribution
from alef.active_learners.amortized_policies.simulated_processes.base_process import BaseSimulatedProcess

import logging
from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)

class RandomSubsequenceHelper:
    def _reshape_tensors(self, low, up):
        """
        :param low: int or [*batch_shape] int tensor
        :param up: int or [*batch_shape] int tensor
        
        :return: low, up as the same shape [*batch_shape] of int tensor
        """
        if not isinstance(low, torch.Tensor):
            assert isinstance(up, torch.Tensor)
            l = low * torch.ones(up.shape, dtype=int, device=up.device)
            u = up
        elif not isinstance(up, torch.Tensor):
            assert isinstance(low, torch.Tensor)
            l = l
            u = up * torch.ones(low.shape, dtype=int, device=low.device)
        else:
            shape = torch.broadcast_shapes(low.shape, up.shape)
            l = low.expand(shape)
            u = up.expand(shape)
        return l, u
        
    def index_range2mask(self, low, up, max_len: int):
        """
        index range to mask. The mask has max_len of 1 or 0, 1 if index in [low, up), 0 elsewhere.
        
        :param low: int or [*batch_shape] int tensor
        :param up: int or [*batch_shape] int tensor
        :param max_len: int, ideally larger than up
        
        :return: a mask int tensor of shape [*batch_shape, max_len]
        """
        l, u = self._reshape_tensors(low, up)

        id_mask = torch.arange(0, max_len, device=l.device, dtype=int).expand(
            l.shape + (max_len, )
        ) # [*batch_shape, max_len]
        return torch.where(
            torch.logical_and(id_mask >= l.unsqueeze(-1), id_mask < u.unsqueeze(-1) ),
            torch.ones(l.shape + (max_len, ), device=l.device, dtype=int), # [*batch_shape, max_len]
            torch.zeros(l.shape + (max_len, ), device=l.device, dtype=int), # [*batch_shape, max_len]
        )
        
    def index_distribution(self, low, up):
        """
        index distribution, sample integer values in interval [low, up)
        
        :param low: int or [*batch_shape] int tensor
        :param up: int or [*batch_shape] int tensor
        
        :return: a distribution that can sample i~[low, up) in a batch of size [*batch_shape]
        """
        l, u = self._reshape_tensors(low, up)
        # return a Categorical distribution, shifted by min(l), using AffineTransform
        # we need a probs, size should be [*batch_shape, max(u) - min(l)], note: up is exclusive
        # probs[b, i] = 1 if l[b]-min(l) <= i < u[b]-min(l), else 0
        offset = l.min()
        int_count = u.max() - offset
        l_shifted = l - offset
        u_shifted = u - offset

        probs = self.index_range2mask(l_shifted, u_shifted, int_count)
        dist_shifted = Categorical(probs=probs)
        return TransformedDistribution(
            dist_shifted,
            [AffineTransform(offset, 1)]
        )

class BaseMultiStepsSimulatedProcess(BaseSimulatedProcess):

    def __init__(
        self,
        design_net: nn.Module,
        n_initial_min: int,
        n_initial_max: Optional[int] = None,
        n_steps_min: Optional[int] = None,
        n_steps_max: int = 1,
        random_subsequence: bool = False,
        split_subsequence: bool = False,
    ):
        """
        :param design_net: NN to run AL
        :param n_initial_min: int, min num of initial observations in the simulation
        :param n_initial_max: int, max num of initial observations in the simulation
        :param n_steps_min: int, min num of queries actually queried by NN
        :param n_steps_max: int, max num of queries actually queried by NN
        :param random_subsequence: bool, if we want NN to query random num
        :param split_subsequence: bool, if we want NN to maybe query one sequence first,
            then another sequence, the two sequences sum to to max n_steps_max steps.
            This matters when NN is budget aware.
            This flag is useless if random_subsequence==False
        """
        super().__init__(design_net = design_net, n_initial_min=n_initial_min, n_initial_max=n_initial_max, n_steps_min=n_steps_min, n_steps_max=n_steps_max)
        self.random_subsequence = random_subsequence
        self.split_subsequence = split_subsequence
        self.random_subsequence_helper = RandomSubsequenceHelper()

if __name__=='__main__':
    import numpy as np
    B, Nk, Nf = 3, 4, 5
    l = 0
    u = 1+np.random.choice(9, size=[Nk, Nf])
    helper = RandomSubsequenceHelper()
    dist = helper.index_distribution(0, torch.tensor(u))
    t = dist.sample()
    print(l, u, '\n', t)
    mask = helper.index_range2mask(0, t, 9)
    print(mask)
