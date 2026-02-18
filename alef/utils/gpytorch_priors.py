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
from gpytorch.priors import Prior
from gpytorch.priors.utils import _bufferize_attributes
from torch.distributions import constraints, Uniform
from torch.distributions.utils import logits_to_probs, probs_to_logits


class UniformPrior(Prior, Uniform):
    """
    Uniform prior adapted from gpytorch's original UniformPrior.
    https://github.com/cornellius-gp/gpytorch/blob/main/gpytorch/priors/torch_priors.py

    We initialize by bufferizing the low and high attributes.
    """
    def __init__(self, a, b, validate_args=None, transform=None):
        torch.nn.Module.__init__(self)
        Uniform.__init__(self, a, b, validate_args=validate_args)
        _bufferize_attributes(self, ("low", "high"))
        self._transform = transform

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return UniformPrior(self.low.expand(batch_shape), self.high.expand(batch_shape))


"""
Part of the following code is adapted from 
https://pytorch.org/docs/stable/_modules/torch/distributions/categorical.html
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
"""

class CategoricalWithValues(Prior):
    arg_constraints = {"probs": constraints.simplex, "logits": constraints.real_vector}
    has_enumerate_support = False
    def __init__(self, values, probs=None, logits=None, validate_args=None):
        """
        distribution of samples from values
        """
        assert values.dim() >= 1
        torch.nn.Module.__init__(self)
        if (not probs is None) and (logits is None):
            assert values.shape[-1] == probs.shape[-1]
            broadcast_shape = torch.broadcast_shapes(values.shape, probs.shape)
            self.probs = ( probs / probs.sum(-1, keepdim=True) ).expand(broadcast_shape)
            self.logits = probs_to_logits(self.probs)
            self._num_events = self.probs.size()[-1]
            batch_shape = broadcast_shape[:-1]
            super().__init__(batch_shape, validate_args=validate_args)
            del self.probs
            del self.logits
            self.register_buffer('probs', ( probs / probs.sum(-1, keepdim=True) ).expand(broadcast_shape) )
            self.register_buffer('logits', probs_to_logits(self.probs) )
        elif (probs is None) and (not logits is None):
            assert values.shape[-1] == logits.shape[-1]
            broadcast_shape = torch.broadcast_shapes(values.shape, logits.shape)
            self.logits = ( logits - logits.logsumexp(dim=-1, keepdim=True) ).expand(broadcast_shape)
            self.probs = logits_to_probs(self.logits)
            self._num_events = self.logits.size()[-1]
            batch_shape = broadcast_shape[:-1]
            super().__init__(batch_shape, validate_args=validate_args)
            del self.probs
            del self.logits
            self.register_buffer('logits', ( logits - logits.logsumexp(dim=-1, keepdim=True) ).expand(broadcast_shape) )
            self.register_buffer('probs', logits_to_probs(self.logits) )
        else:
            raise ValueError("Specify either `probs` or `logits`, and not both.")
        self.register_buffer('values', values.expand(broadcast_shape) )

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        param_shape = batch_shape + torch.Size((self._num_events,))
        return CategoricalWithValues(self.values.expand(param_shape), probs=self.probs.expand(param_shape))

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        raise NotImplementedError

    @property
    def param_shape(self):
        return self.values.size()

    @property
    def mean(self):
        return torch.full(
            self._extended_shape(),
            torch.nan,
            dtype=self.values.dtype,
            device=self.values.device,
        )

    @property
    def mode(self):
        return torch.gather(
            self.values,
            dim=-1,
            index=self.probs.argmax(axis=-1).unsqueeze(-1)
        ).squeeze(-1)

    @property
    def variance(self):
        return torch.full(
            self._extended_shape(),
            torch.nan,
            dtype=self.values.dtype,
            device=self.values.device,
        )

    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        idx = samples_2d.reshape(self._extended_shape(sample_shape)) # [sample_shape, batch_shape]
        # values: [batch_shape, event_shape]
        values = self.values.expand(torch.Size(sample_shape) + self.param_shape)
        return torch.gather(
            values,
            dim=-1,
            index=idx.unsqueeze(-1)
        ).squeeze(-1)

    def log_prob(self, value):
        # value: [addition shape, batch_shape]
        # self.values: [batch_shape, event_shape]
        v = torch.argmin(
            (self.values - value.unsqueeze(-1)).abs(),
            axis=-1
        )
        if self._validate_args:
            self._validate_sample(v)
        v = v.long().unsqueeze(-1)
        v, log_pmf = torch.broadcast_tensors(v, self.logits)
        v = v[..., :1]
        return log_pmf.gather(-1, v).squeeze(-1)

    def entropy(self):
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        return -p_log_p.sum(-1)

    def enumerate_support(self, expand=True):
        raise NotImplementedError

