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

import numpy as np
import torch
from torch import nn
from typing import Sequence, Optional, Union, Tuple

from .aggregators import ImplicitDeepAdaptiveDesign
from .modules.emitter import MLPEmitter
from .modules.selfattention import SelfAttention, DoubleSequencesSelfAttention
from .policies import _NNPolicyHelper

from alef.configs.base_parameters import INPUT_DOMAIN
from alef.active_learners.amortized_policies.global_parameters import MAX_DIMENSION
from alef.enums.active_learner_amortized_policy_enums import DomainWarpperType

# give necessary modules

class TransformerHistoryEncoder(nn.Module):
    # attend sequences, sequences, ..., sequences, then features, features, ..., features
    def __init__(
        self,
        output_dim, # embedding dim so E
        num_feature_attention_layer=2,
        num_sequence_attention_layer=2,
        attend_sequence_first: bool=True,
        name: str=None,
    ):
        super().__init__()
        self.input_dim = 2 # {([x_t]_d, y_t)_t}_d [D, T, 2]
        self.output_dim = output_dim
        self._name = name
        ## layers
        self.layer1 = nn.Linear(2, output_dim)
        self.feature_attention = SelfAttention(
            output_dim,
            output_dim,
            n_attn_layers=num_feature_attention_layer
        )
        self.sequence_attention = SelfAttention(
            output_dim,
            output_dim,
            n_attn_layers=num_sequence_attention_layer
        )
        self._attend_sequence_first = attend_sequence_first

    def _attend_feature(self, X, mask=None):
        """
        :param X: torch.tensor, [..., T, D, encoding_dim]
        :param mask: torch.tensor, [..., D] if given
        :return: torch.tensor, [..., T, D, encoding_dim]
        """
        batch_size, process_size = X.shape[:-2], X.shape[-2:]
        X = X.reshape( (-1, ) + process_size ) # [flatten, D, E]
        if mask is None:
            X = self.feature_attention(X) # [flatten, D, E]
        else:
            mask = mask.unsqueeze(-2).expand( batch_size + (process_size[0],) ) # [..., T, D]
            mask = mask.flatten(0, -3)
            X = self.feature_attention(X, mask=mask) # [flatten, D, E]
        X = X.reshape(batch_size + process_size) # [..., T, D, E]
        return X

    def _attend_sequence(self, X, mask=None):
        """
        :param X: torch.tensor, [..., T, D, encoding_dim]
        :param mask: torch.tensor, [..., T] if given
        :return: torch.tensor, [..., T, D, encoding_dim]
        """
        X = X.transpose(-2, -3) # [..., D, T, E]
        batch_size, process_size = X.shape[:-2], X.shape[-2:]
        X = X.reshape( (-1, ) + process_size ) # [flatten, T, E]
        if mask is None:
            X = self.sequence_attention(X) # [flatten, T, E]
        else:
            mask = mask.unsqueeze(-2).expand( batch_size + (process_size[0],) ) # [..., D, T]
            mask = mask.flatten(0, -3)
            X = self.feature_attention(X, mask=mask) # [flatten, T, E]
        X = X.reshape(batch_size + process_size) # [..., D, T, E]
        X = X.transpose(-2, -3) # [..., T, D, E]
        return X

    def forward(self, X, Y, mask_sequence=None, mask_feature=None, **kwargs):
        """
        :param X: torch.tensor, [..., T, D]
        :param Y: torch.tensor, [..., T]
        :param mask_sequence: torch.tensor, [..., T] if given
        :param mask_feature: torch.tensor, [..., D] if given
        :return: torch.tensor, [..., T, D, encoding_dim]
        """
        inputs = torch.cat([
            X.unsqueeze(-1),
            Y.unsqueeze(-1).expand(X.shape).unsqueeze(-1)
        ], dim=-1) # [..., T, D, 2]
        out = self.layer1(inputs) # [..., T, D, E]
        if self._attend_sequence_first:
            out = self._attend_sequence(out, mask=mask_sequence) # [..., T, D, E]
            out = self._attend_feature(out, mask=mask_feature) # [..., T, D, E]
        else:
            out = self._attend_feature(out, mask=mask_feature) # [..., T, D, E]
            out = self._attend_sequence(out, mask=mask_sequence) # [..., T, D, E]
        return out

class StripedTransformerHistoryEncoder(nn.Module):
    # attend features, sequences, features, layers, ...
    def __init__(
        self,
        output_dim, # embedding dim so E
        num_self_attention_layer=2,
        attend_sequence_first: bool=False,
        name: str=None,
    ):
        super().__init__()
        self.input_dim = 2 # {([x_t]_d, y_t)_t}_d [D, T, 2]
        self.output_dim = output_dim
        self._name = name
        ## layers
        self.layer1 = nn.Linear(2, output_dim)
        self.attention_layer = DoubleSequencesSelfAttention(
            output_dim,
            output_dim,
            n_attn_layers=num_self_attention_layer
        )
        self._attend_sequence_first = attend_sequence_first

    def forward(self, X, Y, mask_sequence=None, mask_feature=None, **kwargs):
        """
        :param X: torch.tensor, [..., T, D]
        :param Y: torch.tensor, [..., T]
        :param mask_sequence: torch.tensor, [..., T] if given
        :param mask_feature: torch.tensor, [..., D] if given
        :return: torch.tensor, [..., T, D, encoding_dim]
        """
        inputs = torch.cat([
            X.unsqueeze(-1),
            Y.unsqueeze(-1).expand(X.shape).unsqueeze(-1)
        ], dim=-1) # [..., T, D, 2]
        out = self.layer1(inputs) # [..., T, D, E]
        batch_size, process_size = out.shape[:-3], out.shape[-3:]
        out = out.reshape( (-1, ) + process_size ) # [flatten, T, D, E]
        if self._attend_sequence_first:
            out = out.transpose(-2, -3) # [..., D, T, E]
            out = self.attention_layer(
                out,
                mask1= None if mask_sequence is None else mask_sequence.flatten(0, -2),
                mask2= None if mask_feature is None else mask_feature.flatten(0, -2)
            )
            out = out.transpose(-2, -3) # [..., T, D, E]
        else:
            out = self.attention_layer(
                out,
                mask1= None if mask_feature is None else mask_feature.flatten(0, -2),
                mask2= None if mask_sequence is None else mask_sequence.flatten(0, -2)
            ) # [..., T, D, E]
        out = out.reshape(batch_size + process_size) # [..., T, D, E]
        return out

class MLPEmitterFlexDim(MLPEmitter):
    def __init__(
        self,
        input_dim, # embedding dim so E
        hidden_dim,
        output_dim_axis: int=-2, # which dim is output
        activation=nn.ReLU(),
        output_activation=nn.Identity(),
        batch_norm=False,
        name=None,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=(1,),
            activation=activation,
            output_activation=output_activation,
            batch_norm=batch_norm,
            name=name
        )
        self._output_dim_axis = output_dim_axis
    def forward(self, inputs, **kwargs):
        # inputs: [..., D, E]
        out = super().forward(inputs, **kwargs) # [..., D, 1]
        return out.transpose(-1, self._output_dim_axis)

###
# now let's work on the policies

class PermutationInvariantFlexDimBasePolicy(ImplicitDeepAdaptiveDesign):
    # assumes the encoder map x [..., T, D], y [..., T] into [..., T, D, encoding_dim]
    def __init__(
        self, encoder_network, emission_network, empty_value
    ):
        super().__init__(
            encoder_network=encoder_network,
            emission_network=emission_network,
            empty_value=empty_value,
        )
        #self.flexible_dimension.data = torch.tensor(True, dtype=bool, device=self.flexible_dimension.device)
        self.flexible_dimension = True

    def sum_history_encodings(self, encoder_module, X, Y, mask_sequence=None, mask_feature=None):
        """
        :param encoder_module: (X, Y) [*batch, T, D]x[*batch, T] -> [*batch, D, E]
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        :param mask_feature: optional sequence mask, size [*batch, D] if given
        
        :return: whole sequence embedding [*batch, D, E]
        """
        stacked = encoder_module(X, Y, mask_sequence=mask_sequence, mask_feature=mask_feature) # [..., T, D, encoding_dim]
        # sum-pool the resulting encodings across t (dim=-3)
        stacked = stacked if mask_sequence is None else stacked * mask_sequence[..., None, None]
        sum_encoding = stacked.sum(dim=-3) # [..., D, encoding_dim]
        return sum_encoding


class ContinuousGPFlexDimPolicy(PermutationInvariantFlexDimBasePolicy, _NNPolicyHelper):
    def __init__(
        self,
        *,
        encoding_dim: int,
        num_self_attention_layer: int=2,
        attend_sequence_first: bool=True,
        hidden_dim_emitter: Union[int, Sequence[int]],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.ReLU(),
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        """
        history_encoder = TransformerHistoryEncoder(
            encoding_dim,
            num_feature_attention_layer = num_self_attention_layer,
            num_sequence_attention_layer = num_self_attention_layer,
            attend_sequence_first=attend_sequence_first,
            name="policy_history_encoder"
        )
        """
        history_encoder = StripedTransformerHistoryEncoder(
            encoding_dim,
            num_self_attention_layer=num_self_attention_layer,
            attend_sequence_first=attend_sequence_first,
            name="policy_history_encoder"
        )
        design_emitter = MLPEmitterFlexDim(
            self._get_emitter_input_dim(encoding_dim),
            hidden_dim_emitter,
            output_dim_axis=-2,
            activation=activation,
            output_activation=self._return_domain_warpper(domain_warpper, input_domain),
            name="policy_design_emitter",
        )
        empty_value = torch.zeros((1, MAX_DIMENSION)) # (1, max_query_dim)
        # Design net: takes pairs *[(x_i, y_i), ...] as input
        PermutationInvariantFlexDimBasePolicy.__init__(
            self,
            history_encoder,
            design_emitter,
            empty_value=empty_value,
        )
        self.register_buffer("input_domain", torch.tensor(input_domain, dtype=torch.float))

    def empty_forward(self):
        a, b = self.input_domain
        return torch.rand_like(self.empty_value) * (b-a) + a

    def forward(
        self,
        X: torch.Tensor=None,
        Y: torch.Tensor=None,
        mask_sequence: torch.Tensor=None,
        mask_feature: torch.Tensor=None,
        **kwargs
    ):
        """
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        :param mask_feature: optional sequence mask, size [*batch, D] if given
        
        :return: new x
        """
        if X is None or Y is None:
            output = self.empty_forward()
        else:
            sum_encoding = self.sum_history_encodings(
                self.encoder,
                X, Y,
                mask_sequence=mask_sequence,
                mask_feature=mask_feature
            )# [D, encoding_dim] or [*batch_size, D, encoding_dim]
            output = self.emitter(sum_encoding)
        return output

class ContinuousGPFlexDimBudgetParsedPolicy(ContinuousGPFlexDimPolicy):
    def __init__(
        self,
        *,
        encoding_dim: int,
        hidden_dim_budget_encoder: Optional[int] = None,
        num_self_attention_layer: int=2,
        attend_sequence_first: bool=True,
        hidden_dim_emitter: Union[int, Sequence[int]],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.ReLU(),
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        super().__init__(
            encoding_dim = encoding_dim,
            num_self_attention_layer = num_self_attention_layer,
            attend_sequence_first = attend_sequence_first,
            hidden_dim_emitter = hidden_dim_emitter,
            input_domain = input_domain,
            activation = activation,
            domain_warpper = domain_warpper,
            **kwargs
        )
        self.forward_with_budget.data = torch.tensor(True, dtype=bool, device=self.forward_with_budget.device)
        if hidden_dim_budget_encoder is None:
            hidden_dim_budget_encoder = hidden_dim_emitter
        self.budget_encoder = self._get_budget_encoder( 1, hidden_dim_budget_encoder, encoding_dim, activation=activation )

    def _get_emitter_input_dim(self, encoding_dim):
        return 2*encoding_dim

    def forward(
        self,
        T,
        X: torch.Tensor=None,
        Y: torch.Tensor=None,
        mask_sequence: torch.Tensor=None,
        mask_feature: torch.Tensor=None,
        **kwargs
    ):
        """
        :param T: size [*batch, 1], how many points will we be querying
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        :param mask_feature: optional sequence mask, size [*batch, D] if given
        
        :return: new x
        """
        if X is None or Y is None:
            output = self.empty_forward()
        else:
            sum_encoding = self.sum_history_encodings(
                self.encoder,
                X, Y,
                mask_sequence=mask_sequence,
                mask_feature=mask_feature
            )# [D, encoding_dim] or [*batch_size, D, encoding_dim]
            budget_encoding = self.budget_encoder(T)# [encoding_dim] or [*batch_size, encoding_dim]
            budget_encoding = budget_encoding.unsqueeze(-2).expand(sum_encoding.shape)# [D, encoding_dim] or [*batch_size, D, encoding_dim]
            output = self.emitter(
                torch.cat([sum_encoding, budget_encoding], dim=-1)
            )
        return output

class SafetyAwareContinuousGPFlexDimPolicy(PermutationInvariantFlexDimBasePolicy, _NNPolicyHelper):
    def __init__(
        self,
        *,
        encoding_dim: int,
        num_self_attention_layer: int=2,
        attend_sequence_first: bool=True,
        hidden_dim_emitter: Union[int, Sequence[int]],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.ReLU(),
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        """
        history_encoder = TransformerHistoryEncoder(
            encoding_dim,
            num_feature_attention_layer = num_self_attention_layer,
            num_sequence_attention_layer = num_self_attention_layer,
            attend_sequence_first=attend_sequence_first,
            name="policy_history_encoder"
        )
        safety_history_encoder = TransformerHistoryEncoder(
            encoding_dim,
            num_feature_attention_layer = num_self_attention_layer,
            num_sequence_attention_layer = num_self_attention_layer,
            attend_sequence_first=attend_sequence_first,
            name="policy_safety_history_encoder"
        )
        """
        history_encoder = StripedTransformerHistoryEncoder(
            encoding_dim,
            num_self_attention_layer=num_self_attention_layer,
            attend_sequence_first=attend_sequence_first,
            name="policy_history_encoder"
        )
        safety_history_encoder = StripedTransformerHistoryEncoder(
            encoding_dim,
            num_self_attention_layer=num_self_attention_layer,
            attend_sequence_first=attend_sequence_first,
            name="policy_safety_history_encoder"
        )

        design_emitter = MLPEmitterFlexDim(
            self._get_emitter_input_dim(encoding_dim),
            hidden_dim_emitter,
            output_dim_axis=-2,
            activation=activation,
            output_activation=self._return_domain_warpper(domain_warpper, input_domain),
            name="policy_design_emitter",
        )
        empty_value = torch.zeros((1, MAX_DIMENSION)) # (1, max_query_dim)

        PermutationInvariantFlexDimBasePolicy.__init__( # still lack safety encoder part
            self,
            history_encoder,
            design_emitter,
            empty_value=empty_value,
        )
        # add safety encoder now
        self.safety_encoder = safety_history_encoder
        #
        self.register_buffer("input_domain", torch.tensor(input_domain, dtype=torch.float))

    def _get_emitter_input_dim(self, encoding_dim):
        return 2*encoding_dim

    def empty_forward(self):
        a, b = self.input_domain
        return torch.rand_like(self.empty_value) * (b-a) + a

    def forward(
        self,
        X: torch.Tensor=None,
        Y: torch.Tensor=None,
        Z: torch.Tensor=None,
        mask_sequence: torch.Tensor=None,
        mask_feature: torch.Tensor=None,
        **kwargs
    ):
        """
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param Z: safety observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        :param mask_feature: optional sequence mask, size [*batch, D] if given
        
        :return: new x
        """
        if X is None or Y is None or Z is None:
            output = self.empty_forward()
        else:
            observation_sum_encoding = self.sum_history_encodings(self.encoder, X, Y, mask_sequence=mask_sequence, mask_feature=mask_feature)
            safety_observation_sum_encoding = self.sum_history_encodings(self.safety_encoder, X, Z, mask_sequence=mask_sequence, mask_feature=mask_feature)
            output = self.emitter(
                torch.cat([observation_sum_encoding, safety_observation_sum_encoding], dim=-1)
            )
        return output

class SafetyAwareContinuousGPFlexDimBudgetParsedPolicy(SafetyAwareContinuousGPFlexDimPolicy):
    def __init__(
        self,
        *,
        encoding_dim: int,
        hidden_dim_budget_encoder: Optional[int] = None,
        num_self_attention_layer: int=2,
        attend_sequence_first: bool=True,
        hidden_dim_emitter: Sequence[int],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.ReLU(),
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        super().__init__(
            encoding_dim = encoding_dim,
            num_self_attention_layer = num_self_attention_layer,
            attend_sequence_first = attend_sequence_first,
            hidden_dim_emitter = hidden_dim_emitter,
            input_domain = input_domain,
            activation = activation,
            domain_warpper = domain_warpper,
            **kwargs
        )
        self.forward_with_budget.data = torch.tensor(True, dtype=bool, device=self.forward_with_budget.device)
        if hidden_dim_budget_encoder is None:
            hidden_dim_budget_encoder = hidden_dim_emitter
        self.budget_encoder = self._get_budget_encoder( 1, hidden_dim_budget_encoder, encoding_dim, activation=activation )

    def _get_emitter_input_dim(self, encoding_dim):
        return 3*encoding_dim

    def forward(
        self,
        T,
        X: torch.Tensor=None,
        Y: torch.Tensor=None,
        Z: torch.Tensor=None,
        mask_sequence: torch.Tensor=None,
        mask_feature: torch.Tensor=None,
        **kwargs
    ):
        """
        :param T: size [*batch, 1], how many points will we be querying
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param Z: safety observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        :param mask_feature: optional sequence mask, size [*batch, D] if given
        
        :return: new x
        """
        if X is None or Y is None or Z is None:
            output = self.empty_forward()
        else:
            observation_sum_encoding = self.sum_history_encodings(self.encoder, X, Y, mask_sequence=mask_sequence, mask_feature=mask_feature)# [D, encoding_dim] or [*batch_size, D, encoding_dim]
            safety_observation_sum_encoding = self.sum_history_encodings(self.safety_encoder, X, Z, mask_sequence=mask_sequence, mask_feature=mask_feature)# [D, encoding_dim] or [*batch_size, D, encoding_dim]
            budget_encoding = self.budget_encoder(T)# [encoding_dim] or [*batch_size, encoding_dim]
            budget_encoding = budget_encoding.unsqueeze(-2).expand(observation_sum_encoding.shape)# [D, encoding_dim] or [*batch_size, D, encoding_dim]
            output = self.emitter(
                torch.cat([
                    observation_sum_encoding,
                    safety_observation_sum_encoding,
                    budget_encoding
                ], dim=-1)
            )
        return output
