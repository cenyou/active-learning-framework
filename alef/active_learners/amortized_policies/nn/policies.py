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
from torch import nn
from typing import Sequence, Optional, Union, Tuple

from .aggregators import PermutationInvariantImplicitDAD
from .modules.essential import GeneralizedSigmoid, GeneralizedTanh
from .modules.data_encoder import MLPDataEncoder
from .modules.selfattention import MLP, SelfAttention
from .modules.emitter import MLPEmitter
from alef.configs.base_parameters import INPUT_DOMAIN
from alef.enums.active_learner_amortized_policy_enums import DomainWarpperType

__all__ = [
    'ContinuousGPPolicy',
    'ContinuousGPBudgetParsedPolicy',
    'SafetyAwareContinuousGPPolicy',
    'SafetyAwareContinuousGPBudgetParsedPolicy',
]

class _NNPolicyHelper:
    def _return_domain_warpper(
        self,
        domain_warpper: DomainWarpperType,
        input_domain: Tuple[Union[int, float], Union[int, float]]
    ):
        if domain_warpper==DomainWarpperType.SIGMOID:
            return GeneralizedSigmoid(torch.tensor(input_domain, dtype=torch.float))
        elif domain_warpper==DomainWarpperType.TANH:
            return GeneralizedTanh(torch.tensor(input_domain, dtype=torch.float))
        else:
            raise ValueError

    def _get_data_encoder(self, input_dim, observation_dim, hidden_dim, output_dim, activation, name = "policy_history_encoder"):
        return MLPDataEncoder(
            input_dim = (1, input_dim),
            observation_dim = observation_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim,
            activation = activation,
            output_activation = nn.Identity(),
            name = name,
        )

    def _get_budget_encoder(self, input_dim, hidden_dim, output_dim, activation, name = "policy_budget_encoder"):
        if hasattr(hidden_dim, '__len__'):
            try:
                h_size = hidden_dim[0]
                n_h_layers = len(hidden_dim)
            except: # this happen if we don't assign hidden_dim
                h_size = 32
                n_h_layers = 1
        else:
            h_size = hidden_dim
            n_h_layers = 1
        # we use 1 hidden layer, no batch_norm, no dropout, no residual
        # in this case MLP, MLPHistoryEncoder, MLPEmitter are pretty much the same
        #   these implementation only deals with input and output dims differently
        return MLP(
            input_dim,
            output_dim,
            h_size,
            n_h_layers,
            activation=activation
        )

    def _get_emitter(self, input_dim, hidden_dim, output_dim, activation, domain_warpper, input_domain, name = "policy_design_emitter"):
        return MLPEmitter(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim,
            activation = activation,
            output_activation = self._return_domain_warpper(domain_warpper, input_domain),
            name = name,
        )

    def _get_emitter_input_dim(self, encoding_dim: int):
        """
        :param encoding_dim: the output dim of each data point embedding
        
        :return: the dimension emitter takes, this depends on the NN, e.g. how we pass safety embedding etc
        """
        return encoding_dim


class ContinuousGPPolicy(PermutationInvariantImplicitDAD, _NNPolicyHelper):
    def __init__(
        self,
        input_dim: int,
        observation_dim: int,
        *,
        hidden_dim_encoder: Union[int, Sequence[int]],
        encoding_dim: int,
        hidden_dim_emitter: Union[int, Sequence[int]],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.Softplus(),
        num_self_attention_layer: int=2,
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        data_encoder = self._get_data_encoder(
            input_dim,
            observation_dim,
            hidden_dim_encoder,
            encoding_dim,
            activation=activation
        )
        design_emitter = self._get_emitter(
            self._get_emitter_input_dim(encoding_dim),
            hidden_dim_emitter,
            (1, input_dim),
            activation=activation,
            domain_warpper=domain_warpper,
            input_domain=input_domain
        )
        empty_value = torch.zeros((1, input_dim))
        # Design net: takes pairs *[(x_i, y_i), ...] as input
        PermutationInvariantImplicitDAD.__init__(
            self,
            data_encoder,
            design_emitter,
            empty_value=empty_value,
            self_attention_layer=SelfAttention(
                encoding_dim,
                encoding_dim,
                n_attn_layers=num_self_attention_layer
            ) if num_self_attention_layer > 0 else None,
        )
        # this self_attention_layer is transformer encoder without positional encoding
        # (see Attention is all you need, fig 1)
        self.register_buffer("input_domain", torch.tensor(input_domain, dtype=torch.float))

    def empty_forward(self):
        a, b = self.input_domain
        return torch.rand_like(self.empty_value) * (b-a) + a

class ContinuousGPBudgetParsedPolicy(ContinuousGPPolicy):
    def __init__(
        self,
        input_dim: int,
        observation_dim: int,
        *,
        hidden_dim_encoder: Union[int, Sequence[int]],
        encoding_dim: int,
        hidden_dim_emitter: Union[int, Sequence[int]],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.Softplus(),
        num_self_attention_layer: int=2,
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        super().__init__(
            input_dim, observation_dim,
            hidden_dim_encoder = hidden_dim_encoder,
            encoding_dim = encoding_dim,
            hidden_dim_emitter = hidden_dim_emitter,
            input_domain = input_domain,
            activation = activation,
            num_self_attention_layer = num_self_attention_layer,
            domain_warpper = domain_warpper,
            **kwargs
        )
        self.forward_with_budget.data = torch.tensor(True, dtype=bool, device=self.forward_with_budget.device)
        self.budget_encoder = self._get_budget_encoder(1, hidden_dim_encoder, encoding_dim, activation=activation)

    def _get_emitter_input_dim(self, encoding_dim):
        return 2*encoding_dim

    def forward(self, T, X: torch.Tensor=None, Y: torch.Tensor=None, mask_sequence: torch.Tensor=None, **kwargs):
        """
        :param T: size [*batch, 1], how many points will we be querying
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        
        :return: new x
        """
        if X is None or Y is None:
            output = self.empty_forward()
        else:
            sum_encoding = self.sum_history_encodings(
                self.encoder,
                self.selfattention_layer,
                X, Y,
                mask_sequence=mask_sequence
            )# [encoding_dim] or [*batch_size, encoding_dim]
            budget_encoding = self.budget_encoder(T)# [encoding_dim] or [*batch_size, encoding_dim]
            output = self.emitter(
                torch.cat([sum_encoding, budget_encoding], dim=-1)
            )
        return output


class SafetyAwareContinuousGPPolicyV1(PermutationInvariantImplicitDAD, _NNPolicyHelper):
    def __init__(
        self,
        input_dim: int,
        observation_dim: int,
        safety_dim: int,
        *,
        hidden_dim_encoder: Sequence[int],
        encoding_dim: int,
        hidden_dim_emitter: Sequence[int],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.Softplus(),
        num_self_attention_layer: int=2,
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        data_encoder = self._get_data_encoder(
            input_dim,
            observation_dim + safety_dim,
            hidden_dim_encoder,
            encoding_dim,
            activation=activation
        )

        design_emitter = self._get_emitter(
            self._get_emitter_input_dim(encoding_dim),
            hidden_dim_emitter,
            (1, input_dim),
            activation=activation,
            domain_warpper=domain_warpper,
            input_domain=input_domain
        )
        empty_value = torch.zeros((1, input_dim))

        PermutationInvariantImplicitDAD.__init__(
            self,
            data_encoder,
            design_emitter,
            empty_value=empty_value,
            self_attention_layer=SelfAttention(
                encoding_dim,
                encoding_dim,
                n_attn_layers=num_self_attention_layer
            ) if num_self_attention_layer > 0 else None,
        )
        # this self_attention_layer is transformer encoder without positional encoding
        # (see Attention is all you need, fig 1)
        ###
        self.register_buffer("input_domain", torch.tensor(input_domain, dtype=torch.float))

    def empty_forward(self):
        a, b = self.input_domain
        return torch.rand_like(self.empty_value) * (b-a) + a

    def sum_history_encodings(self, encoder_module, self_attention_layer, X, Y, Z, mask_sequence=None):
        assert X.shape[:-1] == Y.shape == Z.shape
        YZ = torch.cat([Y.unsqueeze(-1), Z.unsqueeze(-1)], dim=-1) # [*batch, T, 2]
        stacked = torch.stack(
            [
                encoder_module(X[..., i, None, :], YZ[..., i, :]) # [*batch, 1, D], [*batch, 2] -> [*batch, 1, E]
                for i in range(X.size(-2))
            ],
            dim=-2,
        ) # [*batch, T, E]
        # apply attention (or identity if attention=None)
        batch_size, stack_size = stacked.shape[:-2], stacked.shape[-2:]

        stacked = stacked.reshape( (-1, ) + stack_size ) # [batch_size_flatten, T, E]
        # mask is [*batch_size, T]
        stacked = self_attention_layer( stacked, mask= None if mask_sequence is None else mask_sequence.flatten(0, -2) )
        stacked = stacked.reshape(batch_size + stack_size) # [*batch, T, E]
        # sum-pool the resulting encodings across t (dim=-2)
        stacked = stacked if mask_sequence is None else stacked * mask_sequence.unsqueeze(-1)
        sum_encoding = stacked.sum(dim=-2)

        return sum_encoding

    def forward(self, X: torch.Tensor=None, Y: torch.Tensor=None, Z: torch.Tensor=None, mask_sequence: torch.Tensor=None, **kwargs):
        """
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param Z: safety observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        
        :return: new x
        """
        if X is None or Y is None or Z is None:
            output = self.empty_forward()
        else:
            sum_encoding = self.sum_history_encodings(self.encoder, self.selfattention_layer, X, Y, Z, mask_sequence=mask_sequence)
            output = self.emitter(sum_encoding)
        return output

class SafetyAwareContinuousGPBudgetParsedPolicyV1(SafetyAwareContinuousGPPolicyV1):
    def __init__(
        self,
        input_dim: int,
        observation_dim: int,
        safety_dim: int,
        *,
        hidden_dim_encoder: Sequence[int],
        encoding_dim: int,
        hidden_dim_emitter: Sequence[int],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.Softplus(),
        num_self_attention_layer: int=2,
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        super().__init__(
            input_dim, observation_dim, safety_dim,
            hidden_dim_encoder = hidden_dim_encoder,
            encoding_dim = encoding_dim,
            hidden_dim_emitter = hidden_dim_emitter,
            input_domain = input_domain,
            activation = activation,
            num_self_attention_layer = num_self_attention_layer,
            domain_warpper = domain_warpper,
            **kwargs
        )
        self.forward_with_budget.data = torch.tensor(True, dtype=bool, device=self.forward_with_budget.device)
        self.budget_encoder = self._get_budget_encoder(1, hidden_dim_encoder, encoding_dim, activation=activation)

    def _get_emitter_input_dim(self, encoding_dim):
        return 2*encoding_dim

    def forward(self, T, X: torch.Tensor=None, Y: torch.Tensor=None, Z: torch.Tensor=None, mask_sequence: torch.Tensor=None, **kwargs):
        """
        :param T: size [*batch, 1], how many points will we be querying
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param Z: safety observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        
        :return: new x
        """
        if X is None or Y is None or Z is None:
            output = self.empty_forward()
        else:
            sum_encoding = sum_encoding = self.sum_history_encodings(self.encoder, self.selfattention_layer, X, Y, Z, mask_sequence=mask_sequence)
            budget_encoding = self.budget_encoder(T)# [encoding_dim] or [*batch_size, encoding_dim]
            output = self.emitter(
                torch.cat([sum_encoding, budget_encoding], dim=-1)
            )
        return output


class SafetyAwareContinuousGPPolicyV2(PermutationInvariantImplicitDAD, _NNPolicyHelper):
    def __init__(
        self,
        input_dim: int,
        observation_dim: int,
        safety_dim: int,
        *,
        hidden_dim_encoder: Union[int, Sequence[int]],
        encoding_dim: int,
        hidden_dim_emitter: Union[int, Sequence[int]],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.Softplus(),
        num_self_attention_layer: int=2,
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        data_encoder = self._get_data_encoder(
            input_dim,
            observation_dim,
            hidden_dim_encoder,
            encoding_dim,
            activation=activation
        )
        # the super().__init__ is for non-safe policy, we have to add safety by ourselves
        safety_data_encoder = self._get_data_encoder(
            input_dim,
            safety_dim,
            hidden_dim_encoder,
            encoding_dim,
            activation=activation,
            name="policy_safety_history_encoder"
        )

        design_emitter = self._get_emitter(
            self._get_emitter_input_dim(encoding_dim),
            hidden_dim_emitter,
            (1, input_dim),
            activation=activation,
            domain_warpper=domain_warpper,
            input_domain=input_domain
        )
        empty_value = torch.zeros((1, input_dim))

        super().__init__( # still lack safety encoder part
            data_encoder,
            design_emitter,
            empty_value=empty_value,
            self_attention_layer=SelfAttention(
                encoding_dim,
                encoding_dim,
                n_attn_layers=num_self_attention_layer
            ) if num_self_attention_layer > 0 else None,
        )
        # this self_attention_layer is transformer encoder without positional encoding
        # (see Attention is all you need, fig 1)
        ###
        # add safety encoder now
        self.safety_encoder = safety_data_encoder
        self.safety_selfattention_layer = (
            SelfAttention(
                encoding_dim,
                encoding_dim,
                n_attn_layers=num_self_attention_layer
            ) if num_self_attention_layer > 0 else nn.Identity()
        )
        #
        self.register_buffer("input_domain", torch.tensor(input_domain, dtype=torch.float))

    def _get_emitter_input_dim(self, encoding_dim):
        return 2*encoding_dim

    def empty_forward(self):
        a, b = self.input_domain
        return torch.rand_like(self.empty_value) * (b-a) + a

    def forward(self, X: torch.Tensor=None, Y: torch.Tensor=None, Z: torch.Tensor=None, mask_sequence: torch.Tensor=None, **kwargs):
        """
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param Z: safety observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        
        :return: new x
        """
        if X is None or Y is None or Z is None:
            output = self.empty_forward()
        else:
            observation_sum_encoding = self.sum_history_encodings(self.encoder, self.selfattention_layer, X, Y, mask_sequence=mask_sequence)
            safety_observation_sum_encoding = self.sum_history_encodings(self.safety_encoder, self.safety_selfattention_layer, X, Z, mask_sequence=mask_sequence)
            output = self.emitter(
                torch.cat([observation_sum_encoding, safety_observation_sum_encoding], dim=-1)
            )
        return output

class SafetyAwareContinuousGPBudgetParsedPolicyV2(SafetyAwareContinuousGPPolicyV2):
    def __init__(
        self,
        input_dim: int,
        observation_dim: int,
        safety_dim: int,
        *,
        hidden_dim_encoder: Sequence[int],
        encoding_dim: int,
        hidden_dim_emitter: Sequence[int],
        input_domain: Tuple[Union[int, float], Union[int, float]] = INPUT_DOMAIN,
        activation: nn.Module=nn.Softplus(),
        num_self_attention_layer: int=2,
        domain_warpper: DomainWarpperType=DomainWarpperType.TANH,
        **kwargs,
    ):
        super().__init__(
            input_dim, observation_dim, safety_dim,
            hidden_dim_encoder = hidden_dim_encoder,
            encoding_dim = encoding_dim,
            hidden_dim_emitter = hidden_dim_emitter,
            input_domain = input_domain,
            activation = activation,
            num_self_attention_layer = num_self_attention_layer,
            domain_warpper = domain_warpper,
            **kwargs
        )
        self.forward_with_budget.data = torch.tensor(True, dtype=bool, device=self.forward_with_budget.device)
        self.budget_encoder = self._get_budget_encoder(1, hidden_dim_encoder, encoding_dim, activation=activation)

    def _get_emitter_input_dim(self, encoding_dim):
        return 3*encoding_dim

    def forward(self, T, X: torch.Tensor=None, Y: torch.Tensor=None, Z: torch.Tensor=None, mask_sequence: torch.Tensor=None, **kwargs):
        """
        :param T: size [*batch, 1], how many points will we be querying
        :param X: designs, size [*batch, T, D], if None then return empty design
        :param Y: observations, size [*batch, T], if None then return empty design
        :param Z: safety observations, size [*batch, T], if None then return empty design
        :param mask_sequence: optional sequence mask, size [*batch, T] if given
        
        :return: new x
        """
        if X is None or Y is None or Z is None:
            output = self.empty_forward()
        else:
            observation_sum_encoding = self.sum_history_encodings(self.encoder, self.selfattention_layer, X, Y, mask_sequence=mask_sequence)
            safety_observation_sum_encoding = self.sum_history_encodings(self.safety_encoder, self.safety_selfattention_layer, X, Z, mask_sequence=mask_sequence)
            budget_encoding = self.budget_encoder(T)# [encoding_dim] or [*batch_size, encoding_dim]
            output = self.emitter(
                torch.cat([
                    observation_sum_encoding,
                    safety_observation_sum_encoding,
                    budget_encoding
                ], dim=-1)
            )
        return output


SafetyAwareContinuousGPPolicy = SafetyAwareContinuousGPPolicyV2
SafetyAwareContinuousGPBudgetParsedPolicy = SafetyAwareContinuousGPBudgetParsedPolicyV2
