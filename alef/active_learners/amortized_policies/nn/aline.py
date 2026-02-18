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

import os
import torch
import random
from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer
from torch.nn import functional as F
from torch.distributions import Normal, Categorical
from typing import Any, List, Tuple, Optional
from attrdictionary import AttrDict
from omegaconf import OmegaConf
import hydra
from hydra import initialize_config_dir, compose

"""
The following code is taken from:
https://github.com/huangdaolang/aline
"""

# ==============================
"""https://github.com/huangdaolang/ALINE/blob/main/model/embedder.py"""
class Embedder(nn.Module):
    """
    Embedder module that processes batches in three different modes:
    - 'theta': For predicting latent variables only
    - 'data': For predicting data only
    - 'mix': Combination of data and theta prediction
    
    The embedder takes a batch with context, query, and target data,
    and produces embeddings in the order: context, query, target.
    """
    
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        n_target_theta: int = 0,
        embedding_type: str = "data",
        **kwargs: Any
    ) -> None:
        """
        Initialize the embedder module
        
        Args:
            dim_x: Dimension of input x 
            dim_y: Dimension of output y
            dim_embedding: Dimension of embedding vectors
            dim_feedforward: Dimension of feedforward layer
            n_target_theta: Number of latent variables (theta) to create theta identifiers
            embedding_type: Type of embedding ('data', 'theta', or 'mix')
        """
        super().__init__()
        
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_embedding = dim_embedding
        self.n_target_theta = n_target_theta
        self.embedding_type = embedding_type
        
        # Create embedders for x and y
        self.x_embedder = nn.Sequential(
            nn.Linear(dim_x, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding)
        )
        
        self.y_embedder = nn.Sequential(
            nn.Linear(dim_y, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_embedding)
        )
        
        # Create theta tokens for theta and mix modes
        if embedding_type in ["theta", "mix"]:
            if self.n_target_theta <= 0:
                raise ValueError("dim_theta must be positive for theta or mix embedding type")
            
            # Create learnable theta tokens (one for each dimension of theta)
            self.theta_tokens = nn.Parameter(torch.randn(self.n_target_theta, dim_embedding))
    
    def forward(self, batch: AttrDict) -> torch.Tensor:
        """
        Process batch and generate embeddings based on the selected mode
        
        Args:
            batch: Batch containing context_x, context_y, query_x, and other tasks
                  depending on the embedding mode
        
        Returns:
            embeddings: Tensor of shape [B, N, dim_embedding] with context, query, 
                      and target embeddings concatenated
        """
        batch_size = batch.context_x.shape[0]
        
        # Extract dimensions
        n_context = batch.context_x.shape[1]
        n_query = batch.query_x.shape[1]

        # Process according to embedding type
        if self.embedding_type == "data":
            embeddings = self._embed_data_mode(batch, batch_size, n_context, n_query)
        elif self.embedding_type == "theta":
            embeddings = self._embed_theta_mode(batch, batch_size, n_context, n_query)
        elif self.embedding_type == "mix":
            embeddings = self._embed_mix_mode(batch, batch_size, n_context, n_query)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

        return embeddings

    def _embed_data_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        """
        Data mode embedding: predict data only

        Args:
            batch: Batch containing context_x, context_y, query_x, and other tasks
                  depending on the embedding mode
            batch_size: Batch size
            n_context: Number of context data
            n_query: Number of query data

        Returns:
            embeddings: Tensor of shape [B, N, dim_embedding] with context, query, 
                      and target embeddings concatenated
        """
        # Embed x
        x_all = torch.cat(
            [batch.context_x, 
             batch.query_x,
             batch.target_x], 
            dim=1
        )
        embeddings = self.x_embedder(x_all)  # [B, n_context+n_query+n_target, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        embeddings[:, :n_context] = embeddings[:, :n_context] + y_embeddings_context

        return embeddings
    
    def _embed_theta_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        """
        Theta mode embedding: predict latent variables

        Args:
            batch: Batch containing context_x, context_y, query_x, and other tasks
                  depending on the embedding mode
            batch_size: Batch size
            n_context: Number of context data
            n_query: Number of query data

        Returns:
            embeddings: Tensor of shape [B, N, dim_embedding] with context, query, 
                      and target embeddings concatenated
        """
        # Embed x (context and query only)
        x_context_query = torch.cat(
            [batch.context_x, 
             batch.query_x], 
            dim=1
        )
        x_embeddings = self.x_embedder(x_context_query)  # [B, n_context+n_query, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        # Add x and y embeddings for context
        context_embeddings = x_embeddings[:, :n_context] + y_embeddings_context
        
        # Extract query embeddings
        query_embeddings = x_embeddings[:, n_context:]
        
        # Create theta tokens (expanded to batch size)
        theta_embeddings = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, dim_theta, dim_embedding]
        
        # Concatenate all embeddings in order: context, query, theta tokens
        embeddings_list = [context_embeddings, query_embeddings, theta_embeddings]

        embeddings = torch.cat(embeddings_list, dim=1)  # [B, N, dim_embedding]
        
        return embeddings
    
    def _embed_mix_mode(self, batch: AttrDict, batch_size: int, n_context: int, n_query: int) -> torch.Tensor:
        """
        Mix mode embedding: combination of data and theta prediction

        Args:
            batch: Batch containing context_x, context_y, query_x, and other tasks
                  depending on the embedding mode
            batch_size: Batch size
            n_context: Number of context data
            n_query: Number of query data

        Returns:
            embeddings: Tensor of shape [B, N, dim_embedding] with context, query, 
                      and target embeddings concatenated
        """
        
        # Embed x (concatenate all x tasks)
        x_all = torch.cat(
            [batch.context_x, 
             batch.query_x,
             batch.target_x], 
            dim=1
        )
        x_embeddings = self.x_embedder(x_all)  # [B, n_context+n_query+n_target, dim_embedding]
        
        # Embed y (only context_y)
        y_embeddings_context = self.y_embedder(batch.context_y)  # [B, n_context, dim_embedding]
        
        # Add x and y embeddings for context
        context_embeddings = x_embeddings[:, :n_context] + y_embeddings_context
        
        # Extract query and target x embeddings
        query_embeddings = x_embeddings[:, n_context:n_context+n_query]
        target_embeddings = x_embeddings[:, n_context+n_query:]
        
        # Simply expand theta tokens to batch size
        theta_embeddings = self.theta_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, dim_theta, dim_embedding]
        
        # Combine target embeddings by concatenation (not addition)
        # Concatenate target_x_embeddings and theta_embeddings
        embeddings_list = [context_embeddings, query_embeddings, target_embeddings, theta_embeddings]
        
        embeddings = torch.cat(embeddings_list, dim=1)  # [B, N, dim_embedding]
        
        return embeddings

# ==============================
"""https://github.com/huangdaolang/ALINE/blob/main/model/encoder.py"""
class EfficientTransformerEncoderLayer(TransformerEncoderLayer):
    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor],
            key_padding_mask: Optional[Tensor],
            is_causal: bool = False,
    ) -> Tensor:
        # Calculate number of context elements and other dimensions
        batch_size, seq_len, embed_dim = x.shape
        num_ctx = torch.sum(attn_mask[0, :] == 0).item()

        # Split into context and non-context parts
        context = x[:, :num_ctx, :]
        non_context = x[:, num_ctx:, :]

        # Process context with self-attention
        context_out = self.self_attn(
            context, context, context,
            attn_mask=None,
            key_padding_mask=key_padding_mask[:, :num_ctx] if key_padding_mask is not None else None,
            need_weights=False,
            is_causal=is_causal,
        )[0]

        # Process non-context with attention to the full sequence based on the mask
        non_context_out = self.self_attn(
            non_context,  # Query: non-context elements
            x,  # Key: full sequence
            x,  # Value: full sequence
            attn_mask=attn_mask[num_ctx:, :],  # Use the relevant part of the attention mask
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]

        # Combine results
        x_out = torch.cat([context_out, non_context_out], dim=1)

        return self.dropout1(x_out)

class Encoder(nn.Module):
    """
    Encoder module that processes batches in three different modes:
    - 'theta': For predicting latent variables only
    - 'data': For predicting data only
    - 'mix': Combination of data and theta prediction
    
    """
    def __init__(
            self,
            dim_embedding,
            dim_feedforward,
            n_head,
            dropout,
            num_layers,
    ):
        """
        Initialize the encoder module
        
        Args:
            dim_embedding: Dimension of embedding vectors
            dim_feedforward: Dimension of feedforward layer
            n_head: Number of attention heads   
            dropout: Dropout rate
            num_layers: Number of layers
        """
        super().__init__()
        # Create the encoder layer
        encoder_layer = EfficientTransformerEncoderLayer(
            dim_embedding, n_head, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

    def create_mask(self, batch):
        """
        Create a mask for the encoder
        
        Args:
            batch: Batch containing context_x, query_x, and target_all

        Returns:
            mask: Mask for the encoder
        """
        # Get base dimensions
        num_context = batch.context_x.shape[1]
        num_query = batch.query_x.shape[1]
        num_target = batch.target_all.shape[1]
        num_all = num_context + num_query + num_target

        # Create mask where all positions initially can't attend to any position
        mask = torch.zeros(num_all, num_all, device=self.device).fill_(float("-inf"))

        query_start = num_context
        query_end = query_start + num_query
        target_start = query_end

        # All positions can attend to context positions
        mask[:, :num_context] = 0.0

        # Query points can attend to target points based on target_mask
        if hasattr(batch, 'target_mask') and batch.target_mask is not None:
            # Get the target mask (same for all batch elements)
            target_mask = batch.target_mask  # Shape: [num_target]

            # Find indices of selected targets
            selected_targets = torch.where(target_mask)[0]

            # Map selected target indices to positions in the full sequence
            target_positions = selected_targets + target_start

            # Enable attention from all queries to all selected targets at once
            mask[query_start:query_end, target_positions] = 0.0
        else:
            # Default behavior: all queries don't attend to any target
            mask[query_start:query_end, target_start:] = float("-inf")

        return mask

    def forward(self, batch, embeddings):
        """
        Forward pass through the encoder
        
        Args:
            batch: Batch containing context_x, query_x, and target_all
            embeddings: Embeddings for the encoder

        Returns:
            out: Output of the encoder
        """
        mask = self.create_mask(batch)
        out = self.encoder(embeddings, mask=mask)
        return out

# ==============================
"""https://github.com/huangdaolang/ALINE/blob/main/model/head.py"""
class AcquisitionHead(nn.Module):
    """
    Acquisition head that predicts the acquisition scores for the query data
    """
    def __init__(self, dim_embedding: int, dim_feedforward: int, time_token: bool, **kwargs: Any) -> None:
        """
        Initialize the acquisition head
        
        Args:
            dim_embedding: Dimension of embedding vectors
            dim_feedforward: Dimension of feedforward layer
            time_token: Whether to include a time token in the embeddings
        """
        super().__init__()
        
        if time_token:
            dim_embedding += 1  # add time token to embedding

        self.predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1),
            nn.Flatten(start_dim=-2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the acquisition head
        
        Args:
            z: Embedding feature tensor [B, N, dim_embedding]
        Returns:
            acquisition_scores: Acquisition scores [B, N]
        """
        return self.predictor(z)

class ValueHead(nn.Module):
    def __init__(self, dim_embedding: int, dim_feedforward: int, **kwargs: Any) -> None:
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1)
        )

        # value for zero context
        self.empty_value = nn.Parameter(torch.zeros(1))

    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Embedding feature tensor [B, t, dim_embedding]
        Returns:
            value: Values [B]
        """
        if z.shape[1] == 0:
            z = self.empty_value.expand(z.shape[0], 1)  # [B, 1]
        else:
            # z = z.detach()                              # detach to prevent gradient flow
            z = self.predictor(z).squeeze(-1)           # [B, t]

        return z.mean(1)

class GMMTargetHead(nn.Module):
    """
    Target head that predicts the posterior distribution for the target data
    """
    def __init__(
        self,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        num_components: int,
        single_head: bool = False,
        std_min: float = 1e-4,
        **kwargs: Any
        ) -> None:
        """
        Initialize the target head
        
        Args:
            dim_y: Dimension of output y
            dim_embedding: Dimension of embedding vectors
            dim_feedforward: Dimension of feedforward layer
            num_components: Number of components in the GMM
            single_head: Whether to use a single head or multiple heads
            std_min: Minimum standard deviation for the GMM
        """
        super().__init__()
        self.dim_embedding = dim_embedding
        self.dim_feedforward = dim_feedforward
        self.dim_y = dim_y
        self.single_head = single_head
        self.num_components = num_components
        self.heads = self.initialize_head(self.dim_embedding, self.dim_feedforward, self.dim_y, self.single_head,
                                          num_components)

        self.std_min = std_min
        # TODO: support multi-output case

    def forward(self, batch: AttrDict, z_target: torch.Tensor) -> AttrDict:
        """
        Forward pass through the target head
        
        Args:
            batch: Batch containing context and target data
            z_target: Embedding tensor [B, N, dim_embedding] where N = n_target
        
        Returns:
            outs: AttrDict containing posterior distribution parameters
        """
        # Iterate over each head to get their outputs
        if self.single_head:
            output = self.heads(z_target)
            if self.num_components == 1:
                raw_mean, raw_std = torch.chunk(output, 2, dim=-1)
                raw_weights = torch.ones_like(raw_mean)
            else:
                raw_mean, raw_std, raw_weights = torch.chunk(output, 3, dim=-1)
        else:
            outputs = [head(z_target) for head in self.heads]  # list of [B, n_target, dim_y * 3] * components
            raw_mean, raw_std, raw_weights = self._map_raw_output(outputs)

        mean = raw_mean
        std = F.softplus(raw_std) + self.std_min
        weights = F.softmax(raw_weights, dim=-1)

        outs = AttrDict()

        # also outputs params
        outs.mixture_means = mean
        outs.mixture_stds = std
        outs.mixture_weights = weights

        return outs
    
    def initialize_head(self,
                        dim_embedding: int,
                        dim_feedforward: int,
                        dim_outcome: int,
                        single_head: bool,
                        num_components: int,
                        ) -> nn.Module:
        """
        Initializes the model with either a single head or multiple heads based on the `single_head` flag.

        Args:
            dim_embedding: The input dimension.
            dim_feedforward: The dimension of the feedforward network.
            dim_outcome: The output dimension.
            single_head: Flag to determine whether to initialize a single head or multiple heads.
            num_components: The number of components if `single_head` is False.

        Returns:
            model: The initialized model head(s).
        """
        if single_head & num_components > 1:
            output_dim = dim_outcome * 3
        else:
            output_dim = dim_outcome * 2

        if single_head:
            model = nn.Sequential(
                nn.Linear(dim_embedding, dim_feedforward),
                nn.ReLU(),
                nn.Linear(dim_feedforward, output_dim),
            )
        else:
            model = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(dim_embedding, dim_feedforward),
                        nn.ReLU(),
                        nn.Linear(dim_feedforward, dim_outcome * 3),
                    )
                    for _ in range(num_components)
                ]
            )
        return model

    @staticmethod
    def compute_ll(value: torch.Tensor, means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Computes log-likelihood loss for a Gaussian mixture model.

        Args:
            value: Value tensor [B, T]
            means: Means tensor [B, T, components]
            stds: Standard deviations tensor [B, T, components]
            weights: Weights tensor [B, T, components]

        Returns:
            log_likelihood: Log-likelihood tensor [B, T]
        """
        components = Normal(means, stds, validate_args=False)
        log_probs = components.log_prob(value)
        weighted_log_probs = log_probs + torch.log(weights)
        return torch.logsumexp(weighted_log_probs, dim=-1)

    @staticmethod
    def _map_raw_output(outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Maps the raw output of the target head to the mean, standard deviation, and weights.

        Args:
            outputs: List of tensors [B, T, dim_y * 3] * components

        Returns:
            raw_mean: Means tensor [B, T, components]
            raw_std: Standard deviations tensor [B, T, components]
            raw_weights: Weights tensor [B, T, components]
        """
        concatenated = torch.stack(outputs).movedim(0, -1).flatten(-2, -1) # [B, T, dim_y * 3 * components]
        raw_mean, raw_std, raw_weights = torch.chunk(concatenated, 3, dim=-1) # 3 x [B, T, components]
        return raw_mean, raw_std, raw_weights

class OutputHead(nn.Module):
    """
    Combined head that processes batches and routes to acquisition and target heads.
    Similar to DPTNP's forward method, it splits input into query and posterior parts.
    """
    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        dim_embedding: int,
        dim_feedforward: int,
        num_components: int = 10,
        single_head: bool = False,
        std_min: float = 1e-4,
        value_head: bool = False,
        time_token: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y

        self.time_token = time_token

        # Acquisition head for design selection
        self.acquisition_head = AcquisitionHead(
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            time_token=time_token,
        )

        # Target head for posterior prediction
        self.target_head = GMMTargetHead(
            dim_y=dim_y,
            dim_embedding=dim_embedding,
            dim_feedforward=dim_feedforward,
            num_components=num_components,
            single_head=single_head,
            std_min=std_min
        )

        # Value head for value prediction
        self.value_head = value_head
        if value_head:
            self.value_head = ValueHead(
                dim_embedding=dim_embedding,
                dim_feedforward=dim_feedforward
            )

    def forward(self, batch: AttrDict, z: torch.Tensor) -> AttrDict:
        """
        Process batch by splitting into context, query and target parts
        
        Args:
            batch: Batch containing context and target tasks
            z: Embedding tensor [B, N, dim_embedding] where N = n_context + n_query + n_target
                Context embeddings are on the left, query in the middle, target on the right
            
        Returns:
            outs: AttrDict containing acquisition and posterior prediction results
        """
        batch_size = z.shape[0]
        
        # Get dimensions from batch
        n_context = batch.context_x.shape[1]
        n_query = batch.query_x.shape[1]
        
        # Extract query and target embeddings
        z_query = z[:, n_context:n_context+n_query]
        z_target = z[:, n_context+n_query:]  # embeddings of target data + target theta
        
        # Use acquisition head for query selection (design point)
        if self.time_token:
            time_info = batch.t.unsqueeze(1).unsqueeze(1).expand(batch_size, n_query, 1)  # [B, n_query, 1]
            z_query_with_time = torch.cat([z_query, time_info], dim=-1)
            zt = self.acquisition_head(z_query_with_time)  # [B, n_query]
        else:
            zt = self.acquisition_head(z_query)  # [B, n_query]
        
        # Select design with the highest probability during inference, sample during training
        if self.training:
            # Choose design with probabilities zt
            m_design = Categorical(zt)
            idx_next = m_design.sample()  # [B]
            log_prob = m_design.log_prob(idx_next)
        else:
            # Choose design with the largest probability
            log_prob, idx_next = torch.max(zt, -1)
            log_prob = torch.log(log_prob)
            
        
        # Get the selected design point
        idx_next = idx_next.unsqueeze(1) # [B, 1]
        
        # Use target head for posterior prediction
        posterior_out = self.target_head(batch, z_target)
        posterior_out_query = self.target_head(batch, z_query)  # For ACE uncertainty sampling baseline
        
        # Combine results
        # Value prediction
        if self.value_head:
            z_context = z[:, :n_context]
            value = self.value_head(z_context)
            outs = AttrDict(
                posterior_out_query=posterior_out_query,
                posterior_out=posterior_out,
                design_out=AttrDict(
                    idx=idx_next,       # [B, 1]
                    log_prob=log_prob,  # [B]
                    zt=zt               # [B, N_query]
                ),
                value=value             # [B]
            )
        else:
            outs = AttrDict(
                posterior_out_query=posterior_out_query,
                posterior_out=posterior_out,
                design_out=AttrDict(
                    idx=idx_next,       # [B, 1]
                    log_prob=log_prob,  # [B]
                    zt=zt               # [B, N_query]
                ),
            )
        return outs

# ==============================
"""https://github.com/huangdaolang/ALINE/blob/main/model/base.py"""
class Aline(nn.Module):
    """
    Base ALINE model that consists of an embedder, encoder, and head to process
    input tasks.

    Attributes:
        embedder (Embedder): An embedder module used to convert input tasks into
            embeddings.
        encoder (Encoder): An encoder module that performs the attention mechanism.
        head (OutputHead): A head module that outputs predictions or computes
            log-likelihoods.
    """

    def __init__(
        self, embedder: Embedder, encoder: Encoder, head: OutputHead
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.head = head

    def forward(
        self,
        batch: AttrDict,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer model, processing the input batch
        of tasks.

        Args:
            batch (AttrDict): A batch of input tasks to be processed.

        Returns:
            torch.Tensor: The output tensor from the head module, which could be
                predictions or log-likelihoods.
        """
        embedding = self.embedder(batch)
        encoding = self.encoder(batch, embedding)

        return self.head(batch, encoding)

# ==============================
"""https://github.com/huangdaolang/ALINE/blob/main/utils/misc.py"""

def load_config_and_model(path, config_name="config.yaml", file_name="aline.pth", 
                          load_type="ckpt", device=None):
    """
    Loads configuration and model from a specified path.
    
    Args:
        path (str): Path to the model directory (relative to project root)
        config_name (str): Name of the config file
        file_name (str): Name of the model file
        load_type (str): "ckpt" or "pth"
        device (str): Device to load model on (if None, uses default)
    
    Returns:
        tuple: (resolved_config, model)
    """

    # Normalize to absolute path based on project root (parent of utils)
    if os.path.isabs(path):
        full_dir = path
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        full_dir = os.path.abspath(os.path.join(project_root, path))

    config_dir = os.path.join(full_dir, ".hydra")

    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Config path not found: {config_dir}")

    # Use absolute config directory to avoid Hydra treating it as relative to module path
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
        resolved_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

        if device is not None:
            torch.set_default_device(device)

        # Instantiate model components
        def manual_instantiate(cls, cfg):
            kwargs = OmegaConf.to_container(cfg, resolve=True)
            kwargs.pop("_target_", None)
            return cls(**kwargs)
        
        embedder = manual_instantiate(Embedder, cfg.embedder)
        encoder = manual_instantiate(Encoder, cfg.encoder)
        head = manual_instantiate(OutputHead, cfg.head)
        model = Aline(embedder, encoder, head)

        # Load model weights using resolved absolute path
        file_path = os.path.join(full_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            if load_type.lower() == "ckpt":
                ckpt = torch.load(file_path, map_location=torch.get_default_device(), weights_only=False)
                
                if isinstance(ckpt, dict) and "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(ckpt)
                    
            elif load_type.lower() == "pth":
                state_dict = torch.load(file_path, map_location=torch.get_default_device(), weights_only=True)
                model.load_state_dict(state_dict)
                
            else:
                raise ValueError(f"Invalid load_type: {load_type}. Must be 'ckpt' or 'pth'")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {file_path}: {str(e)}")
    
    return resolved_cfg, model

def calculate_gmm_variance(mixture_means, mixture_stds, mixture_weights):
    """
    Calculate the uncertainty of GMM predictions for each query point.

    Args:
        mixture_means: [batch_size, n_query, n_components] tensor of means
        mixture_stds: [batch_size, n_query, n_components] tensor of standard deviations
        mixture_weights: [batch_size, n_query, n_components] or [batch_size, n_components] weights

    Returns:
        variance: [batch_size, n_query] tensor of variance
    """
    batch_size, n_query, n_components = mixture_means.shape

    # Adjust weights shape if necessary
    if len(mixture_weights.shape) == 2:  # [batch_size, n_components]
        # Expand to [batch_size, n_query, n_components]
        weights = mixture_weights.unsqueeze(1).expand(batch_size, n_query, n_components)
    else:
        weights = mixture_weights

    # Calculate weighted mean for each query point
    # Sum along component dimension (dim=2)
    weighted_means = torch.sum(weights * mixture_means, dim=2)  # [batch_size, n_query]

    # Calculate variance for GMM: var = Σ weight * (std^2 + (mean - weighted_mean)^2)
    # First, calculate (mean - weighted_mean)^2
    mean_diffs_squared = (mixture_means - weighted_means.unsqueeze(2)) ** 2  # [batch_size, n_query, n_components]

    # Calculate total variance
    var = torch.sum(
        weights * (mixture_stds ** 2 + mean_diffs_squared),
        dim=2
    )  # [batch_size, n_query]

    return var

# =============================
"""https://github.com/huangdaolang/ALINE/blob/main/utils/target_mask.py"""
def create_target_mask(mask_type,
                       embedding_type,
                       n_target_data,
                       n_target_theta,
                       n_selected_targets,
                       predefined_masks,
                       predefined_mask_weights,
                       mask_index,
                       attend_to):
    """
    Create a target mask based on configuration settings.

    This function creates a boolean mask of length n_target, where True values
    indicate targets that should be attended to during inference.

    Args:

        - embedding_type: 'data', 'theta', or 'mix'
        - n_target_data: Number of target data points
        - n_target_theta: Number of target theta parameters
        - mask_type: 'all', 'partial', 'none', 'predefined', or 'split'
        - n_selected_targets: For 'partial' mask_type, number of targets to select
        - predefined_masks: List of predefined masks to choose from
        - mask_index: Index of predefined mask to use (if None, randomly select)
        - attend_to: For 'split' mask_type, whether to attend to 'data' or 'theta'

    Returns:
        torch.Tensor: Boolean tensor of shape [n_target] where True values indicate
                     targets to attend to
    """

    # Total number of targets
    n_target = n_target_data + n_target_theta

    # Initialize mask
    mask = torch.zeros(n_target, dtype=torch.bool)

    # Determine mask based on type
    if mask_type == 'all':
        # Attend to all targets
        mask.fill_(True)

    elif mask_type == 'none':
        # Don't attend to any targets (ACE case)
        mask.fill_(False)

    elif mask_type == 'partial':
        # For data mode: attend to random subset of data
        if embedding_type == 'data':
            # Randomly select n_selected_targets indices
            indices = torch.randperm(n_target)[:n_selected_targets]
            mask[indices] = True

        # For theta mode: attend to random subset of theta
        elif embedding_type == 'theta':
            indices = torch.randperm(n_target)[:n_selected_targets]
            mask[indices] = True

    elif mask_type == 'predefined':
        # Use predefined mask patterns
        if mask_index is not None:
            # Use specific mask by index
            predefined_mask = predefined_masks[mask_index]
        else:
            # Weighted random selection of predefined mask
            if predefined_mask_weights is not None and len(predefined_mask_weights) == len(predefined_masks):
                # Convert weights to probabilities
                weights = torch.tensor(predefined_mask_weights, dtype=torch.float)
                probabilities = weights / weights.sum()
                # Sample according to weights
                index = torch.multinomial(probabilities, 1).item()
                predefined_mask = predefined_masks[index]
            else:
                # Uniform random selection if no weights provided
                predefined_mask = random.choice(predefined_masks)
        # Convert predefined mask to boolean tensor
        for i, should_attend in enumerate(predefined_mask):
            if i < n_target and should_attend:
                mask[i] = True

    elif mask_type == 'split':
        # For mix mode: attend to either all data or all theta
        if embedding_type == 'mix':
            # Decide whether to attend to data or theta
            if attend_to is not None:
                attend_to_data = attend_to == 'data'
            else:
                attend_to_data = random.choice([True, False])

            if attend_to_data:
                # Attend to all data points
                mask[:n_target_data] = True
            else:
                # Attend to all theta parameters
                mask[n_target_data:] = True

            # TODO: Third type which attends to both data and theta
            # TODO: Fourth type which attends to part of the data and part of the theta

    return mask

def select_targets_by_mask(target_results, target_mask):
    """
    Select target results based on the target mask.

    Args:
        target_results (torch.Tensor): Tensor of shape [batch_size, n_target, ...]
        target_mask (torch.Tensor): Boolean tensor of shape [n_target]

    Returns:
        torch.Tensor: Tensor of shape [batch_size, num_selected, ...] containing only
                     the results for selected targets
    """
    # Get the indices of True values in the mask
    selected_indices = torch.where(target_mask)[0]

    # Select these indices from the target results
    selected_results = target_results[:, selected_indices]

    return selected_results

# ==============================
"""https://github.com/huangdaolang/ALINE/blob/main/utils/eval.py"""
def compute_ll(value: torch.Tensor, means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Computes log-likelihood loss for a Gaussian mixture model.
    """
    components = Normal(means, stds, validate_args=False)
    log_probs = components.log_prob(value)
    weighted_log_probs = log_probs + torch.log(weights)
    return torch.logsumexp(weighted_log_probs, dim=-1)

