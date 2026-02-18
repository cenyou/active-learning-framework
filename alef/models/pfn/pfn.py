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

"""
Prior-Data Fitted Networks (PFNs).
Adapted from
github.com/automl/TransformersCanDoBayesianInference
github.com/automl/PFNs

Mueller et al. ICLR 2022. TRANSFORMERS CAN DO BAYESIAN INFERENCE
Appendix E1:
 - x, y are encoded (Linear maps) and summed
 - no positional encoding
 - target predictions: outputs as positioned

"""

import logging
import torch
import torch.nn as nn
from typing import Optional

from alef.models.pfn.embedder import PFNv1Embedder
from alef.models.pfn.head import (
    BarDistribution,
    FullSupportBarDistribution,
    LogitsKnownDistribution,
    get_bucket_borders,
)
from alef.models.pfn.utils import LossAttr

logger = logging.getLogger(__name__)

class PFN(nn.Module):
    """PFN"""

    def __init__(
        self,
        dim_x: int,
        dim_y: int=1,
        d_model: int=128,
        dim_feedforward: int=256,
        nhead: int=4,
        dropout: float=0.0,
        num_layers: int=6,
        head_num_buckets: int=1000,
        head_bucket_samples: Optional[torch.Tensor] = torch.randn([10000]),
        pos_emb_init: bool = False,
    ):
        """
        Args:
            dim_x: Dimension of input x.
            dim_y: Dimension of input y. PFN supports only dim_y=1.
            d_model: embedding size.
            dim_feedforward: Dimension of hidden layers.
            nhead: Number of attention heads.
            dropout: Dropout rate.
            num_layers: Number of transformer encoder layers.
            head_num_buckets: Number of buckets for the head distribution.
            head_bucket_samples: Samples to determine each bucket border. This can be None, for example when we later load bucket borders (torch Module buffer) from trained models.
            pos_emb_init: Whether to use positional encoding initialization for embedding markers.
        """

        assert dim_y==1, f"PFN only supports dim_y=={dim_y}"

        super(PFN, self).__init__() # nn.Module.__init__

        self.embedder = PFNv1Embedder(dim_x, dim_y, d_model, pos_emb_init=pos_emb_init)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, head_num_buckets)
        )

        # Initialize bar distribution
        if head_bucket_samples is None:
            logger.info("Head bucket samples are not provided, using random samples.")
            bucket_borders = get_bucket_borders(head_num_buckets, full_range=(-2.0, 2.0))
            # the borders will be torch Module buffer (can be loaded)
        else:
            bucket_borders = get_bucket_borders(head_num_buckets, ys=head_bucket_samples)
        self.predictor = FullSupportBarDistribution(bucket_borders)

    def create_mask(self, nc: int, nt: int, device: str) -> torch.Tensor:
        """Create attention mask for the transformer encoder."""

        mask = torch.concat([
            torch.concat([
                torch.ones([nc, nc], device=device), # context to context attention
                torch.zeros([nc, nt], device=device) # context to target attention
            ], dim=1),
            torch.concat([
                torch.ones([nt, nc], device=device), # target to context attention
                torch.eye(nt, device=device) # target self attention
            ], dim=1),
        ], dim=0)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

    def encode(self, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        """Encode context and target points."""
        # Embed context
        xc_enc = self.embedder.embed_context(xc, yc)
        # Embed targets (without y values)
        xt_enc = self.embedder.embed_target(xt)

        """
        # Concatenate all embeddings
        encoder_input = torch.cat([xc_enc, xt_enc], dim=1)

        # Pass through transformer
        mask = self.create_mask(xc.shape[-2], xt.shape[-2], device=encoder_input.device)
        out = self.encoder(encoder_input, mask=mask)

        # Return only target encodings
        num_targets = xt.shape[-2]
        return out[:, -num_targets:, :]

        """
        # what we really need is to have target points attend to context points only
        # so we can do this more efficiently by processing each target point separately
        # don't need mask in this case
        B, Nc, d = xc_enc.shape
        Nt = xt_enc.shape[-2]
        #
        xc_enc_expanded = xc_enc.unsqueeze(1).expand(-1, Nt, -1, -1) # [B, Nt, Nc, d_model]
        xt_enc_expanded = xt_enc.unsqueeze(2) # [B, Nt, 1, d_model]
        encoder_input = torch.cat([xc_enc_expanded, xt_enc_expanded], dim=2) # [B, Nt, Nc + 1, d_model]

        out = self.encoder(encoder_input.view(-1, Nc + 1, d), mask=None) # [B * Nt, Nc + 1, d_model]

        out = out.view(B, Nt, Nc + 1, d) # [B, Nt, Nc + 1, d_model]
        return out[:, :, -1, :] # [B, Nt, d_model]

    def forward(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: torch.Tensor,
        reduce_ll: bool = True
    ) -> LossAttr:
        """
        Forward pass through the model.
        
        Args:
            batch: DataAttr containing context and target data.
            reduce_ll: If True, reduce log likelihood to a scalar.
        
        Returns:
            LossAttr containing loss and log likelihood of target data.
        """

        # Encode
        out_encoder = self.encode(xc, yc, xt) # [batch_size, num_target, d_model]

        # Decode to get logits for bar distribution
        logits = self.decoder(out_encoder) # [batch_size, num_target, num_buckets]

        # Compute loss and log likelihood
        return self.predictor(logits, yt[..., 0], reduce_ll=reduce_ll)

    def predict(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        num_samples: int = 50,
        return_samples: bool = False
    ) -> torch.Tensor:
        """Make predictions at target locations."""
        batch_size = xc.shape[0]
        num_target = xt.shape[1]

        # Encode and decode
        out_encoder = self.encode(xc, yc, xt)
        logits = self.decoder(out_encoder)

        if return_samples:
            samples = self.predictor.sample(logits, num_samples=num_samples) # [B, Nt, num_samples]
            return samples.unsqueeze(-1)  # [B, Nt, num_samples, Dy=1]

        return LogitsKnownDistribution(self.predictor, logits)

    def predictive_entropy(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
    ):
        """Compute predictive entropy at target locations."""
        distrib = self.predict(xc, yc, xt, return_samples=False)
        return distrib.entropy() # [B, Nt, 1]

    def sample(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        num_samples: int = 50
    ) -> torch.Tensor:
        """Sample from the model (convenience method).
        
        Args:
            xc: Context inputs [B, Nc, Dx]
            yc: Context outputs [B, Nc, Dy]
            xt: Target inputs [B, Nt, Dx]
            num_samples: Number of samples to generate
            return_samples: If True, return samples; else return distribution
            
        Returns:
            Samples [B, Nt, num_samples, Dy]
        """
        return self.predict(xc, yc, xt, num_samples, return_samples=True)

    def eval_log_likelihood(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate log likelihood at individual target point.
        
        Args:
            xc: Context inputs [B, Nc, Dx]
            yc: Context outputs [B, Nc, Dy]
            xt: Target inputs [B, Nt, Dx]
            yt: Target outputs [B, Nt, Dy]
            
        Returns:
            Samples [B], sum_i log p( [yt]_i | [xt]_i, yc, xc )
        """
        ll = self.forward(xc, yc, xt, yt).log_likelihood.squeeze(-1) # [B, Nt]
        return ll.sum(-1)

if __name__ == "__main__":
    # Simple test
    Dx = 2
    model = PFN(
        dim_x=Dx
    )

    xc = torch.randn(3, 5, Dx)
    yc = torch.randn(3, 5, 1)
    xt = torch.randn(3, 4, Dx)
    yt = torch.randn(3, 4, 1)

    output = model.forward(xc, yc, xt, yt)
    print("Loss:", output.loss)
    print("Log Likelihood:", output.log_likelihood)