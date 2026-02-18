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
TNP, PFN embedders, adapted from their original implementations.
TNP implementation: github.com/tung-nd/TNP-pytorch
PFN implementation: github.com/automl/TransformersCanDoBayesianInference
                    github.com/automl/PFNs
"""
import torch
from torch import nn
from .utils import positional_encoding_init

### TNP Embedder
###
def build_mlp(d_in, d_hid, d_out, depth):
    """Build a simple MLP."""
    modules = [nn.Linear(d_in, d_hid), nn.ReLU()]
    for _ in range(depth - 2):
        modules.extend([nn.Linear(d_hid, d_hid), nn.ReLU()])
    modules.append(nn.Linear(d_hid, d_out))
    return nn.Sequential(*modules)



### PFN Embedder
###
class PFNv1Embedder(nn.Module):
    """ PFN embedder with additional context/target markers."""
    
    def __init__(self, dim_x, dim_y, d_model, pos_emb_init: bool = False):
        super().__init__()
        self.marker_lookup = {"target": 0, "context": 1, "buffer": 2}
        self.enc_x = nn.Linear(dim_x, d_model)
        self.enc_y = nn.Linear(dim_y, d_model)
        self.marker_embed = torch.nn.Embedding(3, d_model)

        if pos_emb_init:
            self.marker_embed.weight = positional_encoding_init(3, d_model, 2)

    def _get_marker_embedding(
        self, 
        batch_size: int,
        marker_type: str, 
        device: torch.device
     ) -> torch.Tensor:
        """Get marker embedding for the specified type."""
        marker = self.marker_lookup[marker_type]
        marker_idx = torch.full(
            (batch_size, 1), marker, dtype=torch.long, device=device
        )
        return self.marker_embed(marker_idx)

    def embed_context(self, x, y) -> torch.Tensor:
        """Embed context pairs (xc, yc) with context marker."""
        x_emb = self.enc_x(x)
        y_emb = self.enc_y(y)
        marker_emb = self._get_marker_embedding(x_emb.size(0), "context", x_emb.device)
        return x_emb + y_emb + marker_emb

    def embed_target(self, x) -> torch.Tensor:
        """Embed target inputs (xt) with target marker."""
        x_emb = self.enc_x(x)
        marker_emb = self._get_marker_embedding(x_emb.size(0), "target", x_emb.device)
        return x_emb + marker_emb

    def forward(self, x, y):
        # pfn uses x encoder and y encoder, and then sum the embeddings
        if y is not None:
            return self.enc_x(x) + self.enc_y(y)
        else:
            return self.enc_x(x)
