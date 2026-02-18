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
from typing import Optional


def positional_encoding_init(seq_len, d, n):
    """Initialize positional encoding matrix using sinusoidal patterns."""
    P = torch.zeros((seq_len, d))
    for k in range(seq_len):
        for i in torch.arange(int(d / 2)):
            period = ((d / 2) / (2 * torch.pi)) / (k + 1)
            phase = 0  # (2*torch.pi*(d/2))* k/(seq_len+1)
            P[k, 2 * i] = torch.sin((i / period) + phase)
            P[k, 2 * i + 1] = torch.cos((i / period) + phase)
    return torch.nn.Parameter(P)

class LossAttr(dict):
    """Dictionary with attribute-style access for loss computation results."""

    def __init__(self, *args, **kwargs):
        defaults = {
            "loss": None,
            "log_likelihood": None,
            "means": None,
            "sds": None,
            "weights": None,
        }
        # Initialize with defaults first
        super().__init__(defaults)
        # Then update with any user-supplied dictionary or keyword arguments
        self.update(*args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            msg = f"'LossAttr' object has no attribute '{key}'"
            raise AttributeError(msg) from None


