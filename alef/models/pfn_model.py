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

import time
import numpy as np
import torch
from typing import List, Tuple, Optional, Callable
from alef.models.base_model import BaseModel
from alef.models.pfn.pfn import PFN
import logging
from alef.utils.custom_logging import getLogger

logger = getLogger(__name__)

class PFNModel(BaseModel):
    """
    Class that implements standard PFN regression for our API
    """

    def __init__(
        self,
        pfn_backend_config,
        checkpoint_path: Optional[str],
        device: str = "cpu",
        **kwargs,
    ):
        self.device = device
        if checkpoint_path is None:
            self.model = PFN(
                dim_x=pfn_backend_config.input_dimension,
                dim_y=pfn_backend_config.output_dimension,
                d_model=pfn_backend_config.d_model,
                dim_feedforward=pfn_backend_config.dim_feedforward,
                nhead=pfn_backend_config.nhead,
                dropout=pfn_backend_config.dropout,
                num_layers=pfn_backend_config.num_layers,
                head_num_buckets=pfn_backend_config.head_num_buckets,
            )
        else:
            self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
            cfg = self.checkpoint["config"]["model"]
            self.model = PFN(
                dim_x=cfg["dim_x"],
                dim_y=cfg["dim_y"],
                d_model=cfg["d_model"],
                dim_feedforward=cfg["dim_feedforward"],
                nhead=cfg["nhead"],
                dropout=cfg["dropout"],
                num_layers=cfg["num_layers"],
                head_num_buckets=cfg["head_num_buckets"],
            )
            state_dict = self.checkpoint["model_state_dict"]
            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                # Remove _orig_mod. prefix from keys
                state_dict = {
                    key.replace("_orig_mod.", ""): value 
                    for key, value in state_dict.items()
                }
            
            self.model.load_state_dict(state_dict)
        #
        self.model = self.model.to(self.device)
        self.model.eval()

        self.print_summaries = True

        self.xc = None
        self.yc = None

    def reset_model(self):
        self.xc = None
        self.yc = None

    def infer(self, x_data: np.array, y_data: np.array, class_mask: np.array=None):
        """
        Main entrance method for learning the model - training methods are called if hp should be done otherwise only gpflow model is build

        Arguments:
            x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
            class_mask: (n,) bool array, y_data[class_mask] is binary class (0 or 1) and y_data[~class_mask] is regression value
        """
        self.xc = torch.tensor(x_data, dtype=torch.float32).to(self.device).unsqueeze(0)  # [1, N, d]
        self.yc = torch.tensor(y_data, dtype=torch.float32).to(self.device).unsqueeze(0)  # [1, N, 1]

    def build_model(self, *args, **kwargs):
        pass

    def print_model_summary(self):
        if logger.isEnabledFor(logging.DEBUG):
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Model built: {trainable_params:,} trainable parameters (total: {total_params:,})")

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> float:
        raise NotImplementedError

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,)
        sigma array with shape (n,)
        """
        assert self.xc is not None and self.yc is not None, "Model has not been inferred yet. Please call infer() first."
        with torch.no_grad():
            xt = torch.tensor(x_test, dtype=torch.float32).to(self.device).unsqueeze(0)  # [1, N, d]
            samples = self.model.sample(self.xc, self.yc, xt, num_samples=100) # [1, N, 100, 1]
            pred_mus = samples.mean(dim=-2).cpu().squeeze(0)  # [N, 1]
            pred_sigmas = samples.std(dim=-2).cpu().squeeze(0)  # [N, 1]
            return pred_mus.squeeze(-1).numpy(), pred_sigmas.squeeze(-1).numpy()

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        """
        Method for calculating the log likelihood value of the the predictive distribution at the test input points (evaluated at the output values)
        - method is therefore for validation purposes only

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points
        y_test: Array of test output points with shape (n,1)

        Returns:
        array of shape (n,) with log liklihood values
        """
        assert self.xc is not None and self.yc is not None, "Model has not been inferred yet. Please call infer() first."
        with torch.no_grad():
            xt = torch.tensor(x_test, dtype=torch.float32).to(self.device).unsqueeze(0)  # [1, N, d]
            yt = torch.tensor(y_test, dtype=torch.float32).to(self.device).unsqueeze(0)  # [1, N, 1]
            log_likelis = self.model.forward(self.xc, self.yc, xt, yt).log_likelihood.cpu().squeeze(-1) # [1, Nt]
            return log_likelis.squeeze(0).numpy()

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Method for calculating the entropy of the predictive distribution for test sequence - used for acquistion function in active learning

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        entropy array with shape (n,1)
        """
        assert self.xc is not None and self.yc is not None, "Model has not been inferred yet. Please call infer() first."
        with torch.no_grad():
            xt = torch.tensor(x_test, dtype=torch.float32).to(self.device).unsqueeze(0)  # [1, N, d]
            ent = self.model.predictive_entropy(self.xc, self.yc, xt) # [1, N, 1]
            return ent.squeeze(0).cpu().numpy()

    def entropy_predictive_dist_full_cov(self, *args, **kwargs):
        raise NotImplementedError

    def deactivate_summary_printing(self):
        self.print_summaries = False
