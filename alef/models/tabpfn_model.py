import time
import numpy as np
import torch
from typing import List, Tuple, Optional, Callable
from alef.models.base_model import BaseModel
from tabpfn import TabPFNRegressor
import logging
from alef.utils.custom_logging import getLogger

logger = getLogger(__name__)

class TabPFNModel(BaseModel):
    """
    Class that implements standard PFN regression for our API
    """

    def __init__(
        self,
        device: str = "cpu",
        **kwargs,
    ):
        self.device = device
        self.model = None
        self.print_summaries = True

    def reset_model(self):
        self.model = None

    def build_model(self, *args, **kwargs):
        self.model = TabPFNRegressor(n_estimators=1, device=self.device)

    def infer(self, x_data: np.array, y_data: np.array, class_mask: np.array=None):
        """
        Main entrance method for learning the model - training methods are called if hp should be done otherwise only gpflow model is build

        Arguments:
            x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
            class_mask: (n,) bool array, y_data[class_mask] is binary class (0 or 1) and y_data[~class_mask] is regression value
        """
        if self.model is None:
            self.build_model()
        self.model.fit(x_data, y_data[..., 0])

    def print_model_summary(self):
        if logger.isEnabledFor(logging.DEBUG):
            pass

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
        with torch.no_grad():
            full_out_dict = self.model.predict(x_test, output_type="full")
            mean = full_out_dict["mean"] # [n,]
            criterion = full_out_dict["criterion"].to(self.device)
            logits = full_out_dict["logits"].to(self.device) #  [n, 5000]
            variance = criterion.variance(logits).cpu().numpy() # [n,]
            return mean, np.sqrt(variance)

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
        with torch.no_grad():
            yt = torch.tensor(y_test, dtype=torch.float32).to(self.device).squeeze(-1)  # [n]
            full_out_dict = self.model.predict(x_test, output_type="full")
            criterion = full_out_dict["criterion"].to(self.device)
            logits = full_out_dict["logits"].to(self.device) #  [n, 5000]

            log_likelis = criterion.forward(logits, yt).cpu() # [n]
            return log_likelis.numpy()

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Method for calculating the entropy of the predictive distribution for test sequence - used for acquistion function in active learning

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        entropy array with shape (n,1)
        """
        with torch.no_grad():
            full_out_dict = self.model.predict(x_test, output_type="full")
            criterion = full_out_dict["criterion"].to(self.device)
            logits = full_out_dict["logits"].to(self.device) #  [n, 5000]
            # 
            probs = torch.softmax(logits, -1) # [n, 5000]
            log_pdf = criterion.compute_scaled_log_probs(logits) # [n, 5000]
            valid_mask = torch.isfinite(log_pdf) # might have buckets of almost zero width, which contributes no entropy
            entropy_per_bucket = torch.zeros_like(probs)
            entropy_per_bucket[valid_mask] = - probs[valid_mask] * log_pdf[valid_mask] # (n, 5000)
            return entropy_per_bucket.sum(-1).cpu().unsqueeze(-1).numpy() # (n, 1)

    def entropy_predictive_dist_full_cov(self, *args, **kwargs):
        raise NotImplementedError

    def deactivate_summary_printing(self):
        self.print_summaries = False

if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    model = TabPFNModel()
    
    # Load Boston Housing data
    df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
    X = df.data
    y = df.target.astype(float)  # Ensure target is float for regression

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train the model
    model.infer(X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1))

    # Predict on the test set
    mu, std = model.predictive_dist(X_test.to_numpy())
    print(f"Predictions mean: {mu}")
    print(f"Predictions std: {std}")
    # Evaluate the model
    mse = mean_squared_error(y_test, mu)
    r2 = r2_score(y_test, mu)

    print("Mean Squared Error (MSE):", mse)
    print("R² Score:", r2)
