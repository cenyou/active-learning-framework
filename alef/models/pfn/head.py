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
Head modules of the Prior-Data Fitted Networks (PFNs).
Large part taken from
github.com/automl/TransformersCanDoBayesianInference
github.com/automl/PFNs
"""
import warnings
import torch
from torch import nn
from torch.distributions import HalfNormal, constraints
from typing import Optional, Tuple, Union

from .utils import LossAttr

__all__ = [
    'get_bucket_borders',
    'BarDistribution',
    'FullSupportBarDistribution',
    'LogitsKnownDistribution',
]

###### Bucket borders computation ######
######
def _get_bucket_borders_from_ys(
    num_buckets: int,
    ys: torch.Tensor,
    full_range: Optional[Tuple[float, float]] = None,
    *,
    widen_borders_factor: Optional[float] = None,
) -> torch.Tensor:
    """ Compute bucket boundaries for discretizing values based on ys.
    
    Args:
        num_buckets: Number of buckets.
        ys: 1D tensor of values to use for quantile-based buckets.
        full_range: Optional (min, max) tuple for the range to cover. If provided, it will be used to ensure the buckets cover this range.
        widen_borders_factor: Optional factor to widen the bucket borders.
    Returns:
        Tensor of bucket boundaries of shape (num_buckets + 1,)
    """
    ys = ys.flatten()
    ys = ys[~torch.isnan(ys)]
    assert len(ys) > num_buckets, f"Number of ys :{len(ys)} must be larger than num_buckets: {num_buckets}"
    # warn if ys contain NaN values
    if torch.any(torch.isnan(ys)):
        warnings.warn(
            "PFN bucket borders are obtained from ys. The ys contain NaN values, which will be removed."
        )
    # if num of ys is not divisible by num_buckets, we drop the last few ys
    if len(ys) % num_buckets:
        warnings.warn(
            f"PFN bucket borders are obtained from ys."
            + f" The number of ys ({len(ys)}) is not divisible by the specified num_buckets ({num_buckets})."
            + f" Dropping last {len(ys) % num_buckets} ys."
        )
        ys = ys[: -(len(ys) % num_buckets)]

    # finish data processing
    # compute borders
    ys_per_bucket = len(ys) // num_buckets

    # if full_range is provided, we use it to ensure the buckets cover this range
    if full_range is None:
        full_range = (ys.min(), ys.max())
    else:
        assert full_range[0] <= ys.min()
        assert full_range[1] >= ys.max()
        full_range = torch.tensor(full_range)  # type: ignore

    # compute the bucket borders based on ys
    ys_sorted, ys_order = ys.sort(0)  # type: ignore
    borders = (
        ys_sorted[ys_per_bucket - 1 :: ys_per_bucket][:-1]
        + ys_sorted[ys_per_bucket::ys_per_bucket]
    ) / 2
    borders = torch.cat(
        [full_range[0].unsqueeze(0), borders, full_range[1].unsqueeze(0)],  # type: ignore
        0,
    )
    if widen_borders_factor is not None:
        borders = borders * widen_borders_factor
    
    return borders


def _get_bucket_borders_from_range(
        num_buckets: int,
        full_range: Tuple[float, float],
        *,
        widen_borders_factor: Optional[float] = None,
    ) -> torch.Tensor:
    """ Compute bucket boundaries for discretizing values based on a full range.
    Args:
        num_buckets: Number of buckets.
        full_range: (min, max) tuple for the range to cover.
        widen_borders_factor: Optional factor to widen the bucket borders.
    Returns:
        Tensor of bucket boundaries of shape (num_buckets + 1,)
    """
    assert len(full_range) == 2, f'Expected full_range to be a tuple of (min, max), got {full_range}'
    assert full_range[0] < full_range[1], f'Invalid range {full_range}'

    class_width = (full_range[1] - full_range[0]) / num_buckets
    borders = torch.cat([
            full_range[0] + torch.arange(num_buckets).float() * class_width,
            torch.tensor(full_range[1]).unsqueeze(0)
        ], 0)
    
    return borders


def get_bucket_borders(
    num_buckets: int,
    full_range: Optional[Tuple[float, float]] = None,
    ys: Optional[torch.Tensor] = None,
    *,
    widen_borders_factor: Optional[float] = None,
) -> torch.Tensor:
    """Decide for a set of borders for the buckets based on a distritbution of ys.
    The bucket borders are computed by computing the quantiles of the ys.

    Args:
        num_buckets:
            This is only tested for num_buckets=1, but should work for larger
            num_buckets as well.
        full_range:
            If ys is not passed, this is the range of the ys that should be
            used to estimate the bucket borders.
        ys:
            If ys is passed, this is the ys that should be used to estimate the bucket
            borders. Do not pass full_range in this case.
        widen_borders_factor:
            If set, the bucket borders are widened by this factor.
            This allows to have a slightly larger range than the actual data.
    Returns:
        Tensor of bucket boundaries of shape (num_buckets + 1,).
    """
    assert (ys is not None) or (full_range is not None)
    if ys is not None:
        return _get_bucket_borders_from_ys(num_buckets, ys=ys, full_range=full_range, widen_borders_factor=widen_borders_factor)
    elif full_range is not None:
        return _get_bucket_borders_from_range(num_buckets, full_range=full_range, widen_borders_factor=widen_borders_factor)
    else:
        raise ValueError("Either ys or full_range must be provided.")


###### Riemann distribution, without/with tails ######
######
class BarDistribution(nn.Module):
    def __init__(self, borders: torch.Tensor):
        """
        Bar distribution with given borders.
        
        :param borders: tensor of shape (num_bars + 1,) the borders of bins, sorted in ascending order.
        """
        # sorted list of borders
        super().__init__()
        assert len(borders.shape) == 1
        self.register_buffer('borders', borders)
        full_width = self.bucket_widths.sum()
        assert torch.isclose(full_width, self.borders[-1] - self.borders[0], rtol=1e-4), f'diff: {full_width - (self.borders[-1] - self.borders[0])}'
        assert ( self.bucket_widths >= 0.0 ).all(), "Please provide sorted borders!"

    @property
    def bucket_widths(self) -> torch.Tensor:
        return self.borders[1:] - self.borders[:-1]

    @property
    def bucket_means(self) -> torch.Tensor:
        return self.borders[:-1] + self.bucket_widths / 2

    @property
    def num_bars(self) -> int:
        return len(self.borders) - 1

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, -1)
        return p @ self.bucket_means

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        density = logits.softmax(-1) / self.bucket_widths
        mode_inds = density.argmax(-1)
        return self.bucket_means[mode_inds]

    def map_to_bucket_idx(self, y: torch.Tensor) -> torch.Tensor:
        """
        Map the values in y to the bucket indices based on the borders.
        The outputs are indices.
        In particular (idx: interval)::
            >>> -1:  ( -inf, borders[0] ],
            >>> 0:   (borders[0], borders[1] ],
            >>> ...,
            >>> num_bars - 1:  (borders[-2], borders[-1] ],
            >>> num_bars:      (borders[-1], +inf ).

        :param y: tensor of shape (*batch_shape,)
        :return: tensor of shape (*batch_shape,) with indices of the buckets.
        """
        target_sample = torch.searchsorted(self.borders, y) - 1
        target_sample[y == self.borders[0]] = 0
        target_sample[y == self.borders[-1]] = self.num_bars - 1
        return target_sample

    def bucket_log_densities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the densities of the buckets given logits.
        :param logits: tensor of shape (*batch_shape, num_bars) with logits for each bucket.

        :return: tensor of shape (*batch_shape, num_bars)
        """
        assert logits.shape[-1] == self.num_bars, f'{logits.shape[-1]} vs {self.num_bars}'
        bucket_log_probs = torch.log_softmax(logits, -1)
        scaled_bucket_log_probs = bucket_log_probs - torch.log(self.bucket_widths)
        return scaled_bucket_log_probs

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy given logits.
        :param logits: tensor of shape (*batch_shape, num_bars) with logits for each bucket.

        :return: tensor of shape (*batch_shape,)
        """
        probs = torch.softmax(logits, -1) # (*batch_shape, num_bars) cumulative log probs of each bucket
        log_pdf = self.bucket_log_densities(logits) # (*batch_shape, num_bars)
        entropy_per_bucket = torch.zeros_like(log_pdf) # if bucket width is zero, entropy is zero
        valid_max = torch.isfinite(log_pdf)
        entropy_per_bucket[valid_max] = - probs[valid_max] * log_pdf[valid_max] # (*batch_shape, num_bars)
        return entropy_per_bucket.sum(-1) # (*batch_shape,)

    def icdf(self, logits: torch.Tensor, cdf_values: torch.Tensor) -> torch.Tensor:
        """Inverse cumulative distribution function (icdf) for the bar distribution.
        We refactorize PFN implementation. Overall, the icdf is computed as follows::
            >>> 1. find the bucket index corresponding to cdf_values
            >>> 2. cumulate the densities until the previous bucket
            >>> 3. in the current bucket, density is uniform, so the position is computed
            >>> 4. add the residual density of the current bucket to the cumulative density until the previous bucket
        
        :param logits: Tensor of shape (*batch_shape, num_bars)
        :param cdf_values: Tensor of shape (*batch_shape, num_samples)
        :return: inverse cdf values of shape (*batch_shape, num_samples)
        """
        assert torch.all(cdf_values >= 0) and torch.all(cdf_values <= 1), f'cdf_values {cdf_values} must be in [0, 1]'
        probs = logits.softmax(-1)
        cumprobs = torch.cumsum(probs, -1) # (*batch_shape, num_bars), starting from right border of first bucket
        cumprobs = torch.cat([ torch.zeros_like(cumprobs[..., :1]), cumprobs ], dim=-1) # (*batch_shape, num_bars + 1), cdf at each border

        # find the bucket index corresponding to cdf_values
        idx = torch.searchsorted( cumprobs[..., 1:], cdf_values ).clip(min=0, max=self.num_bars - 1) # (*batch_shape, num_samples), values in [0, num_bars - 1]

        # then compute the position in the bucket
        residual_prob = cdf_values - cumprobs.gather(-1, idx)
        residual_ratio = residual_prob / probs.gather(-1, idx)

        left_border = self.borders.expand(cumprobs.shape).gather(-1, idx)
        bucket_width = self.bucket_widths.expand(logits.shape).gather(-1, idx)

        return left_border + bucket_width * residual_ratio

    def sample(self, logits: torch.Tensor, num_samples: int=1) -> torch.Tensor:
        """
        Sample from the distribution.

        :param logits: tensor of shape (*batch_shape, num_bars) with logits for each bucket.
        :param num_samples: number of samples to generate for each distribution.
        :return: tensor of shape (*batch_shape, num_samples).
        """
        assert num_samples > 0, f'num_samples must be positive, got {num_samples}'

        batch_shape = logits.shape[:-1]
        sample_shape = batch_shape + (num_samples,)

        p_cdf = torch.rand(sample_shape, device=logits.device)

        return self.icdf(logits, p_cdf)

    def log_prob(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the log likelihood of y given model density logits.

        :param logits: tensor of shape (*batch_shape, num_bars) with logits for each bucket.
        :param y: tensor of shape (*batch_shape, num_samples), num_samples can be 1.

        :return: tensor of shape (*batch_shape, num_samples) with log likelihood of y.
        """
        bucket_idx = self.map_to_bucket_idx(y)
        assert (bucket_idx >= 0).all() and (bucket_idx < self.num_bars).all(), f'y not in support set for borders (min_y, max_y) {self.borders}'

        scaled_bucket_log_probs = self.bucket_log_densities(logits)

        return scaled_bucket_log_probs.gather(-1, bucket_idx)

    def forward(self, logits: torch.Tensor, y: torch.Tensor, reduce_ll: bool = True) -> torch.Tensor:
        """
        Compute the loss and log likelihood of the target data y given model density logits.

        :param logits: tensor of shape (*batch_shape, num_bars) with logits for each bucket.
        :param y: tensor of shape (*batch_shape,) with target data.
        :param reduce_ll: If True, reduce log likelihood to a scalar.
        :return: LossAttr containing loss and log likelihood of target data.
        """
        tar_ll = self.log_prob(logits, y.unsqueeze(-1)).squeeze(-1)  # shape (*batch_shape)
        
        if reduce_ll:
            loss = -tar_ll.mean()
        else:
            loss = -tar_ll
        
        return LossAttr(
            loss=loss,
            log_likelihood=tar_ll,
        )


class FullSupportBarDistribution(BarDistribution):
    @staticmethod
    def halfnormal_with_p_weight_before(range_max, p=.5):
        """
        Create a half-normal distribution with width according to range_max.
        The density is unnormalized, meaning that the integral is 1.
        """
        s = range_max / HalfNormal(torch.tensor(1.)).icdf(torch.tensor(p))
        return HalfNormal(s)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        p = torch.softmax(logits, -1)
        side_normals = (self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
                        self.halfnormal_with_p_weight_before(self.bucket_widths[-1]))
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]
        return p @ bucket_means

    def mode(self, logits: torch.Tensor) -> torch.Tensor:
        density = logits.softmax(-1) / self.bucket_widths
        mode_inds = density.argmax(-1)

        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        side_normals = (self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
                        self.halfnormal_with_p_weight_before(self.bucket_widths[-1]))
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]

        return bucket_means[mode_inds]

    def sample(self, logits: torch.Tensor, num_samples: int=1) -> torch.Tensor:
        """
        Sample from the distribution.
        Note that PFN did not implement tailed inverse cdf, and thus the sampling ignore the tails.
        The samples are in the buckets borders only.

        :param logits: tensor of shape (*batch_shape, num_bars) with logits for each bucket.
        :param num_samples: number of samples to generate for each distribution.
        :return: tensor of shape (*batch_shape, num_samples).
        """
        assert num_samples > 0, f'num_samples must be positive, got {num_samples}'

        batch_shape = logits.shape[:-1]
        sample_shape = batch_shape + (num_samples,)

        p_cdf = torch.rand(sample_shape, device=logits.device)

        return self.icdf(logits, p_cdf)

    def log_prob(self, logits, y):
        """
        Compute the log likelihood of y given model density logits.

        :param logits: tensor of shape (*batch_shape, num_bars) with logits for each bucket.
        :param y: tensor of shape (*batch_shape, num_samples), num_samples can be 1.

        :return: tensor of shape (*batch_shape, num_samples) with log likelihood of y.
        """
        assert self.num_bars > 1
        bucket_idx = self.map_to_bucket_idx(y)
        bucket_idx.clamp_(0, self.num_bars-1)

        scaled_bucket_log_probs = self.bucket_log_densities(logits) # (*batch_shape, num_bars)
        log_probs = scaled_bucket_log_probs.gather(-1, bucket_idx) # (*batch_shape, num_samples)

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1])
        )

        # bucket cumulative prob is bucket_width * density
        # half-normal, after normalization, would be bucket_width * density * HalfNormal.pdf
        log_probs[bucket_idx == 0] += torch.log(self.bucket_widths[0]) + \
            side_normals[0].log_prob( (self.borders[1] - y[bucket_idx == 0] ).clamp(min=.00000001))
        log_probs[bucket_idx == self.num_bars-1] += torch.log(self.bucket_widths[-1]) + \
            side_normals[1].log_prob( y[bucket_idx == self.num_bars-1] - self.borders[-2] )

        return log_probs


###### Interface for distributions with known logits
###### We implement this so we can use the distributions as if they were torch.distributions.Distribution
######
class LogitsKnownDistribution(torch.distributions.Distribution):
    arg_constraints = {}
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        bar_distribution: Union[BarDistribution, FullSupportBarDistribution],
        logits: torch.Tensor,
    ):
        """
        BarDistribution or FullSupportBarDistribution with known logits.
        Use this to resemle the behaviors of torch.distributions.Distribution
        when the logits are already known.
        Note that BarDistribution & FullSupportBarDistribution handle 1D samples of shape (*batch_shape, num_samples),
        while this wrapper adds the output dimension back: (*batch_shape, num_samples, dim_y=1).

        :param bar_distribution: BarDistribution or FullSupportBarDistribution with known logits.
        :param logits: tensor of shape (*batch_shape, num_bars) with logits for each bucket.
        """
        batch_shape = logits.shape[:-1]
        num_bars = logits.shape[-1]
        assert isinstance(bar_distribution, (BarDistribution, FullSupportBarDistribution))
        assert num_bars == bar_distribution.num_bars

        super().__init__(batch_shape=batch_shape, validate_args=False)
        self._bar_distribution = bar_distribution
        self._logits = logits

    @property
    def logits(self) -> torch.Tensor:
        """
        Return the logits of the distribution.
        :return: tensor of shape (*batch_shape, num_bars)
        """
        return self._logits

    @logits.setter
    def logits(self, logits: torch.Tensor):
        """
        Set the logits of the distribution.
        :param logits: tensor of shape (*batch_shape, num_bars)
        """
        assert logits.shape == self._logits.shape, f"Expected logits shape {self._logits.shape}, got {logits.shape}"
        self._logits = logits

    @property
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the distribution.
        :return: tensor of shape (*batch_shape, dim_y=1)
        """
        return self._bar_distribution.mean(self._logits).unsqueeze(-1)

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the distribution.
        :return: tensor of shape (*batch_shape, dim_y=1)
        """
        return self._bar_distribution.mode(self._logits).unsqueeze(-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute the log likelihood of the value.

        :param value: tensor of shape (*batch_shape, num_samples, dim_y=1).
        :return: tensor of shape (*batch_shape, num_samples, dim_y=1) with log likelihood of the value.
        """
        assert value.shape[:-2] == self._logits.shape[:-1], f"Expected value *batch_shape {self.batch_shape}, got {value.shape[:-2]}"
        return self._bar_distribution.log_prob(self._logits, value.squeeze(-1)).unsqueeze(-1)

    def entropy(self) -> torch.Tensor:
        """
        Compute the entropy given logits.
        :return: tensor of shape (*batch_shape, dim_y=1)
        """
        return self._bar_distribution.entropy(self._logits).unsqueeze(-1)

    def sample(self, num_samples: int=1) -> torch.Tensor:
        """
        Sample from the distribution.

        :param logits: tensor of shape (*batch_shape, num_bars) with logits for each bucket.
        :param num_samples: number of samples to generate for each distribution.
        :return: tensor of shape (*batch_shape, num_samples, dim_y=1).
        """
        return self._bar_distribution.sample(self._logits, num_samples).unsqueeze(-1)
