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

from typing import Tuple, Union, Optional
import math
import torch
from pyro import distributions as dist
from alef.configs.base_parameters import INPUT_DOMAIN, CENTRAL_DOMAIN, NUMERICAL_POSITIVE_LOWER_BOUND


"""
ref:
Rahimi & Recht, NeurIPS 2007, Random Features for Large-Scale Kernel Machines
Wilson et al., ICML 2020, Efficiently Sampling Functions from Gaussian Process Posteriors
"""

class BayesianLinearModel(torch.nn.Module):
    def __init__(
        self,
        omega,
        bias,
        weight,
        x_expanded_already: bool,
        input_domain=INPUT_DOMAIN,
        shift_mean: bool=False,
        flip_center: bool=False,
        central_interval=CENTRAL_DOMAIN,
        mask=None
    ):
        """
        for the following shapes, Nk=num_kernels, Nf=num_functions, D=data dimension, L=num_fourier_features.
        
        :param omega: torch.Tensor, size [Nk, Nf, D, L]
        :param bias: torch.Tensor, size [Nk, Nf, L]
        :param weight: torch.Tensor, size [Nk, Nf, L]
        :param x_expanded_already: bool, if True, forward data contains [Nk, Nf] in dim 1, 2; otherwise, forward data has no kernel & function nums in batch shape yet
        :param input_domain: (lower bound, upper bound), each bound is float or [Nk, Nf, D] tensor
        :param shift_mean: bool, whether this model should be standardized to zero mean
        :param flip_center: bool, whether this function is forced positive at the center
        :param central_interval: (lower bound, upper bound), each bound is float or [Nk, Nf, D] tensor, if flip_center=True, then the center is determined from this interval
        :param mask: None or torch.Tensor, size [Nk, Nf, D], of 1 or 0, whether the dimension is masked for each function
        """
        super().__init__()
        if mask is None:
            self.register_buffer("omega", omega.clone()) # [Nk, Nf, input_dim, L]
        else:
            self.register_buffer(
                "omega",
                omega.masked_fill(mask.unsqueeze(-1) == 0, 0).clone()
            ) # [Nk, Nf, input_dim, L]
        self.register_buffer("bias", bias.clone()) # [Nk, Nf, L]
        self.register_buffer("weight", weight.clone()) # [Nk, Nf, L]
        scaler = torch.tensor( math.sqrt(2.0 / self.weight.shape[-1] ), device=self.weight.device)
        self.register_buffer("scaler", scaler)
        if shift_mean:
            shift_mean = self.mean(input_domain, mask=mask) # mind the numerical stability (1/omega bad when omega close to 0)
            #shift_mean = self.mean_by_sample(input_domain, mask=mask) # we are having memory issues, avoid this
        else:
            Nk = self.omega.shape[0]
            Nf = self.omega.shape[1]
            shift_mean = torch.zeros((Nk, Nf), device=self.omega.device, dtype=self.omega.dtype)
        self.register_buffer("shift_mean", shift_mean) # [Nk, Nf]
        flip_coeff = self.center_sign(central_interval, mask = mask) if flip_center else torch.ones_like(shift_mean)
        self.register_buffer("leading_sign", flip_coeff) # [Nk, Nf]
        self.x_expanded_already = x_expanded_already

    @property
    def num_kernels(self):
        return self.weight.shape[0]
    @property
    def num_functions(self):
        return self.weight.shape[1]
    @property
    def num_fourier_samples(self):
        return self.weight.shape[-1]

    def prune_batch_size(self, num_kernels, num_functions, num_fourier_samples):
        assert num_kernels <= self.num_kernels
        assert num_functions <= self.num_functions
        assert num_fourier_samples <= self.num_fourier_samples
        self.omega = self.omega[:num_kernels, :num_functions, :, :num_fourier_samples]
        self.bias = self.bias[:num_kernels, :num_functions, :num_fourier_samples]
        self.weight = self.weight[:num_kernels, :num_functions, :num_fourier_samples]
        self.scaler = torch.tensor( math.sqrt(2.0 / self.weight.shape[-1] ), device=self.weight.device)
        self.shift_mean = self.shift_mean[:num_kernels, :num_functions]

    def _extract_input_interval(self, input_domain: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]):
        r"""
        :param input_domain: (lower bound, upper bound), each bound is float or [Nk, Nf, D] tensor
        
        :return: ([*batch, D] torch.Tensor, [*batch, D] torch.Tensor), D is input dimension
        """
        Nk, Nf, D, L = self.omega.shape
        interval = []
        for i in range(2):
            b = input_domain[i]
            if isinstance(b, float):
                b = b*torch.ones([Nk, Nf, D], device=self.omega.device, dtype=self.omega.dtype)
            elif isinstance(b, torch.Tensor):
                b = b.expand([Nk, Nf, D])
            else:
                raise NotImplementedError
            interval.append(b)

        assert torch.all(interval[1] > interval[0])
        return tuple(interval)

    def center_sign(
        self,
        central_interval=CENTRAL_DOMAIN,
        mask: Optional[torch.Tensor]=None
    ):
        """
        make sure that the center at each dimension is positive.
        at the same time ensure GP prior.
        do this by assigning signs to weights
        
        :return: tensor of 1 or -1 (flipping coeff.), shape [num_kernels, num_functions]
        """
        # do not take sign based on domain integral, because we replace |1/omega.prod()| by a number when this is zero
        # the error is negligible for mean shifting but significant for integral sign
        # do point forward instead
        l, u = self._extract_input_interval(central_interval) # ([Nk, Nf, D], [Nk, Nf, D])
        center = (l + u) / 2 # [Nk, Nf, D]
        if not mask is None:
            center = center.masked_fill(mask == 0, 0) # [Nk, Nf, D]
        central_mean = self.raw_forward(center[None, :, :, None, :], x_expanded_already=True).squeeze(0).squeeze(-1)
        return torch.sign( central_mean - self.shift_mean )

    def mean_by_sample(
        self,
        input_domain: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]],
        mask: Optional[torch.Tensor]=None
    ):
        r"""
        :param input_domain: (lower bound, upper bound), each bound is float or [Nk, Nf, D] tensor
            it can be numerically bad to compute the integral,
            as omega may be 0 and we need 1/omega in the integral
        :param mask: torch.Tensor, size [Nk, Nf, D] if given, elements 0 (masked out) or 1 (kept in)

        :return: torch.Tensor, shape [Nk, Nf]
        """
        Nk, Nf, D, L = self.omega.shape
        N_per_try = 10
        interval = self._extract_input_interval(input_domain) # ([Nk, Nf, D], [Nk, Nf, D])
        lower, upper = interval
        x_pdf = dist.Uniform(low=lower.unsqueeze(-2), high=upper.unsqueeze(-2)).expand((Nk, Nf, N_per_try, D))
        mean = torch.zeros([Nk, Nf], device=self.omega.device, dtype=self.omega.dtype)
        N_try = 10000
        for _ in range(N_try): # for loop to avoid memory overhead
            x = x_pdf.sample() # [Nk, Nf, N_per_try, D]
            if not mask is None:
                x = x.masked_fill(mask.unsqueeze(-2) == 0, 0) # [Nk, Nf, N_per_try, D]
            linear_opt_x = torch.einsum('kfbd,kfdl->kfbl', x, self.omega) + self.bias.unsqueeze(-2) # [Nk, Nf, N_per_try, L]
            phi = self.scaler * torch.cos(linear_opt_x) # [Nk, Nf, N_per_try, L]
            f_out = torch.einsum('kfbl,kfl->kfb', phi, self.weight) # [Nk, Nf, N_per_try]
            mean += f_out.mean(-1)
        return mean / N_try

    def mean(
        self,
        input_domain: Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]],
        mask: Optional[torch.Tensor]=None
    ):
        r"""
        :param input_domain: (lower bound, upper bound), each bound is float or [Nk, Nf, D] tensor
            it can be numerically bad to compute the integral,
            as omega may be 0 and we need 1/omega in the integral
        :param mask: torch.Tensor, size [Nk, Nf, D] if given, elements 0 (masked out) or 1 (kept in)

        :return: torch.Tensor, shape [Nk, Nf]
        """
        Nk, Nf, D, L = self.omega.shape
        # fist solve the main integral part integral{ cos(omega@x + bias) dx} propto {sin or cos}(...)| x=bounds
        interval = self._extract_input_interval(input_domain) # ([Nk, Nf, D] lower bound, [Nk, Nf, D] upper bound)
        interval = torch.concat([interval[0].unsqueeze(-1), interval[1].unsqueeze(-1)], dim=-1) # [Nk, Nf, D, 2]
        rectangle = []
        for i in range(Nk):
            rec_i = []
            for j in range(Nf):
                interval_ij = torch.meshgrid(*torch.unbind(interval[i, j], dim=-2), indexing='ij') # ([2]*D, ..., [2]*D) ------ D len tuple
                rec_ij = []
                for d in range(D):
                    bound_d = torch.ravel(interval_ij[d])
                    rec_ij.append(bound_d.unsqueeze(-1))
                rec_ij = torch.cat(rec_ij, dim=-1) # [2**D, D]
                rec_i.append(rec_ij.unsqueeze(0))
            rec_i = torch.cat(rec_i, dim=0) # [Nf, 2**D, D]
            rectangle.append(rec_i.unsqueeze(0))
        rectangle = torch.cat(rectangle, dim=0).unsqueeze(-1) # [Nk, Nf, 2**D, D, 1]
        # integral{ cos(omega@x + bias) dx} propto {sin or cos}(...)| x=bounds,
        # rectangle contains the "bounds"

        omega = self.omega[..., None, :, :] # [Nk, Nf, 1, D, L]
        bias = self.bias[..., None, :] # [Nk, Nf, 1, L]

        if mask is None:
            cos_int_func = torch.cos if D % 2 == 0 else torch.sin
            integral = cos_int_func( (omega * rectangle).sum(-2) + bias )# [Nk, Nf, 2**D, L]
        else: # mask: [Nk, Nf, D]
            mask_expand = mask.unsqueeze(-2).unsqueeze(-1) # [Nk, Nf, 1, D, 1]
            inside_triangular_func = (
                omega * rectangle.masked_fill(mask_expand == 0, 0)
            ).sum(-2) + bias # [Nk, Nf, 2**D, L]
            masked_dim = mask_expand.sum(-2) # [Nk, Nf, 1, 1]
            integral = torch.where(
                (masked_dim % 2 == 0).expand(inside_triangular_func.shape),
                torch.cos(inside_triangular_func),
                torch.sin(inside_triangular_func)
            ) # [Nk, Nf, 2**D, L]
        integral = integral.reshape( (Nk, Nf) + (2,)*D + (L, ) )# [Nk, Nf, 2, ..., 2, L]
        for _ in range(D):
            integral = integral.diff(dim=-2).squeeze(-2)
        # integral: [Nk, Nf, L]

        # now we multiply the coefficient back, scaler * w * integral coeff. / input_area
        # integral coeff. = 1 / omega.prod(x_dim)
        # mind the numerical stability
        omega = self.omega # [Nk, Nf, D, L]
        if not mask is None:
            omega = omega.masked_fill(mask.unsqueeze(-1) == 0, 1)
        sign = omega.sign().prod(dim=-2) # [Nk, Nf, L]
        if mask is None and D % 4 in [2, 3]:
            sign = sign.neg() # [Nk, Nf, L]
        elif not mask is None:
            masked_dim = mask.sum(-1).unsqueeze(-1) # [Nk, Nf, 1]
            sign = torch.where(
                torch.logical_or(masked_dim % 4 == 2, masked_dim % 4 == 3),
                sign.neg(),
                sign
            )

        omega_abs = omega.abs()
        omega_abs_product = torch.clamp(
            omega_abs.prod(dim=-2), # [Nk, Nf, L]
            min=NUMERICAL_POSITIVE_LOWER_BOUND # make sure |1/omega| won't be |1/0|
        )
        integral_cof = omega_abs_product.reciprocal() # [Nk, Nf, L]
        input_area = (
            interval[..., 1] - interval[..., 0] # [Nk, Nf, D]
        ).sum(dim=-1).unsqueeze(-1)
        cof = self.scaler * integral_cof / input_area # [Nk, Nf, L]
        mean_per_feature = sign * self.weight * cof * integral
        return mean_per_feature.sum(-1)

    def raw_forward(self, x: torch.Tensor, x_expanded_already: bool):
        r"""
        the same as forward, except that mean shifting and flipping is not performed
        
        :param x: [B, Nk, Nf, ..., input_dim] or [B, 1, 1, ..., input_dim] if x_expanded_already,
            [B, ..., input_dim] otherwise
        """
        if x_expanded_already:
            # x: [B, num_kernels, num_functions, ..., input_dim] or [B, 1, 1, ..., input_dim]
            # num_kernels: batch size of kernel hyperparameters
            msg = f'shape of x[1:3] {x.shape[1:3]} does not match [num of kernel, num of funcs per kernel] [{self.omega.shape[0]}, {self.weight.shape[1]}]'
            assert x.shape[1] == 1 or x.shape[1] == self.omega.shape[0], msg
            assert x.shape[2] == 1 or x.shape[2] == self.weight.shape[1], msg
            xx = torch.einsum('bkf...->kfb...', x) # turn to [num_kernels, num_functions, B, ..., input_dim]
            xx = xx.expand(self.omega.shape[:2] + xx.shape[2:]) if xx.flatten(0, 1).shape[0] == 1 else xx
        else:
            #x: [B, ..., input_dim] 
            xx = x.expand(self.omega.shape[:2] + x.shape) # turn to [num_kernels, num_functions, B, ..., input_dim]
        linear_opt_x = torch.einsum('kfbd,kfdl->kfbl', xx.flatten(start_dim=2, end_dim=-2), self.omega) + self.bias.unsqueeze(-2) # [num_kernels, num_functions, xx.shape[2:-1]_flatten, L]
        phi = self.scaler * torch.cos(linear_opt_x) # [num_kernels, num_functions, xx.shape[2:-1]_flatten, L]
        f_out = torch.einsum('kfbl,kfl->kfb', phi, self.weight) # [num_kernels, num_functions, xx.shape[2:-1]_flatten]
        # now reshape and transpose
        f_out = f_out.reshape(xx.shape[:-1]) # [num_kernels, num_functions, xx.shape[2:-1]]
        return torch.einsum('kfb...->bkf...', f_out) # [B, num_kernels, num_functions, x.shape[3:-1]]

    def forward(self, x: torch.Tensor):
        if self.x_expanded_already:
            # x: [B, num_kernels, num_functions, ..., input_dim] or [B, 1, 1, ..., input_dim]
            # num_kernels: batch size of kernel hyperparameters
            msg = f'shape of x[1:3] {x.shape[1:3]} does not match [num of kernel, num of funcs per kernel] [{self.omega.shape[0]}, {self.weight.shape[1]}]'
            assert x.shape[1] == 1 or x.shape[1] == self.omega.shape[0], msg
            assert x.shape[2] == 1 or x.shape[2] == self.weight.shape[1], msg
            xx = torch.einsum('bkf...->kfb...', x) # turn to [num_kernels, num_functions, B, ..., input_dim]
            xx = xx.expand(self.omega.shape[:2] + xx.shape[2:]) if xx.flatten(0, 1).shape[0] == 1 else xx
        else:
            #x: [B, ..., input_dim] 
            xx = x.expand(self.omega.shape[:2] + x.shape) # turn to [num_kernels, num_functions, B, ..., input_dim]
        linear_opt_x = torch.einsum('kfbd,kfdl->kfbl', xx.flatten(start_dim=2, end_dim=-2), self.omega) + self.bias.unsqueeze(-2) # [num_kernels, num_functions, xx.shape[2:-1]_flatten, L]
        phi = self.scaler * torch.cos(linear_opt_x) # [num_kernels, num_functions, xx.shape[2:-1]_flatten, L]
        f_out = torch.einsum('kfbl,kfl->kfb', phi, self.weight) # [num_kernels, num_functions, xx.shape[2:-1]_flatten]
        f_out_zero_mean = self.leading_sign.unsqueeze(-1) * ( f_out - self.shift_mean.unsqueeze(-1) ) # [num_kernels, num_functions, xx.shape[2:-1]_flatten]
        # now reshape and transpose
        f_out_zero_mean = f_out_zero_mean.reshape(xx.shape[:-1]) # [num_kernels, num_functions, xx.shape[2:-1]]
        return torch.einsum('kfb...->bkf...', f_out_zero_mean) # [B, num_kernels, num_functions, x.shape[3:-1]]
