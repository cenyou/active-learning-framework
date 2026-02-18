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

import math
import torch
from torch.nn import Module
from .utils import compute_mean_batch, compute_kernel_batch


class GaussianProcessComputer(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_sequence_milestones(self, n_initial_min, n_initial_max, n_steps_min, n_steps_max):
        """
        We might need only the queries for certain computations.
        The pipeline always have the initial set placed first, followed by the queries.
        The learner does AL from n_initial for n_steps, but training pipeline may sample n_initial & n_steps.
        
        :param n_initial_min: min num of initial points
        :param n_initial_max: max num of initial points
        :param n_steps_min: min num of policy selected in each batch
        :param n_steps_max: max num of policy selected in each batch
        """
        self.n_initial_min = n_initial_min
        self.n_initial_max = n_initial_max
        self.n_steps_min = n_steps_min
        self.n_steps_max = n_steps_max

    def compute_mean_batch(self, mean_list, x, mask=None, *, batch_dim: int=1):
        # X needs to be [*batch_size, D]
        # batch_dim[batch_dim]
        return compute_mean_batch(mean_list, x, mask=mask, batch_dim=batch_dim)

    def compute_kernel_batch(self, kernel_list, x1, x2, noise_var_list=None, mask=None, *, batch_dim: int=1, return_linear_operator: bool=False):
        return compute_kernel_batch(kernel_list, x1, x2, noise_var_list, mask=mask, batch_dim=batch_dim, return_linear_operator=return_linear_operator)

    def compute_cholesky(self, cov_matrix):
        return torch.linalg.cholesky(cov_matrix)

    def compute_cholesky_update(self, L_cov_matrix, cov_cross_matrix, cov_new):
        r"""
        compute cholesky decomposition of K, where
        K = [[ L @ L.T            , cov_cross_matrix ],
             [ cov_cross_matrix.T , cov_new          ]]
        
        :param L_cov_matrix: [*batch_shape, P, P] tensor, cholesky decomposition of cov_matrix
        :param cov_cross_matrix: float or [*batch_shape, P, T] tensor
        :param cov_new: [*batch_shape, T, T] tensor, can be None if return_cov is False
        :return: [*batch_shape, P + T, P + T] tensor,
            [[ L_cov_matrix, 0],
             [ S.T, cholesky(cov_new - S.T @ S)]]
            S = L^-1 @ cov_cross_matrix
        """
        S = self.compute_cholesky_inv_B(L_cov_matrix, cov_cross_matrix) # [*batch_shape, P, T]
        ST = torch.einsum('...ij->...ji', S) # [*batch_shape, T, P]
        L_new = self.compute_cholesky( # [*batch_shape, T, T]
            cov_new - torch.matmul(ST, S)
        )
        return torch.cat((
            torch.cat((L_cov_matrix, torch.zeros_like(S)), dim=-1),
            torch.cat((ST, L_new), dim=-1)
        ), dim=-2)

    def compute_cholesky_inv_B(self, L, B, prior_mask_idx=None):
        r"""
        :param L: [*batch_shape, P, P] tensor, cholesky decomposition of cov_observation_matrix
        :param B: [*batch_shape, P, T] tensor
        :param prior_mask_idx: None or [*batch_shape] tensor containing 1, ..., P. The integers indicate how many observations the posterior conditions on.
        :return: [*batch_shape, P, T] tensor, L^-1 @ B
        """
        # (x0,...,xP-1) = L^-1 (b0,...,bP-1) has this property:
        # x0 depends only on L[0,0], b0
        # x1 depends only on L[:2,:2], b0, b1, x0, and so on
        # Therefore: solve_triangular as it is, then replaced unused prior rows by 0 
        out = torch.linalg.solve_triangular(L, B, upper=False)
        if not prior_mask_idx is None:
            P = L.shape[-1]
            mask_indexer = torch.arange(
                P, dtype=int, device=L.device
            ).expand(L.shape[:-1]) # [*batch_shape, P]
            out = out.masked_fill(
                ( mask_indexer >= prior_mask_idx.unsqueeze(-1) ).unsqueeze(-1), # [*batch_shape, P, 1]
                0
            )
        return out

    def compute_gaussian_entropy(self, cov_matrix, mask_idx=None):
        r"""
        :param cov_matrix: [*batch_shape, T, T] tensor
        :param mask_idx: None or [*batch_shape, 2] tensor of int
        :return: tensor of batch_shape, H = 1/2 * logdet( cov_matrix * 2*pi*e ) if mask_idx is None
            H = 1/2 * logdet( cov_matrix[mask_idx[...,0]::mask_idx[...,1], mask_idx[...,0]::mask_idx[...,1]] * 2*pi*e ) if mask_idx is given
        """
        # torch.linalg.cholesky seems numerically more stable than torch.logdet
        cholesky = self.compute_cholesky(cov_matrix)
        return self.compute_gaussian_entropy_from_cholesky(cholesky, mask_idx=mask_idx)

    def compute_gaussian_entropy_from_cholesky(self, L_cov_matrix, mask_idx=None):
        r"""
        :param L_cov_matrix: [*batch_shape, T, T] tensor, cholesky decomposition of cov_matrix
        :param mask_idx: None or [*batch_shape, 2] tensor of int
        :return: tensor of batch_shape, H = logdet( L * sqrt(2*pi*e) ) if mask_idx is None
            H = logdet( L[mask_idx[...,0]::mask_idx[...,1], mask_idx[...,0]::mask_idx[...,1]] * sqrt(2*pi*e) ) if mask_idx is given
        """
        eigenvalues_sqrt = L_cov_matrix.diagonal(dim1=-2, dim2=-1).log() + 1/2*math.log(2*math.pi*math.e)
        return self._sum_masked(eigenvalues_sqrt, mask_idx)

    def compute_gaussian_log_likelihood(self, Y, mu, cov_matrix, mask_idx=None):
        r"""
        :param Y: [*batch_shape, T] tensor
        :param mu: float or [*batch_shape, T] tensor
        :param cov_matrix: [*batch_shape, T, T] tensor
        :param mask_idx: None or [*batch_shape, 2] tensor of int
        :return: tensor of batch_shape, log N(Y|mu, cov_matrix) if mask_idx is None
            log N(Y[mask_idx[...,0]::mask_idx[...,1]]|mu[mask_idx[...,0]::mask_idx[...,1]], cov_matrix[mask_idx[...,0]::mask_idx[...,1],mask_idx[...,0]::mask_idx[...,1]])  if mask_idx is given
        """
        cholesky = self.compute_cholesky(cov_matrix)
        return self.compute_gaussian_log_likelihood_from_cholesky(Y, mu, cholesky, mask_idx=mask_idx)

    def compute_gaussian_log_likelihood_from_cholesky(self, Y, mu, L_cov_matrix, mask_idx=None):
        r"""
        :param Y: [*batch_shape, T] tensor
        :param mu: float or [*batch_shape, T] tensor
        :param L_cov_matrix: [*batch_shape, T, T] tensor, cholesky decomposition of cov_matrix
        :param mask_idx: None or [*batch_shape, 2] tensor of int
        :return: tensor of batch_shape, log N(Y|mu, cov_matrix) if mask_idx is None
            log N(Y[mask_idx[...,0]::mask_idx[...,1]]|mu[mask_idx[...,0]::mask_idx[...,1]], cov_matrix[mask_idx[...,0]::mask_idx[...,1],mask_idx[...,0]::mask_idx[...,1]])  if mask_idx is given
        """
        # L^-1 @ Y
        transformed_Y = self.compute_cholesky_inv_B( L_cov_matrix, (Y - mu).unsqueeze(-1) ).squeeze(-1)
        # sqrt(2 pi) L
        eigenvalues_sqrt = (L_cov_matrix * math.sqrt(2*math.pi) ).diagonal(dim1=-2, dim2=-1).log()

        # - log(|sqrt(2 pi) L|) - 1/2 <L^-1 @ Y, L^-1 @ Y>
        log_prob = - eigenvalues_sqrt - 1/2 * torch.pow(transformed_Y, 2)
        return self._sum_masked(log_prob, mask_idx)

    def compute_gaussian_process_posterior(
        self,
        cov_observation_matrix,
        cov_cross_matrix,
        cov_test=None,
        Y_observation=None,
        mean_prior=None,
        mean_test=None,
        prior_mask_idx=None,
        *,
        return_mu: bool=True,
        return_cov: bool=True
    ):
        r"""
        :param cov_observation_matrix: [*batch_shape, P, P] tensor
        :param cov_cross_matrix: float or [*batch_shape, P, T] tensor
        :param cov_test: [*batch_shape, T, T] tensor, can be None if return_cov is False
        :param Y_observation: [*batch_shape, P] tensor, can be None if return_mu is False
        :param mean_prior: [*batch_shape, P] tensor, can be None if return_mu is False or if we have zero mean
        :param mean_test: [*batch_shape, T] tensor, can be None if return_mu is False or if we have zero mean
        :param prior_mask_idx: None or [*batch_shape] tensor containing 1, ..., P. The integers indicate how many observations the posterior conditions on.
        :param return_mu: bool, return posterior mean or not
        :param return_cov: bool, return posterior covariance or not
        :return: (mu, cov), or mu, or cov
            mu: tensor of [*batch_shape, T], mean_test + cov_cross_matrix.T @ cov_observation_matrix^{-1} @ (Y_observation - mean_prior)
            cov: tensor of [*batch_shape, T, T], cov_test - cov_cross_matrix.T @ cov_observation_matrix^{-1} @ cov_cross_matrix
        """
        assert cov_cross_matrix.shape[-2] == cov_observation_matrix.shape[-2] == cov_observation_matrix.shape[-1], f'cov_cross_matrix: {cov_cross_matrix.shape}, cov_observation_matrix: {cov_observation_matrix.shape}'

        if not return_mu and not return_cov:
            raise ValueError("At least one of return_mu and return_cov must be True")

        else:
            assert cov_observation_matrix.shape[-1] == cov_observation_matrix.shape[-2] == cov_cross_matrix.shape[-2], f'cov_cross_matrix: {cov_cross_matrix.shape}, cov_observation_matrix: {cov_observation_matrix.shape}'
            cholesky = self.compute_cholesky(cov_observation_matrix) # [*batch_shape, P, P]
            return self.compute_gaussian_process_posterior_from_cholesky(
                cholesky,
                cov_cross_matrix,
                cov_test,
                Y_observation=Y_observation,
                mean_prior=mean_prior,
                mean_test=mean_test,
                prior_mask_idx=prior_mask_idx,
                return_mu=return_mu,
                return_cov=return_cov
            )

    def compute_gaussian_process_posterior_from_cholesky(
        self,
        L_cov_observation_matrix,
        cov_cross_matrix,
        cov_test=None,
        Y_observation=None,
        mean_prior=None,
        mean_test=None,
        prior_mask_idx=None,
        *,
        return_mu: bool=True,
        return_cov: bool=True
    ):
        r"""
        :param L_cov_observation_matrix: [*batch_shape, P, P] tensor, cholesky decomposition of cov_observation_matrix
        :param cov_cross_matrix: float or [*batch_shape, P, T] tensor
        :param cov_test: [*batch_shape, T, T] tensor, can be None if return_cov is False
        :param Y_observation: [*batch_shape, P] tensor, can None if return_mu is False
        :param mean_prior: [*batch_shape, P] tensor, can be None if return_mu is False or if we have zero mean
        :param mean_test: [*batch_shape, T] tensor, can be None if return_mu is False or if we have zero mean
        :param prior_mask_idx: None or [*batch_shape] tensor containing 1, ..., P. The integers indicate how many observations the posterior conditions on.
        :param return_mu: bool, return posterior mean or not
        :param return_cov: bool, return posterior covariance or not
        :return: (mu, cov), or mu, or cov
            mu: tensor of [*batch_shape, T], mean_test + cov_cross_matrix.T @ cov_observation_matrix^{-1} @ (Y_observation - mean_prior)
            cov: tensor of [*batch_shape, T, T], cov_test - cov_cross_matrix.T @ cov_observation_matrix^{-1} @ cov_cross_matrix
        """
        cholesky = L_cov_observation_matrix
        assert cov_cross_matrix.shape[-2] == cholesky.shape[-2] == cholesky.shape[-1], f'cov_cross_matrix: {cov_cross_matrix.shape}, cholesky: {cholesky.shape}'

        if not return_mu and not return_cov:
            raise ValueError("At least one of return_mu and return_cov must be True")

        else:
            assert cholesky.shape[-1] == cholesky.shape[-2] == cov_cross_matrix.shape[-2]
            K_right = self.compute_cholesky_inv_B(cholesky, cov_cross_matrix, prior_mask_idx=prior_mask_idx) # [*batch_shape, P, T]

            if return_mu:
                assert Y_observation is not None
                err = Y_observation if mean_prior is None else Y_observation - mean_prior
                mean_test = mean_test if not mean_test is None else \
                    torch.zeros([*cov_cross_matrix.shape[:-2], cov_cross_matrix.shape[-1]], device=cov_cross_matrix.device)
                transformed_Y_observation = self.compute_cholesky_inv_B( cholesky, err.unsqueeze(-1), prior_mask_idx ) # [*batch_shape, P, 1]

                mu_conditional = mean_test + \
                    torch.matmul(
                        K_right.transpose(-1, -2),
                        transformed_Y_observation
                    ).squeeze(-1) # [*batch_shape, T]

            if return_cov:
                assert cov_test is not None
                assert cov_test.shape[-1] == cov_test.shape[-2] == cov_cross_matrix.shape[-1], f'cov_cross_matrix: {cov_cross_matrix.shape}, cov_test: {cov_test.shape}'

                K_conditional = cov_test - torch.matmul(K_right.transpose(-1, -2), K_right) # [*batch_shape, T, T]

            if return_mu and not return_cov:
                return mu_conditional
            elif not return_mu and return_cov:
                return K_conditional
            else:
                return mu_conditional, K_conditional

    def _sum_masked(self, value_tensor, index_tensor=None):
        """
        value_tensor: [*batch_shape, T]
        index_tensor: None or [*batch_shape, 2] int tensor, values range [0, T] inclusive
        
        return value_tensor.sum(-1), but only at index index_tensor[..., 0](inclusive) til index_tensor[..., 1](exclusive)
        """
        if index_tensor is None:
            return value_tensor.sum(-1)
        else:
            value_sum = value_tensor.cumsum(dim=-1)
            mask_start = index_tensor[..., 0, None].expand(value_sum.shape[:-1] + (1,)) # [*batch_shape, 1]
            mask_end = index_tensor[..., 1, None].expand(value_sum.shape[:-1] + (1,)) # [*batch_shape, 1]
            out = torch.gather(
                value_sum, dim=-1, index=(mask_end-1).clamp(min=0) # no length indices will be excluded
            ) - torch.gather( value_sum, dim=-1, index=mask_start) + \
                torch.gather( value_tensor, dim=-1, index=mask_start) # [*batch_shape, 1]
            out = torch.where(
                mask_end<1, torch.zeros_like(value_sum[...,0, None]), # filter out no length indices
                out ).squeeze(-1)
            return out

