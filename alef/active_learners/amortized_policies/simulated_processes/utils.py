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
import numpy as np
import pandas as pd

import torch
import pyro
import pyro.distributions as dist
from typing import List
from alef.active_learners.amortized_policies.simulated_processes.gp_sampler.base_sampler import BaseSampler
from alef.active_learners.amortized_policies.utils.gp_computers import GaussianProcessComputer

def sample_safe_xyz(
    max_try: int,
    lower_bound: float,
    x_pdf: dist.Distribution,
    y_sampler: BaseSampler,
    z_sampler: BaseSampler,
):
    x_grid = x_pdf.sample()
    pdf_safety = z_sampler.y_sampler(x_grid)#.to_event(1)
    z_grid = pdf_safety.sample()
    constraint_satisfied = (z_grid >= lower_bound)
    n_try = 0
    while n_try <= max_try and not constraint_satisfied.all():
        x_new = x_pdf.sample()
        pdf_safety_new = z_sampler.y_sampler(x_new)#.to_event(1)
        z_new = pdf_safety_new.sample()
        new_constraint_satisfied = (z_new >= lower_bound)
        x_grid = torch.where(
            torch.logical_and(
                ~constraint_satisfied.unsqueeze(-1).expand(x_new.shape),
                new_constraint_satisfied.unsqueeze(-1).expand(x_new.shape)
            ),
            x_new,
            x_grid
        )
        z_grid = torch.where(
            torch.logical_and(
                ~constraint_satisfied,
                new_constraint_satisfied
            ),
            z_new,
            z_grid
        )
        constraint_satisfied = torch.logical_or(constraint_satisfied, new_constraint_satisfied)
        n_try += 1

    pdf = y_sampler.y_sampler(x_grid)#.to_event(1)
    y_grid = pdf.sample()

    return x_grid, y_grid, z_grid


def sample_high_determinant_x(
    max_try: int,
    previous_x: torch.Tensor,
    num_samples: int,
    x_pdf: dist.Distribution,
    kernel_list,
    noise_var_list,
):
    r"""
    make sure x_pdf is a dist of shape [B, Nk, Nf, 1, D]
    kernel_list has Nk kernels
    noise_var_list has Nk vars
    
    return list of num_samples x
    """
    assert num_samples >= 1
    gp_helper = GaussianProcessComputer()
    if previous_x is None:
        x0 = x_pdf.sample() # [B, Nk, Nf, 1, D]
        x = x0
        out_x = [x0]
        iterator = range(1, num_samples)
    else:
        x = previous_x
        out_x = []
        iterator = range(num_samples)
    L = gp_helper.compute_cholesky( # [B, Nk, Nf, num_x, num_x]
        gp_helper.compute_kernel_batch(
            kernel_list,
            x, x,
            noise_var_list,
            batch_dim=1,# Nf
        )
    )
    for t in iterator:
        set_max_already = False
        for _ in range(max_try):
            xt_try = x_pdf.sample() # [B, Nk, Nf, 1, D]
            K_cross = gp_helper.compute_kernel_batch( # [B, Nk, Nf, num_x, 1]
                kernel_list,
                x, xt_try,
                batch_dim=1,# Nf
            )
            K_new = gp_helper.compute_kernel_batch( # [B, Nk, Nf, 1, 1]
                kernel_list,
                xt_try,
                xt_try,
                noise_var_list,
                batch_dim=1,# Nf
            )
            L_new_try = gp_helper.compute_cholesky_update(L, K_cross, K_new) # [B, Nk, Nf, num_x + 1, num_x + 1]
            det_values = L_new_try[..., -1, -1] # [B, Nk, Nf]
            if not set_max_already:
                set_max_already = True
                xt = xt_try.clone()
                max_det_values = det_values.clone()
                L_new_max = L_new_try.clone()
            else:
                update_conditions = det_values > max_det_values # [B, Nk, Nf]
                xt = torch.where(
                    update_conditions[...,None,None].expand(xt_try.shape),
                    xt_try,
                    xt
                )
                max_det_values = torch.where(
                    update_conditions,
                    det_values.clone(),
                    max_det_values
                )
                L_new_max = torch.where(
                    update_conditions[...,None,None].expand(L_new_max.shape),
                    L_new_try.clone(),
                    L_new_max
                )

        x = torch.cat([x, xt], dim=-2)
        out_x.append(xt)
        L = L_new_max
    return torch.cat(out_x, dim=-2)
                

def sample_high_determinant_safe_xyz(
    max_try: int,
    lower_bound: float,
    previous_x: torch.Tensor,
    num_samples: int,
    x_pdf: dist.Distribution,
    y_sampler: BaseSampler,
    z_sampler: BaseSampler,
):
    r"""
    make sure x_pdf is a dist of shape [B, Nk, Nf, 1, D]
    
    return x [B, Nk, Nf, num_samples, D], y, z
    """
    assert num_samples >= 1
    gp_helper = GaussianProcessComputer()
    if previous_x is None:
        x0, y0, z0 = sample_safe_xyz(max_try, lower_bound, x_pdf, y_sampler, z_sampler)
        x = x0
        out_x = [x0]
        out_y = [y0]
        out_z = [z0]
        iterator = range(1, num_samples)
    else:
        x = torch.cat(previous_x, dim=-2)
        out_x = []
        out_y = []
        out_z = []
        iterator = range(num_samples)
    L = gp_helper.compute_cholesky( # [B, Nk, Nf, num_x, num_x]
        gp_helper.compute_kernel_batch(
            y_sampler.kernel_list,
            x, x,
            y_sampler.noise_variance,
            batch_dim=1,# Nf
        )
    )
    for t in iterator:
        set_values_already = False
        for _ in range(max_try):
            x_new = x_pdf.sample()
            pdf_safety_new = z_sampler.y_sampler(x_new)
            z_new = pdf_safety_new.sample()
            new_constraint_satisfied = (z_new >= lower_bound) # [B, Nk, Nf, 1]
            # get determinant
            K_cross = gp_helper.compute_kernel_batch( # [B, Nk, Nf, num_x, 1]
                y_sampler.kernel_list,
                x, x_new,
                batch_dim=1,# Nf
            )
            K_new = gp_helper.compute_kernel_batch( # [B, Nk, Nf, 1, 1]
                y_sampler.kernel_list,
                x_new,
                x_new,
                y_sampler.noise_variance,
                batch_dim=1,# Nf
            )
            L_new = gp_helper.compute_cholesky_update(L, K_cross, K_new) # [B, Nk, Nf, num_x + 1, num_x + 1]
            det_values = L_new[..., -1, -1] # [B, Nk, Nf]
            if not set_values_already:
                set_values_already = True
                xt = x_new.clone()
                zt = z_new.clone()
                constraint_satisfied = new_constraint_satisfied.clone()
                max_det_values = det_values.clone()
                L_new_max = L_new.clone()
            else:
                # for previous unsafe but currently safe, just update
                unsafe2safe_conds = torch.logical_and(
                    ~constraint_satisfied,
                    new_constraint_satisfied
                ) # [B, Nk, Nf, 1]
                # for previously and currently safe, compare determinant
                # this condition must be disjoint to unsafe2safe_conds
                safe2safe_conds = torch.logical_and(
                    constraint_satisfied,
                    new_constraint_satisfied
                ) # [B, Nk, Nf, 1]
                safe2safe_conds = torch.logical_and(
                    safe2safe_conds,
                    det_values.unsqueeze(-1) > max_det_values.unsqueeze(-1)
                )
                # so what will be updated: unsafe2safe or (safe2safe & det larger)
                update_condition = torch.logical_or(unsafe2safe_conds, safe2safe_conds)
                
                xt = torch.where( update_condition.unsqueeze(-1).expand(x_new.shape), x_new, xt )
                zt = torch.where( update_condition, z_new, zt )
                constraint_satisfied = torch.logical_or(constraint_satisfied, new_constraint_satisfied)
                max_det_values = torch.where( update_condition.squeeze(-1), det_values.clone(), max_det_values )
                L_new_max = torch.where( update_condition.unsqueeze(-1).expand(L_new.shape), L_new.clone(), L_new_max )

        pdf = y_sampler.y_sampler(xt)#.to_event(1)
        yt = pdf.sample()
        x = torch.cat([x, xt], dim=-2)
        out_x.append(xt)
        out_y.append(yt)
        out_z.append(zt)
        L = L_new_max

    return torch.cat(out_x, dim=-2), torch.cat(out_y, dim=-1), torch.cat(out_z, dim=-1)



