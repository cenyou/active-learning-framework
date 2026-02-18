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

import random
import copy
import torch
from torch.nn.parallel.replicate import replicate
import pyro
import pyro.distributions as dist

from typing import List, Union, Optional

from alef.configs.initial_data_conditions import INITIAL_OUTPUT_HIGHER_BY, INITIAL_OUTPUT_LOWER_BY
from alef.active_learners.amortized_policies.global_parameters import (
    OVERALL_VARIANCE,
    AL_MEAN_VARIANCE,
    AL_FUNCTION_VARIANCE_LOWERBOUND, AL_FUNCTION_VARIANCE_UPPERBOUND,
    SAFE_AL_MEAN_VARIANCE,
    SAFE_AL_FUNCTION_VARIANCE_LOWERBOUND, SAFE_AL_FUNCTION_VARIANCE_UPPERBOUND,
    SAFE_INTERVAL_SIZE
)
from alef.active_learners.amortized_policies.utils.oed_primitives import (
    observation_sample,
    latent_sample,
    prior_sample,
    compute_design,
)
from alef.active_learners.amortized_policies.simulated_processes.base_process import XPDF
from alef.active_learners.amortized_policies.simulated_processes.multiple_steps.base_multi_steps_process import BaseMultiStepsSimulatedProcess
from alef.active_learners.amortized_policies.simulated_processes.utils import sample_safe_xyz, sample_high_determinant_safe_xyz
# need GP model
from alef.configs.means.pytorch_means import BaseMeanPytorchConfig, BasicZeroMeanPytorchConfig
from alef.configs.kernels.pytorch_kernels.base_kernel_pytorch_config import BaseKernelPytorchConfig
from alef.enums.active_learner_amortized_policy_enums import LengthscaleDistribution
from alef.active_learners.amortized_policies.simulated_processes.gp_sampler.rff_sampler import RandomFourierFeatureSampler

"""
sample subtype overview: (Don't change this, otherwise baseline DAD loss goes wrong)
    hyper_prior_sample: GP hyperparameters
    prior_sample: Bayesian linear model
    latent_sample: num of initial data, sequence len
    design_sample: x
    observation_sample: y, z
"""

class SequentialSafeGaussianProcessContinuousDomain(BaseMultiStepsSimulatedProcess):

    def __init__(
        self,
        design_net,
        kernel_config: BaseKernelPytorchConfig,
        mean_config: Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]] = BasicZeroMeanPytorchConfig(batch_shape=[]),
        n_initial_min: int = 1,
        n_initial_max: Optional[int] = 1,
        n_steps_min: int = None,
        n_steps_max: int = 20,
        *,
        sample_gp_prior: bool = True,
        lengthscale_distribution: LengthscaleDistribution = LengthscaleDistribution.GAMMA,
        random_subsequence: bool = False,
        split_subsequence: bool = False,
        device: torch.device = torch.device("cpu"),
        safety_kernel_config: Optional[BaseKernelPytorchConfig] = None,
        safety_mean_config: Optional[Union[BaseMeanPytorchConfig, List[BaseMeanPytorchConfig]]] = None,
        **kwargs,
    ):
        """
        :param design_net: NN to run AL
        :param n_initial_min: int, min num of initial observations in the simulation
        :param n_initial_max: int, max num of initial observations in the simulation
        :param n_steps_min: int, min num of queries actually queried by NN
        :param n_steps_max: int, max num of queries actually queried by NN
        :param random_subsequence: bool, if we want NN to query random num
        :param split_subsequence: bool, if we want NN to maybe query one sequence first,
            then another sequence, the two sequences sum to to max n_steps_max steps.
            This matters when NN is budget aware.
            This flag is useless if random_subsequence==False
        """
        # make sure design_net is on the desired device
        super().__init__(design_net = design_net, n_initial_min=n_initial_min, n_initial_max=n_initial_max, n_steps_min=n_steps_min, n_steps_max=n_steps_max, random_subsequence=random_subsequence, split_subsequence=split_subsequence)
        self.gp_dist = RandomFourierFeatureSampler(
            kernel_config,
            0.1,
            overall_variance=OVERALL_VARIANCE,
            function_variance_interval=(
                AL_FUNCTION_VARIANCE_LOWERBOUND,
                AL_FUNCTION_VARIANCE_UPPERBOUND
            ),
            mean_variance=AL_MEAN_VARIANCE,
            mean_config=mean_config,
            lengthscale_distribution=lengthscale_distribution,
        )
        self.safety_gp_dist = RandomFourierFeatureSampler(
            kernel_config if safety_kernel_config is None else safety_kernel_config,
            0.1,
            overall_variance=OVERALL_VARIANCE,
            function_variance_interval=(
                SAFE_AL_FUNCTION_VARIANCE_LOWERBOUND,
                SAFE_AL_FUNCTION_VARIANCE_UPPERBOUND
            ),
            mean_variance=SAFE_AL_MEAN_VARIANCE,
            mean_config=mean_config if safety_mean_config is None else safety_mean_config,
            lengthscale_distribution=lengthscale_distribution,
        )
        self._sample_gp_prior = sample_gp_prior
        self.set_device(device)
        self.set_name('Default')

    def set_name(self, name: str):
        self.name = name
        self.gp_dist.name = f'{name}.y_sampler'
        self.safety_gp_dist.name = f'{name}.z_sampler'

    def set_device(self, device: torch.device, replicate_gp_priors: bool = False):
        self.design_net.to(device)
        if replicate_gp_priors:
            self.gp_dist = self.gp_dist.clone_module()
            self.safety_gp_dist = self.safety_gp_dist.clone_module()
        self.gp_dist.set_device(device)
        self.safety_gp_dist.set_device(device)
        self.device = device
        self.set_name(f'Device_{device.index}')

    def replicate(self, device_ids, detach=False):
        design_net_list = replicate(self.design_net, device_ids, detach)
        new_list = []
        for d, new_net in enumerate(design_net_list):
            new = self.__new__(type(self))
            super(SequentialSafeGaussianProcessContinuousDomain, new).__init__(
                new_net,
                self.n_initial_min,
                self.n_initial_max,
                self.n_steps_min,
                self.n_steps_max
            )
            new.gp_dist = self.gp_dist.clone_module()
            new.safety_gp_dist = self.safety_gp_dist.clone_module()
            new._sample_gp_prior = self._sample_gp_prior
            new.set_device(device_ids[d])
            #new.set_name(f'Device_{d}')
            new_list.append(new)
        return new_list

    def global_variables_in_dict(self):
        y_dict = self.gp_dist.global_variables_in_dict(name='y_sampler')
        z_dict = self.safety_gp_dist.global_variables_in_dict(name='z_sampler')
        out = {**y_dict, **z_dict}
        #out['initial_observations'] = f'z >= {INITIAL_OUTPUT_HIGHER_BY}'
        out['initial_observations'] = 'z >= 0'
        out['safe_set_size'] = SAFE_INTERVAL_SIZE
        return out

    def process(
        self,
        batch_size: int=1,
        num_kernels: int=1,
        num_functions: int=1,
        sample_domain_grid_points: bool=False,
        constraint_domain_grid_points: bool=False,
        num_grid_points: int=100,
        device: Optional[torch.device]=None
    ):
        r"""
        Generate a sequence of data. Each data point is an input-output pair,
        where inputs are experimental designs xi and outputs are observations y.
        This class provide observations sampled from GP simulators.

        :param batch_size: batch size.
        :param num_kernels: batch size of kernel hyperparameters (i.e. num of kernels).
        :param num_functions: number of functional sample given a GP prior.
        :param sample_domain_grid_points: if True, return some domain samples
                and their corresponding y values which are jointly GP with y
        :param constraint_domain_grid_points: if True, the generated domain samples are constrained (z_grid >= 0)
        :param num_grid_points: number of grid points per batch
        :param device: device to run the simulation on
        :return:
            simulation main output mean,
            simulation main output kernel,
            simulation main output noise_var,
            simulation safety output mean,
            simulation safety output kernel,
            simulation safety output noise_var,
            dimension_mask [B, Nk, Nf, max_D],  ---> mask of the sampled D
            n_init [B, Nk, Nf],  ---> the sampled num of initial data
            n_query1 [B, Nk, Nf],  ---> num of queries prestart (will be 0 unless we use NN to query 2 seqs)
            n_query2 [B, Nk, Nf],  ---> the length of policy queries
            X_init, Y_init, Z_init,
            X_query, Y_query, Z_query,
            (X_grid, Y_grid, Z_grid)
        """
        assert num_kernels > 0
        assert num_functions > 0

        if device is not None:
            self.set_device(device, replicate_gp_priors=True)

        if hasattr(self.design_net, "parameters"):
            #! this is required for the pyro optimizer
            pyro.module("design_net", self.design_net)
        #######################################################################
        ### configurations
        #######################################################################
        n_init_l = self.n_initial_min
        n_init_u = self.n_initial_max
        T_run = self.n_steps_max
        B, Nk, Nf = batch_size, num_kernels, num_functions
        with torch.no_grad():
            if self.flexible_dimension:
                max_D = self.gp_dist.kernel.input_dimension
                dim_dist = self.random_subsequence_helper.index_distribution(
                    1, (max_D + 1)*torch.ones([Nk, Nf], dtype=int, device=self.device)
                ) # sample range [1, max_D + 1) or equal to 1
                D = latent_sample(f"{self.name}_dimension", dim_dist ) # [Nk, Nf]
                mask_dim = self.random_subsequence_helper.index_range2mask(
                    0, D, max_D
                ) # [Nk, Nf, max_D]
            else:
                max_D = self.gp_dist.kernel.input_dimension
                D = max_D
                mask_dim = None
        ########################################################################
        # Sample GPs
        ########################################################################
        self.gp_dist.draw_parameter(
            num_priors=num_kernels,
            num_functions=num_functions,
            draw_hyper_prior=self._sample_gp_prior,
            input_domain=self.input_domain,
            shift_mean=True,
            mask=mask_dim,
            sample_executor=prior_sample,
        )
        self.safety_gp_dist.draw_parameter(
            num_priors=num_kernels,
            num_functions=num_functions,
            draw_hyper_prior=self._sample_gp_prior,
            input_domain=self.input_domain,
            shift_mean=True,
            flip_center=False,
            center_length_ratio=SAFE_INTERVAL_SIZE, # this is useless
            mask=mask_dim,
            sample_executor=prior_sample,
        )
        with torch.no_grad(): # need this later
            mean_list = self.gp_dist.mean_list
            kernel_list = self.gp_dist.kernel_list
            noise_var = self.gp_dist.noise_variance
            mean_list_safety = self.safety_gp_dist.mean_list
            kernel_list_safety = self.safety_gp_dist.kernel_list
            noise_var_safety = self.safety_gp_dist.noise_variance
        #######################################################################
        ### sample initial data
        ### add few more queries not obtained by policy but just samplers
        ### consider this as flex num of initial points
        #######################################################################
        with torch.no_grad():
            _n_init_up = n_init_u if self.random_subsequence else n_init_l
            n_init_dist = self.random_subsequence_helper.index_distribution(
                n_init_l, (_n_init_up + 1) * torch.ones([B, Nk, Nf], device=self.device, dtype=int)
            ) # sample range [0, n_init_u - n_init + 1)
            n_init = latent_sample(f"{self.name}_n_init", n_init_dist )
            X_init, Y_init, Z_init = self._sample_initial_observations(B, Nk, Nf, n_init.max(), self.gp_dist, self.safety_gp_dist)
        #######################################################################
        ### sample T ~ Uniform[1, T_max]
        #######################################################################
        with torch.no_grad():
            # sequence of {y_i} should be like this
            # i= 0, ..., n_init - 1 (initial points)
            # i= n_init, ..., n_init + t1 - 1 (NN forward, 1st sequence, budget {-(i - n_init) + t1, ..., 1} )
            # i= n_init + t1, ..., n_init + t1 + t2 - 1 (NN forward, 2nd sequence, budget {-(i - n_init) + t1 + t2, ..., 1} )
            # t1 + t2 <= T
            _offset = torch.zeros_like(n_init, dtype=int)
            t1_dist = self.random_subsequence_helper.index_distribution(
                _offset, ( _offset + T_run if self.random_subsequence and self.split_subsequence else _offset + 1 )
            ) # sample range [_offset, _offset + n_steps_max) or equal to _offset
            t1 = latent_sample(f"{self.name}_t1", t1_dist )

            t2_dist = self.random_subsequence_helper.index_distribution(
                (1 if self.random_subsequence else T_run - t1), T_run - t1 + 1
            ) # sample range [1, n_steps_max - t1 + 1) or equal to n_steps_max - t1
            t2 = latent_sample(f"{self.name}_t2", t2_dist )

        X = None if X_init is None else X_init
        Y = None if Y_init is None else Y_init
        Z = None if Z_init is None else Z_init
        X_query, Y_query, Z_query = None, None, None
        mask_seq = None if Y_init is None or torch.all(n_init==n_init_l) else self.random_subsequence_helper.index_range2mask(0, n_init, Y_init.shape[-1])
        mask_dim= None if mask_dim is None else mask_dim.unsqueeze(0).expand((B, Nk, Nf, max_D))

        for t in range(T_run):
            ####################################################################
            # Get a design x
            ####################################################################
            if self.design_net.forward_with_budget:
                budget = torch.where(
                    t >= t1,
                    -t + t1 + t2, # 2nd sequence
                    -t + t1 # 1st sequence
                ).unsqueeze(-1).to(torch.get_default_dtype())
                NN_forward_dist = self.design_net.lazy(
                    budget, X, Y, Z,
                    mask_sequence=mask_seq,
                    mask_feature=mask_dim
                ).expand((B, Nk, Nf))
            else:
                NN_forward_dist = self.design_net.lazy(
                    X, Y, Z,
                    mask_sequence=mask_seq,
                    mask_feature=mask_dim
                ).expand((B, Nk, Nf))
            query = compute_design(
                f"{self.name}_x{t + 1}", NN_forward_dist
            )# should have batch size [batch_size_for_each_function, num_kernels, num_functions, 1 (T placeholder), D]
            assert not torch.isnan(query).any(), 'this should not happen, typically occurs when gradient explodes in the previous update'
            ####################################################################
            # Sample y, z
            ####################################################################
            pdf = self.gp_dist.y_sampler(query)
            pdf_safety = self.safety_gp_dist.y_sampler(query)
            # [batch_size_for_each_function, num_kernels, num_functions, 1 (T placeholder)]
            y_query = observation_sample(f"{self.name}_y{t + 1}", pdf.to_event(1))
            z_query = observation_sample(f"{self.name}_z{t + 1}", pdf_safety.to_event(1))
            with torch.no_grad():
                mask_seq = torch.cat([
                    mask_seq,
                    torch.ones([B, Nk, Nf, 1], dtype=int, device=self.device)
                ], dim=-1) if not mask_seq is None else None
                X_query = torch.cat([X_query, query], dim=-2) if not X_query is None else query
                Y_query = torch.cat([Y_query, y_query], dim=-1) if not Y_query is None else y_query
                Z_query = torch.cat([Z_query, z_query], dim=-1) if not Z_query is None else z_query
                X = torch.cat([X, query], dim=-2) if not X is None else query
                Y = torch.cat([Y, y_query], dim=-1) if not Y is None else y_query
                Z = torch.cat([Z, z_query], dim=-1) if not Z is None else z_query

        if sample_domain_grid_points:
            X_grid, Y_grid, Z_grid = self._sample_grid_observations(
                B, Nk, Nf, num_grid_points, constraint_domain_grid_points,
                XPDF.BETA, self.gp_dist, self.safety_gp_dist
            )
            return (
                mean_list, kernel_list, noise_var,
                mean_list_safety, kernel_list_safety, noise_var_safety,
                mask_dim,
                n_init, t1, t2,
                X_init, Y_init, Z_init,
                X_query, Y_query, Z_query,
                X_grid, Y_grid, Z_grid,
            )
        else:
            return (
                mean_list, kernel_list, noise_var,
                mean_list_safety, kernel_list_safety, noise_var_safety,
                mask_dim,
                n_init, t1, t2,
                X_init, Y_init, Z_init,
                X_query, Y_query, Z_query,
            )

    def validation(
        self,
        batch_size: int=1,
        num_kernels: int=1,
        num_functions: int=1,
        num_test_points: int=100,
        device: Optional[torch.device]=None
    ):
        r"""
        Generate a sequence of data. Each data point is an input-output pair,
        where inputs are experimental designs xi and outputs are observations y.
        This class provide observations sampled from GP simulators.

        :param batch_size: batch size.
        :param num_kernels: batch size of kernel hyperparameters (i.e. num of kernels).
        :param num_functions: number of functional sample given a GP prior.
        :param num_grid_points: number of grid points per batch
        :param device: device to run the simulation on
        :return:
            simulation main output mean,
            simulation main output kernel,
            simulation main output noise_var,
            simulation safety output mean,
            simulation safety output kernel,
            simulation safety output noise_var,
            dimension_mask [B, Nk, Nf, max_D],
            n_init [B, Nk, Nf],
            X, Y, Z,
            X_test, Y_test, Z_test
        """
        assert num_kernels > 0
        assert num_functions > 0
        self.design_net.eval()

        if device is not None:
            self.set_device(device, replicate_gp_priors=True)
        #######################################################################
        ### configurations
        #######################################################################
        n_init = self.n_initial_min
        n_steps = self.n_steps_max
        if self.random_subsequence:
            n_steps += + self.n_initial_max - n_init
        B, Nk, Nf = batch_size, num_kernels, num_functions
        with torch.no_grad():
            if self.flexible_dimension:
                max_D = self.gp_dist.kernel.input_dimension
                dim_dist = self.random_subsequence_helper.index_distribution(
                    1, (max_D + 1)*torch.ones([Nk, Nf], dtype=int, device=self.device)
                ) # sample range [1, max_D + 1) or equal to 1
                D = latent_sample(f"{self.name}_dimension", dim_dist ) # [Nk, Nf]
                mask_dim = self.random_subsequence_helper.index_range2mask(
                    0, D, max_D
                ) # [Nk, Nf, max_D]
            else:
                max_D = self.gp_dist.kernel.input_dimension
                D = max_D
                mask_dim = None
        ########################################################################
        # Sample GPs
        ########################################################################
        self.gp_dist.draw_parameter(
            num_priors=num_kernels,
            num_functions=num_functions,
            draw_hyper_prior=self._sample_gp_prior,
            input_domain=self.input_domain,
            shift_mean=True,
            mask=None,
            sample_executor=prior_sample,
        )
        self.safety_gp_dist.draw_parameter(
            num_priors=num_kernels,
            num_functions=num_functions,
            draw_hyper_prior=self._sample_gp_prior,
            input_domain=self.input_domain,
            shift_mean=True,
            flip_center=False,
            center_length_ratio=SAFE_INTERVAL_SIZE,
            mask=None,
            sample_executor=prior_sample,
        )
        with torch.no_grad(): # need this later
            mean_list = self.gp_dist.mean_list
            kernel_list = self.gp_dist.kernel_list
            noise_var = self.gp_dist.noise_variance
            mean_list_safety = self.safety_gp_dist.mean_list
            kernel_list_safety = self.safety_gp_dist.kernel_list
            noise_var_safety = self.safety_gp_dist.noise_variance
        #######################################################################
        ### start sampling
        #######################################################################
        X, Y, Z = self._sample_initial_observations(B, Nk, Nf, n_init, self.gp_dist, self.safety_gp_dist)
        mask_dim= None if mask_dim is None else mask_dim.unsqueeze(0).expand((B, Nk, Nf, max_D))

        for i in range(n_steps):
            ####################################################################
            # Get a design xi
            ####################################################################
            if self.design_net.forward_with_budget:
                NN_forward_dist = self.design_net.lazy(
                    -i + n_steps*torch.ones([B, Nk, Nf, 1], dtype=torch.get_default_dtype(), device=self.device),
                    X, Y, Z,
                    mask_feature=mask_dim
                ).expand((B, Nk, Nf))
            else:
                NN_forward_dist = self.design_net.lazy(
                    X, Y, Z,
                    mask_feature=mask_dim
                ).expand((B, Nk, Nf))
            query = compute_design(
                f"{self.name}_x{i + 1}", NN_forward_dist
            )# xi should have batch size [batch_size_for_each_function, num_kernels, num_functions, 1 (T placeholder), D]
            assert not torch.isnan(query).any(), 'this should not happen, typically occurs when gradient explodes in the previous update'
            ####################################################################
            # Sample y, z
            ####################################################################
            pdf = self.gp_dist.y_sampler(query)
            pdf_safety = self.safety_gp_dist.y_sampler(query)
            # [batch_size_for_each_function, num_kernels, num_functions, 1 (T placeholder)]
            y_query = observation_sample(f"{self.name}_y{i + 1}", pdf.to_event(1))
            z_query = observation_sample(f"{self.name}_z{i + 1}", pdf_safety.to_event(1))
            with torch.no_grad():
                X = torch.cat([X, query], dim=-2) if not X is None else query
                Y = torch.cat([Y, y_query], dim=-1) if not Y is None else y_query
                Z = torch.cat([Z, z_query], dim=-1) if not Z is None else z_query

        X_grid, Y_grid, Z_grid = self._sample_grid_observations(
            B, Nk, Nf, num_test_points, True,
            XPDF.UNIFORM, self.gp_dist, self.safety_gp_dist
        )
        self.design_net.train()
        return (
            mean_list,
            kernel_list,
            noise_var,
            mean_list_safety,
            kernel_list_safety,
            noise_var_safety,
            mask_dim,
            n_init*torch.ones([B, Nk, Nf], dtype=int, device=self.device),
            n_steps*torch.ones([B, Nk, Nf], dtype=int, device=self.device),
            X, Y, Z,
            X_grid, Y_grid, Z_grid
        )

    # each sampling sub block
    def _sample_initial_observations(
        self,
        batch_size: int,
        num_kernels: int,
        num_functions: int,
        num_initial_data: int,
        gp_dist, # typically self.gp_dist
        safety_gp_dist, # typically self.gp_dist
    ):
        n_init = num_initial_data
        D = gp_dist.kernel.input_dimension
        B, Nk, Nf = batch_size, num_kernels, num_functions
        
        if n_init > 0:
            with torch.no_grad():
                # pyro.sample cannot handle constrained sampling up to now
                # the following 6 lines are here just for easier pytest control
                safe_intervals = safety_gp_dist.central_interval(# safety_gp_dist.max_interval(
                    input_domain=self.input_domain,
                    length_ratio=SAFE_INTERVAL_SIZE
                ) # [Nk, D, 2]
                safe_intervals = safe_intervals[:, None, None, ...].expand((1, Nk, Nf, n_init, D, 2))
                x_pdf = dist.Uniform(safe_intervals[..., 0], safe_intervals[..., 1]).expand((1, Nk, Nf, n_init, D))
                X = compute_design(f"{self.name}_x_init", x_pdf.to_event(2))
                pdf = gp_dist.y_sampler(X)
                pdf_safety = safety_gp_dist.y_sampler(X)
                Y = observation_sample(f"{self.name}_y_init", pdf.to_event(1))
                Z = observation_sample(f"{self.name}_z_init", pdf_safety.to_event(1))
                #if (z < INITIAL_OUTPUT_HIGHER_BY).any(): # initial data take higher thresholds (smaller starting region)
                if (Z < 0).any():
                    # this if is almost always true in a real pipeline,
                    # the above 5 lines are just for easier pytest control
                    X, Y, Z = sample_safe_xyz(
                        max_try=50,
                        #lower_bound=INITIAL_OUTPUT_HIGHER_BY,
                        lower_bound=0,
                        x_pdf=x_pdf,
                        y_sampler=self.gp_dist,
                        z_sampler=self.safety_gp_dist
                    )
                X = X.expand((B, Nk, Nf, n_init, D))
                Y = Y.expand((B, Nk, Nf, n_init))
                Z = Z.expand((B, Nk, Nf, n_init))
        else:
            X = None
            Y = None
            Z = None
        return X, Y, Z

    def _sample_grid_observations(
        self,
        batch_size: int,
        num_kernels: int,
        num_functions: int,
        num_grid_points: int,
        constraint_domain_grid_points: bool,
        x_dist: XPDF, # 'uniform' or 'beta'
        gp_dist, # typically self.gp_dist
        safety_gp_dist, # typically self.gp_dist
    ):
        D = gp_dist.kernel.input_dimension
        B, Nk, Nf = batch_size, num_kernels, num_functions
        with torch.no_grad():
            Ng = num_grid_points
            x_pdf = self._get_x_pdf(x_dist, [B, Nk, Nf], [Ng, D], self.input_domain)

            if not constraint_domain_grid_points:
                X_grid = x_pdf.sample()
                pdf = gp_dist.y_sampler(X_grid)
                pdf_safety = safety_gp_dist.y_sampler(X_grid)
                Y_grid = pdf.sample()
                Z_grid = pdf_safety.sample()
            else:
                X_grid, Y_grid, Z_grid = sample_safe_xyz(max_try=50, lower_bound=0.0, x_pdf=x_pdf, y_sampler=gp_dist, z_sampler=safety_gp_dist)
        return X_grid, Y_grid, Z_grid


class PytestSequentialSafeGaussianProcessContinuousDomain(SequentialSafeGaussianProcessContinuousDomain):
    
    def _sample_grid_observations(
        self,
        batch_size: int,
        num_kernels: int,
        num_functions: int,
        num_grid_points: int,
        constraint_domain_grid_points: bool,
        x_dist: XPDF, # 'uniform' or 'beta'
        gp_dist, # typically self.gp_dist
        safety_gp_dist, # typically self.gp_dist
    ):
        D = gp_dist.kernel.input_dimension
        B, Nk, Nf = batch_size, num_kernels, num_functions
        X_grid = torch.cat([
            torch.zeros([B, Nk, Nf, 1, D], device=self.device),
            torch.ones([B, Nk, Nf, 1, D], device=self.device),
        ], dim=-2)
        Y_grid = torch.ones([B, Nk, Nf, 2], device=self.device)
        Z_grid = torch.ones([B, Nk, Nf, 2], device=self.device)
        return X_grid, Y_grid, Z_grid



