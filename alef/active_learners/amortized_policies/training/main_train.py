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

import os
import argparse
import json
import datetime
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

import torch
from pyro.infer.util import torch_item
import pyro
from tqdm import trange, tqdm

from alef.active_learners.amortized_policies.training.training_factory import AmortizedLearnerTrainingFactory
from alef.active_learners.amortized_policies.losses.curriculum import BaseCurriculum
from alef.active_learners.amortized_policies.utils.utils import check_safety
from alef.configs.active_learners.amortized_policies import loss_configs
from alef.configs.active_learners.amortized_policies import training_configs
from alef.configs.active_learners.amortized_policies.policy_configs import (
    ContinuousGPPolicyConfig,
    SafetyAwareContinuousGPPolicyConfig,
    ContinuousGPFlexDimPolicyConfig,
    SafetyAwareContinuousGPFlexDimPolicyConfig,
)

# need GP model
from alef.configs.means.pytorch_means import BasicZeroMeanPytorchConfig, BasicSechMeanPytorchConfig, BasicSechRotatedMeanPytorchConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
from alef.configs.kernels.pytorch_kernels.rbf_kernels_for_hpfs_pytorch_configs import HighPressureFluidSystemRBFPytorchConfig
from alef.configs.means.pytorch_means.sech_mean_for_hpfs_pytorch_configs import HighPressureFluidSystemSechRotatedMeanPytorchConfig
from alef.enums.active_learner_amortized_policy_enums import DomainWarpperType, LengthscaleDistribution
from alef.utils.utils import string2bool, write_dict_to_json
from alef.enums.gpytorch_enums import GPytorchPriorEnum

from alef.configs.paths import EXPERIMENT_PATH

class TrainingProcesser:
    def __init__(
        self,
        log_path,
        seed: int,
        device: str,
        fast_tqdm: bool,
    ):
        self.root_path = Path(log_path) # .../ContinuousGP2DPolicy_loss_*/lr*
        #time_stamp = datetime.datetime.now().astimezone().strftime('%Z_%Y_%m_%d__%H_%M_%S')
        time_stamp = datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
        self.root_path = self.root_path.parent / (
            self.root_path.name + ( '_seed%s_%s'%(seed, time_stamp) )
        ) # .../ContinuousGP2DPolicy_loss_*/lr*_seed*_[datetime]
        self.root_path.mkdir(exist_ok=True, parents=True)
        
        self.params_path = self.root_path / 'params'
        self.params_path.mkdir(exist_ok=True)
        self.result_path = self.root_path / 'result'
        self.result_path.mkdir(exist_ok=True)

        self.seed = seed
        self.device = device
        self.fast_tqdm = fast_tqdm

        """
        if device.startswith('cuda'):
            n_devices = torch.cuda.device_count()
            for d in range(n_devices):
                # warm up each GPU
                _ = torch.tensor([], device=d)
                _ = torch.linalg.cholesky(torch.ones((0, 0), device=d))
                logger.info(f'warm up GPU {d}')
        """

    def set_policy_config(
        self,
        input_dimension:int,
        domain_warpper: DomainWarpperType,
        safety_aware: bool=False,
        budget_aware: bool=False,
        resume_policy_path: str=None,
        flexible_dimension: bool=False,
        **kwargs
    ):
        if flexible_dimension:
            if safety_aware:
                self.policy_config = SafetyAwareContinuousGPFlexDimPolicyConfig(
                    forward_with_budget = budget_aware,
                    domain_warpper = domain_warpper,
                    device = self.device,
                    resume_policy_path = resume_policy_path,
                    **kwargs
                )
            else:
                self.policy_config = ContinuousGPFlexDimPolicyConfig(
                    forward_with_budget = budget_aware,
                    domain_warpper = domain_warpper,
                    device = self.device,
                    resume_policy_path = resume_policy_path,
                    **kwargs
                )
        else:
            if safety_aware:
                self.policy_config = SafetyAwareContinuousGPPolicyConfig(
                    input_dim = input_dimension,
                    observation_dim = 1,
                    safety_dim = 1,
                    forward_with_budget = budget_aware,
                    domain_warpper = domain_warpper,
                    device = self.device,
                    resume_policy_path = resume_policy_path,
                    **kwargs
                )
            else:
                self.policy_config = ContinuousGPPolicyConfig(
                    input_dim = input_dimension,
                    observation_dim = 1,
                    forward_with_budget = budget_aware,
                    domain_warpper = domain_warpper,
                    device = self.device,
                    resume_policy_path = resume_policy_path,
                    **kwargs
                )

    def set_loss_config(self, loss_config_name: str, **kwargs):
        self.loss_config = getattr(loss_configs, loss_config_name)(**kwargs)

    def set_training_config(
        self, training_config_name: str,
        kernel_config,
        mean_config_list,
        n_initial: Tuple[int, int],
        n_steps: Tuple[int, int],
        random_subsequence: bool=False,
        split_subsequence: bool = False,
        *,
        lengthscale_distribution = LengthscaleDistribution.GAMMA,
        safety_kernel_config = None,
        safety_mean_config_list = None,
        **kwargs
    ):
        self.kernel_config = kernel_config
        self.mean_config_list = mean_config_list
        self.safety_kernel_config = safety_kernel_config
        self.safety_mean_config_list = safety_mean_config_list

        self.training_config = getattr(training_configs, training_config_name)(
            policy_config = self.policy_config,
            kernel_config = self.kernel_config,
            mean_config = self.mean_config_list,
            n_initial_min = n_initial[0],
            n_initial_max = n_initial[1],
            n_steps_min = n_steps[0],
            n_steps_max = n_steps[1],
            lengthscale_distribution = lengthscale_distribution,
            random_subsequence = random_subsequence, # if True, batch with different num of queries, max T_simulation
            split_subsequence = split_subsequence,
            loss_config = self.loss_config,
            safety_kernel_config = self.safety_kernel_config, # maybe None
            safety_mean_config = self.safety_mean_config_list, # maybe None
            **kwargs
        )

    def save_settings(self):
        print(f'Store configs: {self.params_path}')
        write_dict_to_json(
            json.loads( self.policy_config.json() ),
            self.params_path / f'policy_config_{self.policy_config.__class__.__name__}.json'
        )
        write_dict_to_json(
            json.loads( self.loss_config.json() ),
            self.params_path / f'loss_config_{self.loss_config.__class__.__name__}.json'
        )
        write_dict_to_json(
            json.loads( self.kernel_config.json() ),
            self.params_path / f'kernel_config_{self.kernel_config.__class__.__name__}.json'
        )
        if not self.safety_kernel_config is None:
            write_dict_to_json(
                json.loads( self.safety_kernel_config.json() ),
                self.params_path / f'safety_kernel_config_{self.safety_kernel_config.__class__.__name__}.json'
            )
        for i, config in enumerate(self.mean_config_list):
            write_dict_to_json(
                json.loads( config.json() ),
                self.params_path / f'mean_config_{i}_{config.__class__.__name__}.json'
            )
        if not self.safety_mean_config_list is None:
            for i, config in enumerate(self.safety_mean_config_list):
                write_dict_to_json(
                    json.loads( config.json() ),
                    self.params_path / f'safety_mean_config_{i}_{config.__class__.__name__}.json'
                )
        write_dict_to_json(
            json.loads( self.training_config.json(exclude={'policy_config', 'mean_config', 'kernel_config', 'safety_mean_config', 'safety_kernel_config', 'loss_config'}) ),
            self.params_path / f'training_config_{self.training_config.__class__.__name__}.json'
        )
        write_dict_to_json(
            self.oed.process.global_variables_in_dict(),
            self.params_path / 'global_parameters.json'
        )

    def train(self):
        torch.cuda.empty_cache()
        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        assert hasattr(self, 'policy_config')
        assert hasattr(self, 'loss_config')
        assert hasattr(self, 'training_config')

        self.oed = AmortizedLearnerTrainingFactory.build(self.training_config)
        assert isinstance(self.oed.loss, BaseCurriculum)
        self.save_settings() # some global parameters are loded from self.oed.process

        self.loss_history = []
        self.check_point_overview = pd.DataFrame(columns=['loss', 'epoch_mean_loss', 'rmse_mean', 'rmse_stderr'])

        loss_str = 'Loss (epoch mean):   0.000 '
        if self.fast_tqdm:
            t = trange(1, self.oed.loss.num_steps + 1, desc='Epoch %4d, %s'%(0, loss_str))
        else:
            t = trange(1, self.oed.loss.num_steps + 1, desc='Epoch %4d, %s'%(0, loss_str), mininterval=600, maxinterval=1800)

        total_params = sum(p.numel() for p in self.oed.process.design_net.parameters() if p.requires_grad)
        print(f'we have \'{total_params}\' parameters.')

        for _ in t:
            loss = self.oed.step()
            loss = torch_item(loss)
            self.loss_history.append(loss)
            # Log every few losses -> too slow (and unnecessary to log everything)
            # the loss idx is added by 1 already when we call self.oed.step()
            if self.oed.loss.epoch_idx % 5 == 0 and self.oed.loss.step_per_epoch_idx == 0:
                #loss_str = 'Loss: {:3.3f} '.format(loss)
                loss_str = 'Loss (epoch mean): {:3.3f}'.format(self.check_point_overview.iloc[-1, 1])
                t.set_description('Epoch %4d, %s'%(self.oed.loss.epoch_idx, loss_str))
            # Decrease LR at every new epoch
            if self.oed.loss.step_per_epoch_idx == 0: # the loss idx is added by 1 already when we call self.oed.step()
                self.oed.optim.step()
                # evaluate in the end of each epoch
                epoch_size = self.oed.loss.step_idx if self.oed.loss.epoch_idx==1 else \
                    self.oed.loss.step_idx - self.check_point_overview.index[-1]
                with torch.no_grad():
                    rmse_mean, rmse_stderr = self.oed.validation()
                self.check_point_overview.loc[self.oed.loss.step_idx] = [
                    loss,
                    np.mean(self.loss_history[-epoch_size:]),
                    torch_item(rmse_mean),
                    torch_item(rmse_stderr)
                ]
            # Log model every 10 epochs
            # the loss idx is added by 1 already when we call self.oed.step()
            if self.oed.loss.epoch_idx > 0 and \
            self.oed.loss.epoch_idx % 20 == 0 and \
            self.oed.loss.step_per_epoch_idx == 0 and \
            self.oed.loss.step_idx < self.oed.loss.num_steps - 1:
                self.save_checkpoint(self.oed.loss.step_idx)

        self.check_point_overview = self.check_point_overview.rename(index={self.oed.loss.step_idx: 'final'})

        # evaluate and store results
        print(f"Training completed.")

        self.save_training_results()

        return True

    def save_checkpoint(self, idx: int):
        # Log model
        tqdm.write(f"Storing checkpoint_{idx}... ", end="") # print without breaking tqdm
        # store the model:
        torch.save(
            self.oed.process.design_net.state_dict(),
            self.result_path / f'model_checkpoint_{idx}.pth'
        )
        torch.save(
            self.oed.optim.get_state(),
            self.result_path / f'optimizer_scheduler_checkpoint_{idx}.pth'
        )
        tqdm.write(f"Checkpoint logged in {self.result_path}.") # print without breaking tqdm

    def save_training_results(self):
        # Log model
        print("Storing model... ", end="")
        # store the model:
        torch.save(
            self.oed.process.design_net.state_dict(),
            self.result_path / "model_checkpoint_final.pth"
        )
        torch.save(
            self.oed.optim.get_state(),
            self.result_path / "optimizer_scheduler_checkpoint_final.pth"
        )
        # store losses
        with pd.ExcelWriter(self.root_path / 'check_point_overview.xlsx', mode='w') as writer:
            self.check_point_overview.to_excel(writer, sheet_name='training_overview')

        write_dict_to_json(
            {'loss': self.loss_history},
            self.result_path / "loss.json"
        )
        write_dict_to_json(
            {'loss_first50': self.loss_history[:50], 'loss_last50': self.loss_history[-50:]},
            self.result_path / "loss_50.json"
        )
        print(f"Model sotred in {self.result_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser( description = "Amortized GP AL." )
    parser.add_argument("--experiment_output_dir", default=(EXPERIMENT_PATH / 'amortized_AL').as_posix(), type=str)
    parser.add_argument("--seed", default=[0], type=int, nargs='+')
    parser.add_argument("--train_smoother_pattern", default=False, type=string2bool)
    parser.add_argument("--train_high_pressure_fluid_system", default=False, type=string2bool)
    parser.add_argument("--flexible_dimension", default=False, type=string2bool)
    parser.add_argument("--dimension", default=5, type=int, help="if flexible_dimension=False, this is policy dim, if flexible_dimension=True, this is max policy dim")
    parser.add_argument("--training_config", default='AmortizedContinuousRandomGPPolicyTrainingConfig', type=str, choices=training_configs.__all__)
    parser.add_argument('--loss_config', default='MinUnsafeGPEntropy2LossConfig', type=str, choices=loss_configs.__all__)
    parser.add_argument('--alpha', default=5, type=int, help='safety loss alpha value, in percent')
    parser.add_argument("--lr", default=[1e-4, 3e-4, 5e-4], type=float, nargs='+')
    parser.add_argument('--resume_policy_path', default=None, type=str)
    parser.add_argument("--num_experiments_initial", default=[1, 1], type=int, nargs=2)
    parser.add_argument("--num_experiments", default=[20, 20], type=int, nargs=2)
    parser.add_argument("--batch_random_shorter_sequences", default=False, type=string2bool)
    parser.add_argument("--batch_random_split_sequences", default=False, type=string2bool)
    parser.add_argument("--policy_knows_budget", default=False, type=string2bool)
    parser.add_argument("--policy_encoding_dim", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument('--domain_warpper', default='tanh', type=lambda name: DomainWarpperType[name.upper()], choices=list(DomainWarpperType))
    parser.add_argument('--fast_tqdm', default=True, type=string2bool)
    args = parser.parse_args()

    #torch.autograd.set_detect_anomaly(True)

    flex_dim = args.flexible_dimension
    D = args.dimension
    kernel_config = RBFWithPriorPytorchConfig(
        input_dimension=D,
        base_lengthscale=0.2
    )
    mean_config_list = [BasicZeroMeanPytorchConfig()]
    safety_kernel_config = None # None: will copy kernel_config
    ##
    safety_mean_config_list = [BasicSechRotatedMeanPytorchConfig(input_dimension=D, center=0.5)] # safety un-aware training won't load this
    training_config_str = args.training_config
    lengthscale_distribution = LengthscaleDistribution.GAMMA_SMOOTH if args.train_smoother_pattern else LengthscaleDistribution.GAMMA

    if args.train_high_pressure_fluid_system:
        kernel_config = HighPressureFluidSystemRBFPytorchConfig()
        safety_mean_config_list = [HighPressureFluidSystemSechRotatedMeanPytorchConfig()]
        D = kernel_config.input_dimension
        flex_dim = False
        lengthscale_distribution = LengthscaleDistribution.PERCENTAGE

    lr_list = args.lr
    seed_list = args.seed
    is_safe_al = check_safety(getattr(loss_configs, args.loss_config)())
    loss_kwargs = {}
    if is_safe_al:
        loss_kwargs['probability_function_args'] = (args.alpha / 100, 0.0)
    if training_config_str == 'AmortizedContinuousFixGPPolicyTrainingConfig':
        loss_kwargs['num_kernels'] = 1
    if D <= 2 and 'MI' in args.loss_config:
        loss_kwargs['num_grid_points'] = 100
    # mark the folder a bit
    length_prefix = 'S' if args.batch_random_shorter_sequences and args.batch_random_split_sequences else ''
    length_prefix = length_prefix if args.batch_random_shorter_sequences else 'Fix'
    N0, N1 = args.num_experiments_initial
    T0, T1 = args.num_experiments
    folder_marker = '%sN%dx%d_T%dx%d%s'%(
        length_prefix,
        N0, N1,
        T0, T1,
        f'_alpha{args.alpha}' if is_safe_al else ''
    )
    #
    for lr in lr_list:
        for seed in seed_list:
            training_processer = TrainingProcesser(
                log_path = os.path.join(
                    args.experiment_output_dir,
                    'ContinuousGP%sDPolicy_loss_%s'%(
                        'Flex' if flex_dim else str(D), args.loss_config,
                    ),
                    f'lr{lr}_{folder_marker}'
                ),
                seed = seed,
                device = args.device,
                fast_tqdm = args.fast_tqdm
            )
            training_processer.set_loss_config(args.loss_config, **loss_kwargs)
            training_processer.set_policy_config(
                input_dimension=D,
                encoding_dim=args.policy_encoding_dim,
                domain_warpper=args.domain_warpper,
                safety_aware=is_safe_al,
                budget_aware=args.policy_knows_budget,
                resume_policy_path=args.resume_policy_path,
                flexible_dimension=flex_dim,
            )
            training_processer.set_training_config(
                training_config_str,
                kernel_config,
                mean_config_list,
                n_initial = (N0, N1), # initial set size
                n_steps = (T0, T1),
                random_subsequence=args.batch_random_shorter_sequences,
                split_subsequence=args.batch_random_split_sequences,
                optim_args={"lr": lr},
                lengthscale_distribution=lengthscale_distribution,
                safety_kernel_config=safety_kernel_config, # maybe None
                safety_mean_config_list=safety_mean_config_list
            )
            _ =  training_processer.train() # save settings, train, and save

            del training_processer



