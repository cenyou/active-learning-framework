"""
// Copyright (c) 2024 Robert Bosch GmbH
// SPDX-License-Identifier: AGPL-3.0

Authors: Cen-You Li <cen-you.li@de.bosch.com> & Matthias Bitzer <matthias.bitzer3@de.bosch.com>
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from alef.utils.utils import string2bool
from alef.configs.initial_data_conditions import CONSTRAINED_BOXES, INITIAL_OUTPUT_HIGHER_BY
from alef.active_learners.active_learner_factory import ActiveLearnerFactory
from alef.models.model_factory import ModelFactory
from alef.configs.config_picker import ConfigPicker
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import BasicRBFPytorchConfig
from alef.configs.means.pytorch_means import BasicSechMeanPytorchConfig

from alef import oracles
from alef.oracles import OracleNormalizer, StandardConstrainedOracle, StandardOracle

from alef.configs.paths import EXPERIMENT_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="This is a script for oracle AL")
    # general AL experiment arguments
    parser.add_argument("--experiment_output_dir", default=None, type=str)
    #parser.add_argument("--experiment_output_dir", default=(EXPERIMENT_PATH / 'amortized_AL').as_posix(), type=str)
    parser.add_argument("--experiment_idx", default=0, type=int)
    parser.add_argument("--policy_path", default= (EXPERIMENT_PATH / 'amortized_AL' / 'ContinuousGP2DPolicy_loss_SafeGPMyopicMI2LossConfig' / 'lr0.0001_seed0_CEST_2024_04_01__15_53_29').as_posix(), type=str)
    parser.add_argument("--kernel_config", default='RBFWithPriorConfig', type=str)
    parser.add_argument("--model_config", default='BasicGPModelConfig', type=str)
    parser.add_argument("--oracle", default='BraninHoo', type=str)
    parser.add_argument("--plot_iterations", default=True, type=string2bool, help="this only works for 1 or 2 dim input though")
    parser.add_argument("--evaluate_every_n_iterations", default=100, type=int)
    parser.add_argument("--save_results", default=True, type=string2bool)
    # pool arguments for standard setting
    parser.add_argument("--n_data_initial", default=10, type=int)
    parser.add_argument("--n_steps", default=[20], type=int, nargs='+')
    parser.add_argument("--n_data_test", default=200, type=int)
    parser.add_argument("--query_noisy", default=True, type=string2bool)
    parser.add_argument("--noise_level", default=0.1, type=float, help="noise level (std)")
    args = parser.parse_args()
    return args


def experiment(args):
    store_path = Path(args.experiment_output_dir) if not args.experiment_output_dir is None else Path(args.policy_path)
    exp_idx = args.experiment_idx
    oracle_str = args.oracle
    
    n_data_initial = args.n_data_initial
    n_steps_each = args.n_steps
    n_steps = sum(n_steps_each)
    n_data_test = args.n_data_test
    #val_type = args.validation_type
    eval_gap = args.evaluate_every_n_iterations
    save_results = args.save_results

    al_config = ConfigPicker.pick_active_learner_config('BasicOraclePolicySafeActiveLearnerConfig')(
        policy_path=str(args.policy_path),
        validation_at= [i for i in range(n_steps) if i%eval_gap==(eval_gap-1) or i==n_steps-1],
        constraint_on_y=False,
    )
    learner = ActiveLearnerFactory.build(al_config)

    # create kernels and models
    D = int(str(args.policy_path).split('ContinuousGP')[-1].split('DPolicy')[0])
    kernel_config = ConfigPicker.pick_kernel_config(args.kernel_config)(
        input_dimension=D,
        base_lengthscale=([0.9, 0.6] if D==2 else 0.2),
        fix_variance=True
    )
    model_config = ConfigPicker.pick_model_config(args.model_config)(
        kernel_config=kernel_config,
        observation_noise=args.noise_level,
        optimize_hps=not oracle_str.startswith('GPOracle'),
        train_likelihood_variance=True
    )

    model = ModelFactory.build(model_config)
    learner.set_model(model)

    # need an oracle
    if oracle_str.startswith('GPOracle'):
        oracleMain = getattr(oracles, oracle_str)(
            BasicRBFPytorchConfig(
                input_dimension=D,
                base_lengthscale=np.random.uniform(0.3, 0.8, D).tolist(),#([0.9, 0.6] if D==2 else 0.2),
            ),
            args.noise_level,
            shift_mean=True
        )

        oracleMain.initialize(0, 1, 20)
        oracleConstraint = getattr(oracles, oracle_str)(
            BasicRBFPytorchConfig(
                input_dimension=D,
                base_lengthscale=np.random.uniform(0.3, 0.8, D).tolist(),#([0.9, 0.6] if D==2 else 0.2),
                base_variance=0.5
            ),
            args.noise_level,
            mean_config=BasicSechMeanPytorchConfig(input_dimension=D, scale=3.2*0.5),
            shift_mean=True
        )
        oracleConstraint.initialize(0, 1, 20)
        oracle = StandardConstrainedOracle(oracleMain, oracleConstraint)
    else:
        oracle = getattr(oracles, oracle_str)(args.noise_level)
        if isinstance(oracle, StandardOracle):
            oracleMain = OracleNormalizer(oracle)
            oracleMain.set_normalization_by_sampling( target_scale= np.sqrt(1.0 - args.noise_level**2) )
            oracleConstraint = OracleNormalizer(getattr(oracles, oracle_str)(args.noise_level))
            oracleConstraint.set_normalization_by_sampling( target_scale= np.sqrt(1.0 - args.noise_level**2) )
            oracle = StandardConstrainedOracle(oracleMain, oracleConstraint)
        else:
            assert isinstance(oracle, StandardConstrainedOracle)

    assert oracle.get_dimension() == D
    learner.set_oracle(oracle)

    # initial data & test data
    if oracle_str in CONSTRAINED_BOXES.keys():
        a, box_width = CONSTRAINED_BOXES[oracle_str]
        learner.sample_constrained_train_set_in_box(n_data_initial, a, box_width, set_seed=True, seed=exp_idx, constraint_lower=0)
    else:
        learner.sample_constrained_train_set(n_data_initial, set_seed=True, seed=exp_idx, constraint_lower=INITIAL_OUTPUT_HIGHER_BY)
    learner.sample_constrained_test_set(n_data_test, set_seed=False, constraint_lower=0.0)

    # save settings
    exp_path = store_path / ( 
        '%s_%s_%s%s'%(
            'BasicOraclePolicySafeActiveLearnerConfig',
            args.kernel_config,
            args.model_config,
            ''.join([f'_{T}' for T in n_steps_each])
        )
    ) / oracle_str

    if not exp_path.is_dir():
        exp_path.mkdir(parents=True, exist_ok=True)
    # save experiment setup
    with open(exp_path / f'{exp_idx}_experiment_description.txt', mode='w') as fp:
        for name, content in args.__dict__.items():
            print(f'{name}  :  {content}', file=fp)

    # 
    learner.set_do_plotting(args.plot_iterations)
    if args.plot_iterations and args.save_results:
        learner.save_plots_to_path(exp_path)
    if args.save_results:
        learner.save_experiment_summary_to_path(exp_path, f'{exp_idx}_OracleSafeAL_result.xlsx')

    # run experiments
    learner.learn(n_steps_each)
    

if __name__ == "__main__":
    args = parse_args()
    experiment(args)

