"""
// Copyright (c) 2024 Robert Bosch GmbH
// SPDX-License-Identifier: AGPL-3.0

Authors: Cen-You Li <cen-you.li@de.bosch.com> & Matthias Bitzer <matthias.bitzer3@de.bosch.com>
"""
import argparse
import os
import sys
import numpy as np
import logging
from pathlib import Path
from alef.utils.utils import string2bool
from alef.enums.active_learner_enums import ValidationType
from alef.active_learners.active_learner_factory import ActiveLearnerFactory
from alef.models.model_factory import ModelFactory

from alef.configs.config_picker import ConfigPicker

from alef import oracles
from alef.oracles import OracleNormalizer

from alef.configs.paths import EXPERIMENT_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="This is a script for oracle AL")
    # general AL experiment arguments
    parser.add_argument("--experiment_output_dir", default=(EXPERIMENT_PATH / 'AL').as_posix(), type=str)
    parser.add_argument("--experiment_idx", default=0, type=int)
    parser.add_argument('--active_learner_config', default='PredEntropyOracleActiveLearnerConfig', type=str, choices=['PredVarOracleActiveLearnerConfig', 'PredEntropyOracleActiveLearnerConfig', 'RandomOracleActiveLearnerConfig'])
    parser.add_argument("--kernel_config", default='RBFWithPriorConfig', type=str)
    parser.add_argument("--num_indusing_points", default=5, type=int)
    parser.add_argument("--oracle", default='BraninHoo', type=str)
    parser.add_argument("--plot_iterations", default=True, type=string2bool, help="this only works for 1 or 2 dim input though")
    parser.add_argument("--save_results", default=True, type=string2bool)
    # pool arguments for standard setting
    parser.add_argument("--n_data_initial", default=1, type=int)
    parser.add_argument("--n_steps", default=20, type=int)
    parser.add_argument("--n_data_test", default=200, type=int)
    parser.add_argument("--query_noisy", default=True, type=string2bool)
    parser.add_argument("--noise_level", default=0.1, type=float, help="noise level (std)")
    args = parser.parse_args()
    return args


def experiment(args):
    exp_idx = args.experiment_idx
    
    n_data_initial = args.n_data_initial
    n_steps = args.n_steps
    n_data_test = args.n_data_test
    save_results = args.save_results

    al_config = ConfigPicker.pick_active_learner_config(args.active_learner_config)(validation_at=np.arange(0, n_steps).tolist())
    learner = ActiveLearnerFactory.build(al_config)

    # need an oracle
    oracle_str = args.oracle
    if oracle_str == 'GPOracle1D':
        oracle = getattr(oracles, oracle_str)(
            ConfigPicker.pick_kernel_config(args.kernel_config)(input_dimension=1, base_lengthscale=0.2, fix_variance=True),
            args.noise_level)
        oracle.initialize(0, 1, 50)
    elif oracle_str == 'GPOracle2D':
        oracle = getattr(oracles, oracle_str)(
            ConfigPicker.pick_kernel_config(args.kernel_config)(input_dimension=2, base_lengthscale=[0.9, 0.6], fix_variance=True),
            args.noise_level)
        oracle.initialize(0, 1, 50)
    else:
        oracle = OracleNormalizer(
            getattr(oracles, oracle_str)(args.noise_level)
        )
        oracle.set_normalization_by_sampling( target_scale= np.sqrt(1.0 - args.noise_level**2) )

    learner.set_oracle(oracle)

    # create kernels and models
    kernel_config = ConfigPicker.pick_kernel_config(args.kernel_config)(
        input_dimension=oracle.get_dimension(),
        base_lengthscale=0.2,
        fix_variance=True
    )
    model_config = ConfigPicker.pick_model_config('BasicSparseGPModelConfig')(
        kernel_config=kernel_config,
        n_inducing_points=args.num_indusing_points,
        observation_noise=args.noise_level,
        optimize_hps=True,
        train_likelihood_variance=True
    )

    model = ModelFactory.build(model_config)
    learner.set_model(model)

    # initial data & test data
    learner.sample_train_set(n_data_initial, set_seed=True, seed=exp_idx)
    learner.sample_test_set(n_data_test, set_seed=False)

    # save settings
    exp_path = Path(args.experiment_output_dir) / (
        '%s_%s_BasicSparseGPModelConfig_%div'%(
            al_config.__class__.__name__,
            args.kernel_config,
            args.num_indusing_points
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
        learner.save_experiment_summary_to_path(exp_path, f'{exp_idx}_OracleAL_result.xlsx')

    # run experiments
    learner.learn(n_steps)

if __name__ == "__main__":
    args = parse_args()
    experiment(args)

