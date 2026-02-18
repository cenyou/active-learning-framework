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
from alef.active_learners.active_learner_factory import ActiveLearnerFactory
from alef.models.model_factory import ModelFactory
from alef.configs.config_picker import ConfigPicker

from alef.pools import PoolFromData

from alef.data_sets.airfoil import Airfoil
from alef.data_sets.airline_passenger import AirlinePassenger
from alef.data_sets.lgbb import LGBB
from alef.data_sets.power_plant import PowerPlant

from alef.configs.paths import EXPERIMENT_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="This is a script for AL")
    # general AL experiment arguments
    parser.add_argument("--experiment_output_dir", default=None, type=str)
    #parser.add_argument("--experiment_output_dir", default=(EXPERIMENT_PATH / 'amortized_AL').as_posix(), type=str)
    parser.add_argument("--experiment_input_dir", default=(EXPERIMENT_PATH / 'data').as_posix(), type=str)
    parser.add_argument("--experiment_idx", default=0, type=int)
    parser.add_argument("--policy_path", default= (EXPERIMENT_PATH / 'amortized_AL' / 'ContinuousGP2DPolicy_loss_GPMI2LossConfig' / 'lr0.0003_seed1_CET_2024_02_17__17_02_34').as_posix(), type=str)
    parser.add_argument("--kernel_config", default='RBFWithPriorConfig', type=str)
    parser.add_argument("--model_config", default='BasicGPModelConfig', type=str)
    parser.add_argument("--dataset", default='lgbb', type=str)
    parser.add_argument("--plot_iterations", default=True, type=string2bool, help="this only works for 1 or 2 dim input though")
    parser.add_argument("--evaluate_every_n_iterations", default=100, type=int)
    parser.add_argument("--save_results", default=True, type=string2bool)
    # pool arguments for standard setting
    parser.add_argument("--n_data_initial", default=1, type=int)
    parser.add_argument("--n_steps", default=[20], type=int, nargs='+')
    parser.add_argument("--n_data_test", default=200, type=int)
    parser.add_argument("--query_noisy", default=True, type=string2bool)
    args = parser.parse_args()
    return args


def experiment(args):
    data_path = Path(args.experiment_input_dir)
    store_path = Path(args.experiment_output_dir) if not args.experiment_output_dir is None else Path(args.policy_path)
    exp_idx = args.experiment_idx
    
    n_data_initial = args.n_data_initial
    n_steps_each = args.n_steps
    n_steps = sum(n_steps_each)
    n_data_test = args.n_data_test
    #val_type = args.validation_type
    eval_gap = args.evaluate_every_n_iterations
    save_results = args.save_results

    al_config = ConfigPicker.pick_active_learner_config('BasicPoolPolicyActiveLearnerConfig')(
        policy_path=str(args.policy_path),
        validation_at= [i for i in range(n_steps) if i%eval_gap==(eval_gap-1) or i==n_steps-1],
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
        observation_noise=0.1,
        optimize_hps=True,
        train_likelihood_variance=True
    )

    model = ModelFactory.build(model_config)
    learner.set_model(model)

    # need an pool
    dataset_str = args.dataset.lower()
    if dataset_str == 'airfoil':
        dataset = Airfoil(base_path= data_path / 'airfoil')
    elif dataset_str == 'airlinepassenger':
        dataset = AirlinePassenger(base_path= data_path / 'airlines')
    elif dataset_str == 'lgbb':
        dataset = LGBB(base_path= data_path / 'lgbb')
    elif dataset_str == 'powerplant':
        dataset = PowerPlant(base_path= Path(args.experiment_input_dir) / 'power_plant')
    else:
        raise NotImplementedError
    dataset.load_data_set()
    pool = PoolFromData(*dataset.get_complete_dataset(), data_is_noisy=True, seed=2024 + exp_idx, set_seed=True)

    assert pool.get_dimension() == D
    learner.set_pool(pool)

    # initial data & test data
    learner.set_train_set(
        *pool.get_random_data(n_data_initial, noisy=True)
    )
    learner.set_test_set(
        *pool.get_random_data(n_data_test, noisy=True)
    )

    # save settings
    exp_path = store_path / ( 
        '%s_%s_%s%s'%(
            'BasicPoolPolicyActiveLearnerConfig',
            args.kernel_config,
            args.model_config,
            ''.join([f'_{T}' for T in n_steps_each])
        )
    ) / f"{dataset_str}_{n_data_initial}"

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
        learner.save_experiment_summary_to_path(exp_path, f'{exp_idx}_AL_result.xlsx')

    # run experiments
    learner.learn(n_steps_each)
    

if __name__ == "__main__":
    args = parse_args()
    experiment(args)

