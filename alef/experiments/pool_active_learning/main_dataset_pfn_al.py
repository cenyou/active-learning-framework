"""
// Copyright (c) 2024 Robert Bosch GmbH
// SPDX-License-Identifier: AGPL-3.0

Authors: Cen-You Li <cen-you.li@de.bosch.com> & Matthias Bitzer <matthias.bitzer3@de.bosch.com>
"""
import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from alef.utils.utils import string2bool
from alef.enums.active_learner_enums import ValidationType
from alef.active_learners.active_learner_factory import ActiveLearnerFactory
from alef.configs.models.pfn_config import PFNTorchConfig
from alef.configs.models.pfn_model_config import BasicPFNModelConfig, PFNModelGPUConfig
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
    parser.add_argument("--experiment_output_dir", default=(EXPERIMENT_PATH / 'AL').as_posix(), type=str)
    parser.add_argument("--experiment_input_dir", default=(EXPERIMENT_PATH / 'data').as_posix(), type=str)
    parser.add_argument("--pfn_path", default=(EXPERIMENT_PATH / 'Amor-Struct-GP-pretrained-weights' / 'main_state_dict_paper.pth').as_posix(), type=str)
    parser.add_argument("--experiment_idx", default=0, type=int)
    parser.add_argument('--active_learner_config', default='PredEntropyPoolActiveLearnerConfig', type=str, choices=['PredVarPoolActiveLearnerConfig', 'PredEntropyPoolActiveLearnerConfig', 'RandomPoolActiveLearnerConfig'])
    parser.add_argument("--dataset", default='AirlinePassenger', type=str)
    parser.add_argument("--plot_iterations", default=True, type=string2bool, help="this only works for 1 or 2 dim input though")
    parser.add_argument("--save_results", default=True, type=string2bool)
    # pool arguments for standard setting
    parser.add_argument("--n_data_initial", default=1, type=int)
    parser.add_argument("--n_steps", default=20, type=int)
    parser.add_argument("--n_data_test", default=50, type=int)
    parser.add_argument("--query_noisy", default=True, type=string2bool)
    args = parser.parse_args()
    return args


def experiment(args):
    exp_idx = args.experiment_idx
    
    n_data_initial = args.n_data_initial
    n_steps = args.n_steps
    n_data_test = args.n_data_test
    save_results = args.save_results

    al_config = ConfigPicker.pick_active_learner_config(args.active_learner_config)(
        validation_at=np.arange(0, n_steps).tolist(),
        use_smaller_acquistion_set=False,
    )
    learner = ActiveLearnerFactory.build(al_config)

    # need an pool
    dataset_str = args.dataset.lower()
    if dataset_str == 'airfoil':
        dataset = Airfoil(base_path= Path(args.experiment_input_dir) / 'airfoil')
    elif dataset_str == 'airlinepassenger':
        dataset = AirlinePassenger(base_path= Path(args.experiment_input_dir) / 'airlines')
    elif dataset_str == 'lgbb':
        dataset = LGBB(base_path= Path(args.experiment_input_dir) / 'lgbb')
    elif dataset_str == 'powerplant':
        dataset = PowerPlant(base_path= Path(args.experiment_input_dir) / 'power_plant')
    else:
        raise NotImplementedError
    dataset.load_data_set()
    pool = PoolFromData(*dataset.get_complete_dataset(), data_is_noisy=True, seed=2024 + exp_idx, set_seed=True)

    learner.set_pool(pool)

    # create models
    model_config_class = PFNModelGPUConfig if torch.cuda.is_available() else BasicPFNModelConfig
    model_config = model_config_class(
        pfn_backend_config=PFNTorchConfig( input_dimension=pool.get_dimension() ),
        checkpoint_path=args.pfn_path
    )
    model = ModelFactory.build(model_config)
    learner.set_model(model)

    # initial data & test data
    learner.set_train_set(
        *pool.get_random_data(n_data_initial, noisy=True)
    )
    learner.set_test_set(
        *pool.get_random_data(n_data_test, noisy=True)
    )

    # save settings
    exp_path = Path(args.experiment_output_dir) / (
        '%s_%s'%(
            al_config.__class__.__name__,
            model_config.__class__.__name__
        )
     ) / dataset_str

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
    learner.learn(n_steps)

if __name__ == "__main__":
    args = parse_args()
    experiment(args)

