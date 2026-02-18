import numpy as np
import os
import sys
import argparse
import glob
from pathlib import Path

from alef.experiments.pool_active_learning.main_dataset_al import experiment
from alef.experiments.pool_active_learning.main_dataset_sparse_al import experiment as sgpr_experiment
from alef.experiments.pool_active_learning.main_dataset_amorgp_al import experiment as amorgp_experiment
from alef.experiments.pool_active_learning.main_dataset_pfn_al import experiment as pfn_experiment

from alef.configs.paths import EXPERIMENT_PATH

parser = argparse.ArgumentParser(description="")
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--method', default=None, type=str)
args = parser.parse_args()

class exp_arguments:
    def __init__(
        self,
        experiment_output_dir,
        experiment_input_dir,
        experiment_idx,
        active_learner_config,
        dataset,
        plot_iterations,
        n_data_initial,
        n_steps,
        n_data_test,
        pfn_path = None,
        mgp: bool=False,
    ):
        self.experiment_output_dir = experiment_output_dir
        self.experiment_input_dir = experiment_input_dir
        self.experiment_idx = experiment_idx
        self.active_learner_config = active_learner_config
        self.model_path = (EXPERIMENT_PATH / 'Amor-Struct-GP-pretrained-weights' / 'main_state_dict_paper.pth').as_posix()
        self.pfn_path = pfn_path
        self.kernel_config = 'RBFWithPriorConfig'
        self.model_config = 'BasicGPModelMixtureConfig' if mgp else'BasicGPModelConfig' # BasicGPModelConfig | BasicGPModelMixtureConfig
        self.num_indusing_points = 15
        self.dataset = dataset
        self.plot_iterations = plot_iterations
        self.save_results = True
        self.n_data_initial = n_data_initial
        self.n_steps = n_steps
        self.n_data_test = n_data_test
        self.query_noisy = True


test_dataset_metadata = { # [dim, n_init, n_steps, n_test]
    'AirlinePassenger': {
        'dim': 1, 'n_init': 1, 'n_steps': 30, 'n_test': 50
    },
    'LGBB': {
        'dim': 2, 'n_init': 1, 'n_steps': 40, 'n_test': 200
    },
    'Airfoil': {
        'dim': 5, 'n_init': 20, 'n_steps': 60, 'n_test': 500
    },
}
for dataset, configs in test_dataset_metadata.items():
    if (not args.data is None) and args.data.lower() != dataset.lower():
        continue
    d = configs['dim']
    n_init = configs['n_init']
    n_steps = configs['n_steps']
    n_test = configs['n_test']
    for exp_id in range(5):
        with_to_plot = (exp_id == 0) and (d <= 2)
        for al_config in ['PredEntropyPoolActiveLearnerConfig', 'RandomPoolActiveLearnerConfig']:

            parse_args = f' --experiment_idx {exp_id}' + \
                    f' --plot_iterations {exp_id == 0}' + \
                    f' --active_learner_config {al_config}' + \
                    f' --dataset {dataset}' + \
                    f' --n_data_initial {n_init}' + \
                    f' --n_steps {n_steps}' + \
                    f' --n_data_test {n_test}'
            print(' ### executing main_dataset_al.py' + parse_args)

            exp_args = exp_arguments(
                EXPERIMENT_PATH/'AL', EXPERIMENT_PATH/'data', exp_id, al_config, dataset, with_to_plot, n_init, n_steps, n_test,
                pfn_path=(EXPERIMENT_PATH / 'pfns' / f'gp_model_{d}D_100b_2layers' / 'best_model.pt').as_posix(),
            )
            if args.method is None or args.method.lower() == 'gp':
                try:
                    print(' Running GP experiment...')
                    experiment(exp_args)
                except Exception as e:
                    print(f'experiment failed: {e}', file=sys.stderr)
            if args.method is None or args.method.lower() == 'svgp':
                try:
                    print(' Running Sparse GP experiment...')
                    sgpr_experiment(exp_args)
                except Exception as e:
                    print(f'experiment failed: {e}', file=sys.stderr)
            if args.method is None or args.method.lower() == 'agp':
                try:
                    print(' Running Amortized GP experiment...')
                    amorgp_experiment(exp_args)
                except Exception as e:
                    print(f'experiment failed: {e}', file=sys.stderr)
            if args.method is None or args.method.lower() in ['pfn', 'pfn_gpu']:
                try:
                    print(' Running PFN experiment...')
                    pfn_experiment(exp_args)
                except Exception as e:
                    print(f'experiment failed: {e}', file=sys.stderr)
            if args.method is None or args.method.lower() == 'mgp':
                try:
                    exp_args = exp_arguments(
                        EXPERIMENT_PATH/'AL', EXPERIMENT_PATH/'data', exp_id, al_config, dataset, with_to_plot, n_init, n_steps, n_test,
                        mgp=True,
                    )
                    print(' Running MGP experiment...')
                    experiment(exp_args)
                except Exception as e:
                    print(f'experiment failed: {e}', file=sys.stderr)


