import numpy as np
import os
import sys
import glob
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from alef.experiments.pool_active_learning.main_aline_dataset_al import experiment

from alef.configs.paths import EXPERIMENT_PATH

class exp_arguments:
    def __init__(
        self,
        experiment_output_dir,
        experiment_input_dir,
        experiment_idx,
        policy_path,
        dataset,
        plot_iterations,
        evaluate_every_n_iterations,
        n_data_initial,
        n_steps,
        n_data_test
    ):
        self.experiment_output_dir = experiment_output_dir
        self.experiment_input_dir = experiment_input_dir
        self.experiment_idx = experiment_idx
        self.policy_path = policy_path
        self.dataset = dataset
        self.plot_iterations = plot_iterations
        self.evaluate_every_n_iterations = evaluate_every_n_iterations
        self.save_results = True
        self.n_data_initial = n_data_initial
        self.n_steps = n_steps
        self.n_data_test = n_data_test
        self.query_noisy = True


test_dataset_metadata = { # [dim, n_init, n_steps, n_test]
    'AirlinePassenger': {
        'dim': 1, 'n_init': 1, 'n_steps': [20], 'n_test': 50
    },
    'LGBB': {
        'dim': 2, 'n_init': 1, 'n_steps': [30], 'n_test': 200
    },
    'Airfoil': {
        'dim': 5, 'n_init': 20, 'n_steps': [40], 'n_test': 500
    },
}
for dataset, configs in test_dataset_metadata.items():
    d = configs['dim']
    n_init = configs['n_init']
    n_steps = configs['n_steps']
    n_test = configs['n_test']
    for exp_id in range(5):
        for p in EXPERIMENT_PATH.glob(f'checkpoints/aline/{d}D{n_steps[0]}T/ckpt_al_{d}d_40000.tar'):
            wish_to_plot = False #(exp_id == 0 and d<=2)
            eval_gap = 1 if wish_to_plot else 200

            parse_args = f' --policy_path {p}' + \
                    f' --experiment_idx {exp_id}' + \
                    f' --plot_iterations {wish_to_plot}' + \
                    f' --dataset {dataset}' + \
                    f' --n_data_initial {n_init}' + \
                    f' --n_steps {n_steps}' + \
                    f' --n_data_test {n_test}'

            print(' ### executing main_policy_dataset_al.py' + parse_args)
            exp_args = exp_arguments(EXPERIMENT_PATH/'amortized_AL', EXPERIMENT_PATH / 'data', exp_id, p, dataset, wish_to_plot, eval_gap, n_init, n_steps, n_test)
            try:
                experiment(exp_args)
            except Exception as e:
                print(f'experiment failed: {e}', file=sys.stderr)


