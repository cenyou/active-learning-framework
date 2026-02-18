import numpy as np
import os
import sys
import glob
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from alef.experiments.oracle_active_learning.main_aline_oal import experiment

from alef.configs.paths import EXPERIMENT_PATH

class exp_arguments:
    def __init__(
        self,
        experiment_output_dir,
        experiment_idx,
        policy_path,
        oracle,
        plot_iterations,
        evaluate_every_n_iterations,
        n_data_initial,
        n_steps,
        n_data_test
    ):
        self.experiment_output_dir = experiment_output_dir
        self.experiment_idx = experiment_idx
        self.policy_path = policy_path
        self.oracle = oracle
        self.plot_iterations = plot_iterations
        self.evaluate_every_n_iterations = evaluate_every_n_iterations
        self.save_results = True
        self.n_data_initial = n_data_initial
        self.n_steps = n_steps
        self.n_data_test = n_data_test
        self.query_noisy = True
        self.noise_level = 0.1


test_oracles = {
    1: ['Sinus'],
    2: ['BraninHoo'],
}
n_data_initial = {
    1: 1, 2: 1,
}
n_steps = {
    1: [20], 2: [30],
}
n_data_test = {
    1: 50, 2: 200,
}

for d, oracles in test_oracles.items():
    for oracle in oracles:
        for exp_id in range(5):
            for p in EXPERIMENT_PATH.glob(f'checkpoints/aline/{d}D{n_steps[d][0]}T/ckpt_al_{d}d_40000.tar'):
                wish_to_plot = False #(exp_id == 0 and d<=2)
                eval_gap = 1

                parse_args = f' --policy_path {p}' + \
                        f' --experiment_idx {exp_id}' + \
                        f' --plot_iterations {wish_to_plot}' + \
                        f' --oracle {oracle}' + \
                        f' --n_data_initial {n_data_initial[d]}' + \
                        f' --n_steps {n_steps[d]}' + \
                        f' --n_data_test {n_data_test[d]}'

                print(' ### executing main_policy_oal.py' + parse_args)
                exp_args = exp_arguments(EXPERIMENT_PATH/'amortized_AL', exp_id, p, oracle, wish_to_plot, eval_gap, n_data_initial[d], n_steps[d], n_data_test[d])
                try:
                    experiment(exp_args)
                except Exception as e:
                    print(f'experiment failed: {e}', file=sys.stderr)


