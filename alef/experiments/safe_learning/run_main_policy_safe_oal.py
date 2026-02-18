import sys
from pathlib import Path

from alef.experiments.safe_learning.main_policy_safe_oal import experiment
#import logging
#logging.basicConfig(level=logging.INFO)

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
        self.kernel_config = 'RBFWithPriorConfig'
        self.model_config = 'BasicGPModelConfig'
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
    2: ['Simionescu', 'Townsend'],
}
n_data_initial = {
    2: 5, 4: 10,
}
n_steps = {
    2: [30], 4: [60],
}
n_data_test = {
    2: 200, 4: 5000,
}

for d, oracles in test_oracles.items():
    for oracle in oracles:
        for exp_id in range(5):
            for p in EXPERIMENT_PATH.glob(f'amortized_AL/ContinuousGP{d}DPolicy_loss_*afe*/*/'):
                wish_to_plot = (exp_id == 0 and d<=2)
                eval_gap = 1 if wish_to_plot else 200

                parse_args = f' --policy_path {p}' + \
                        f' --experiment_idx {exp_id}' + \
                        f' --plot_iterations {wish_to_plot}' + \
                        f' --oracle {oracle}' + \
                        f' --n_data_initial {n_data_initial[d]}' + \
                        f' --n_steps {n_steps[d]}' + \
                        f' --n_data_test {n_data_test[d]}' + \
                        '\n' + '###' * 10

                print(' ### executing main_policy_safe_oal.py' + parse_args)
                exp_args = exp_arguments(None, exp_id, p, oracle, wish_to_plot, eval_gap, n_data_initial[d], n_steps[d], n_data_test[d])
                try:
                    experiment(exp_args)
                except Exception as e:
                    print(f'experiment failed: {e}', file=sys.stderr)


