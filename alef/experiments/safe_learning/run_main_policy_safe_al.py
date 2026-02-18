
import sys
from pathlib import Path

from alef.experiments.safe_learning.main_policy_safe_al import experiment
#import logging
#logging.basicConfig(level=logging.INFO)

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
        self.kernel_config = 'RBFWithPriorConfig'
        self.model_config = 'BasicGPModelConfig'
        self.dataset = dataset
        self.plot_iterations = plot_iterations
        self.evaluate_every_n_iterations = evaluate_every_n_iterations
        self.save_results = True
        self.n_data_initial = n_data_initial
        self.n_steps = n_steps
        self.n_data_test = n_data_test


test_dataset_metadata = {
    'lgbb': {
        'dim': 2, 'n_init': 5, 'n_steps': [30], 'n_test': 200, 'n_constraints': 1, 'constraint_on_y': False,
    },
    'Engine3D': {
        'dim': 3, 'n_init': 5, 'n_steps': [60], 'n_test': 200, 'n_constraints': 1, 'constraint_on_y': False,
    },
    'HighPressureFluidSystem': {
        'dim': 7, 'n_init': 10, 'n_steps': [100], 'n_test': 10000, 'n_constraints': 1, 'constraint_on_y': False,
    },
}
for oracle, configs in test_dataset_metadata.items():
    d = configs['dim']
    n_init = configs['n_init']
    n_steps = configs['n_steps']
    n_test = configs['n_test']
    for exp_id in range(5):
        for p in EXPERIMENT_PATH.glob(f'amortized_AL/ContinuousGP{d}DPolicy_loss_*afe*/*/'):
            wish_to_plot = (exp_id == 0 and d<=2)
            eval_gap = 1 if wish_to_plot else 200

            parse_args = f' --policy_path {p}' + \
                    f' --experiment_idx {exp_id}' + \
                    f' --plot_iterations {wish_to_plot}' + \
                    f' --oracle {oracle}' + \
                    f' --n_data_initial {n_init}' + \
                    f' --n_steps {n_steps}' + \
                    f' --n_data_test {n_test}' + \
                    '\n' + '###' * 10

            print(' ### executing main_policy_safe_al.py' + parse_args)
            exp_args = exp_arguments(None, EXPERIMENT_PATH/'data', exp_id, p, oracle, wish_to_plot, eval_gap, n_init, n_steps, n_test)
            try:
                experiment(exp_args)
            except Exception as e:
                print(f'experiment failed: {e}', file=sys.stderr)


