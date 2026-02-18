import numpy as np
import sys
import argparse
from alef.enums.active_learner_enums import ValidationType
from alef.experiments.safe_learning.main_safe_pfn_al import experiment as pfn_experiment
from alef.configs.paths import EXPERIMENT_PATH

import logging
logging.basicConfig(level=logging.WARNING)

parser = argparse.ArgumentParser(description="")
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--exp_id', default=None, type=str)
args = parser.parse_args()

class exp_arguments:
    def __init__(
        self,
        experiment_output_dir,
        experiment_input_dir,
        experiment_idx,
        acquisition_function_config,
        optimize_acquisition_by_gradient,
        oracle,
        constraint_on_y,
        plot_iterations,
        n_pool,
        n_data_initial,
        n_steps,
        n_data_test,
        safe_lower,
        safe_upper,
        pfn_path = None,
    ):
        self.experiment_output_dir = experiment_output_dir
        self.experiment_input_dir = experiment_input_dir
        self.experiment_idx = experiment_idx
        self.acquisition_function_config = acquisition_function_config
        self.optimize_acquisition_by_gradient = optimize_acquisition_by_gradient
        self.validation_type = [ValidationType.MAE, ValidationType.RMSE, ValidationType.NEG_LOG_LIKELI]
        self.model_path = (EXPERIMENT_PATH / 'Amor-Struct-GP-pretrained-weights' / 'main_state_dict_paper.pth').as_posix()
        self.pfn_path = pfn_path
        self.kernel_config = 'RBFWithPriorConfig'
        self.model_config = 'BasicGPModelMixtureConfig' # BasicGPModelConfig | BasicGPModelMixtureConfig
        self.num_indusing_points = 20
        self.oracle = oracle
        self.constraint_on_y = constraint_on_y
        self.plot_iterations = plot_iterations
        self.save_results = True
        self.n_pool = n_pool
        self.n_data_initial = n_data_initial
        self.n_steps = n_steps
        self.n_data_test = n_data_test
        self.query_noisy = True
        self.noise_level = 0.1
        self.safe_lower = safe_lower
        self.safe_upper = safe_upper


test_dataset_metadata = {
    'Simionescu': {
        'dim': 2, 'n_init': 5, 'n_steps': 40, 'n_test': 200, 'n_constraints': 1, 'constraint_on_y': False, 'optimize_acquisition_by_gradient': False,
    },
    'Townsend': {
        'dim': 2, 'n_init': 5, 'n_steps': 40, 'n_test': 200, 'n_constraints': 1, 'constraint_on_y': False, 'optimize_acquisition_by_gradient': False,
    },
    'lgbb': {
        'dim': 2, 'n_init': 5, 'n_steps': 40, 'n_test': 200, 'n_constraints': 1, 'constraint_on_y': False, 'optimize_acquisition_by_gradient': False,
    },
    'Engine3D': {
        'dim': 3, 'n_init': 5, 'n_steps': 100, 'n_test': 200, 'n_constraints': 1, 'constraint_on_y': False, 'optimize_acquisition_by_gradient': False,
    },
    # 'HighPressureFluidSystem': {
    #     'dim': 7, 'n_init': 10, 'n_steps': 100, 'n_test': 10000, 'n_constraints': 1, 'constraint_on_y': False, 'optimize_acquisition_by_gradient': False,
    # },
}
for oracle, configs in test_dataset_metadata.items():
    if (not args.data is None) and args.data.lower() != oracle.lower():
        continue
    d = configs['dim']
    n_pool = 5000 if d <= 4 else 10**6
    n_init = configs['n_init']
    n_steps = configs['n_steps']
    n_test = configs['n_test']
    n_constraints = configs['n_constraints']
    constraint_on_y = configs['constraint_on_y']
    oabg = configs['optimize_acquisition_by_gradient']
    for exp_id in range(5):
        if (not args.exp_id in [None, 'None']) and int(args.exp_id) != exp_id:
            continue
        wish_to_plot = (exp_id == 0) and (d <= 2)
        for acq_func_config in [
            'BasicSafePredEntropyConfig',
            'BasicMinUnsafePredEntropyConfig',
            'BasicSafeRandomConfig',
        ]:
            parse_args = f' --experiment_idx {exp_id}' + \
                    f' --plot_iterations {wish_to_plot}' + \
                    f' --acquisition_function_config {acq_func_config}' + \
                    f' --oracle {oracle}' + \
                    f' --constraint_on_y {constraint_on_y}' + \
                    f' --n_pool {n_pool}' + \
                    f' --n_data_initial {n_init}' + \
                    f' --n_steps {n_steps}' + \
                    f' --n_data_test {n_test}' + \
                    f' --safe_lower {[0.0]*n_constraints}' + \
                    f' --safe_upper {[np.inf]*n_constraints}'

            exp_args = exp_arguments(
                EXPERIMENT_PATH/'safe_AL',
                EXPERIMENT_PATH/'data',
                exp_id,
                acq_func_config,
                oabg,
                oracle,
                constraint_on_y,
                wish_to_plot,
                n_pool,
                n_init,
                n_steps,
                n_test,
                [0.0]*n_constraints,
                [np.inf]*n_constraints,
                pfn_path=(EXPERIMENT_PATH / 'checkpoints' / 'pfns' / f'gp_model_{d}D' / 'best_model.pt').as_posix(),
            )
            
            try:
                print(' ### executing main_safe_pfn_al.py' + parse_args)
                pfn_experiment(exp_args)
            except Exception as e:
                print(f'experiment failed: {e}', file=sys.stderr)
