import sys
import argparse
from alef.experiments.oracle_active_learning.main_pfn_oal import experiment as pfn_experiment
from alef.configs.paths import EXPERIMENT_PATH

parser = argparse.ArgumentParser(description="")
parser.add_argument('--data', default=None, type=str)
args = parser.parse_args()

class exp_arguments:
    def __init__(
        self,
        experiment_output_dir,
        experiment_idx,
        active_learner_config,
        oracle,
        plot_iterations,
        n_data_initial,
        n_steps,
        n_data_test,
        pfn_path = None,
    ):
        self.experiment_output_dir = experiment_output_dir
        self.experiment_idx = experiment_idx
        self.active_learner_config = active_learner_config
        self.model_path = (EXPERIMENT_PATH / 'Amor-Struct-GP-pretrained-weights' / 'main_state_dict_paper.pth').as_posix()
        self.pfn_path = pfn_path
        self.kernel_config = 'RBFWithPriorConfig'
        self.model_config = 'BasicGPModelMixtureConfig' # 'BasicGPModelConfig' # 'BasicGPModelMixtureConfig'
        self.num_indusing_points = 15
        self.oracle = oracle
        self.plot_iterations = plot_iterations
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
    1: 30, 2: 40,
}
n_data_test = {
    1: 50, 2: 200,
}

for d, oracles in test_oracles.items():
    for oracle in oracles:
        if (not args.data is None) and args.data.lower() != oracle.lower():
            continue
        for exp_id in range(5):
            wish_to_plot = (exp_id == 0) and (d <= 2)
            for al_config in ['PredEntropyOracleActiveLearnerConfig', 'RandomOracleActiveLearnerConfig']:

                parse_args = f' --experiment_idx {exp_id}' + \
                        f' --plot_iterations {wish_to_plot}' + \
                        f' --active_learner_config {al_config}' + \
                        f' --oracle {oracle}' + \
                        f' --n_data_initial {n_data_initial[d]}' + \
                        f' --n_steps {n_steps[d]}' + \
                        f' --n_data_test {n_data_test[d]}'
                print(' ### executing main_oal.py' + parse_args)

                exp_args = exp_arguments(
                    EXPERIMENT_PATH/'AL', exp_id, al_config, oracle, wish_to_plot, n_data_initial[d], n_steps[d], n_data_test[d],
                    pfn_path=(EXPERIMENT_PATH / 'checkpoints' / 'pfns' / f'gp_model_{d}D' / 'best_model.pt').as_posix(),
                )
                try:
                    print(' Running PFN experiment...')
                    pfn_experiment(exp_args)
                except Exception as e:
                    print(f'experiment failed: {e}', file=sys.stderr)


