import argparse
import numpy as np
from pathlib import Path
from alef.utils.utils import string2bool
from alef.active_learners.pool_aline import PoolALINE
from alef.enums.active_learner_enums import ValidationType
from alef.configs.config_picker import ConfigPicker

from alef import oracles
from alef.oracles import OracleNormalizer
from alef.pools import PoolFromOracle

from alef.configs.paths import EXPERIMENT_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="This is a script for oracle AL")
    # general AL experiment arguments
    parser.add_argument("--experiment_output_dir", default=None, type=str)
    #parser.add_argument("--experiment_output_dir", default=(EXPERIMENT_PATH / 'amortized_AL').as_posix(), type=str)
    parser.add_argument("--experiment_idx", default=0, type=int)
    parser.add_argument("--policy_path", default= (EXPERIMENT_PATH / 'amortized_AL' / 'ContinuousGP2DPolicy_loss_GPEntropy2LossConfig' / 'CET_2024_01_19__13_50_00_seed2').as_posix(), type=str)
    parser.add_argument("--oracle", default='BraninHoo', type=str)
    parser.add_argument("--plot_iterations", default=True, type=string2bool, help="this only works for 1 or 2 dim input though")
    parser.add_argument("--evaluate_every_n_iterations", default=100, type=int)
    parser.add_argument("--save_results", default=True, type=string2bool)
    # pool arguments for standard setting
    parser.add_argument("--n_data_initial", default=1, type=int)
    parser.add_argument("--n_steps", default=[20], type=int, nargs='+')
    parser.add_argument("--n_data_test", default=200, type=int)
    parser.add_argument("--query_noisy", default=True, type=string2bool)
    parser.add_argument("--noise_level", default=0.1, type=float, help="noise level (std)")
    args = parser.parse_args()
    return args


def experiment(args):
    store_path = Path(args.experiment_output_dir) if not args.experiment_output_dir is None else Path(args.policy_path)
    exp_idx = args.experiment_idx
    oracle_str = args.oracle
    
    n_data_initial = args.n_data_initial
    n_steps_each = args.n_steps
    n_steps = sum(n_steps_each)
    n_data_test = args.n_data_test
    eval_gap = args.evaluate_every_n_iterations
    save_results = args.save_results

    learner = PoolALINE(
        validation_type = [ValidationType.MAE, ValidationType.RMSE, ValidationType.NEG_LOG_LIKELI],
        policy_path=str(args.policy_path),
        validation_at= [i for i in range(n_steps) if i%eval_gap==(eval_gap-1) or i==n_steps-1],
    )

    # need an oracle
    oracle = OracleNormalizer(
        getattr(oracles, oracle_str)(args.noise_level)
    )
    oracle.set_normalization_by_sampling( target_scale= np.sqrt(1.0 - args.noise_level**2) )
    pool = PoolFromOracle(oracle, seed=2024 + exp_idx, set_seed=True)
    pool.discretize_random(5000)
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
        'ALINE' + ''.join([f'_{T}' for T in n_steps_each])
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
    learner.learn(n_steps_each)


if __name__ == "__main__":
    args = parse_args()
    experiment(args)

