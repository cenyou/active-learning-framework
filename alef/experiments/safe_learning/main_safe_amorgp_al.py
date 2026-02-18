"""
// Copyright (c) 2024 Robert Bosch GmbH
// SPDX-License-Identifier: AGPL-3.0

Authors: Cen-You Li <cen-you.li@de.bosch.com>
"""
import numpy as np
import argparse
import os
import sys
import logging
from copy import deepcopy
from pathlib import Path
from alef.utils.utils import string2bool
from alef.configs.initial_data_conditions import CONSTRAINED_BOXES, INITIAL_OUTPUT_HIGHER_BY, INITIAL_OUTPUT_LOWER_BY
from alef.active_learners.pool_safe_active_learner import PoolSafeActiveLearner
from alef.enums.active_learner_enums import ValidationType
from alef.acquisition_functions.acquisition_function_factory import AcquisitionFunctionFactory
from amorstructgp.config.models.gp_model_amortized_structured_config import PaperAmortizedStructuredConfig,AmortizedStructuredWithMaternConfig
from amorstructgp.gp.base_symbols import BaseKernelTypes
from amorstructgp.models.model_factory import ModelFactory
import alef.oracles as oracles
from alef.oracles import OracleNormalizer
import alef.pools as pools
from alef.pools import PoolFromOracle, PoolWithSafetyFromOracle, PoolFromData, PoolWithSafetyFromData
from alef.data_sets.engine import Engine2
from alef.data_sets.lgbb import OutputType as LGBB_OT
from alef.data_sets.lgbb import LGBB
from alef.data_sets.power_plant import PowerPlant

from alef.configs.config_picker import ConfigPicker

import gpflow
gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
#f64 = gpflow.utilities.to_default_float

from alef.configs.paths import EXPERIMENT_PATH

def parse_args():
    parser = argparse.ArgumentParser(description="This is a script for safe AL")
    # general AL experiment arguments
    parser.add_argument("--experiment_output_dir", default=(EXPERIMENT_PATH / 'safe_AL').as_posix(), type=str)
    parser.add_argument("--experiment_input_dir", default=(EXPERIMENT_PATH / 'data').as_posix(), type=str)
    parser.add_argument("--experiment_idx", default=0, type=int)
    parser.add_argument("--model_path", default=(EXPERIMENT_PATH / 'Amor-Struct-GP-pretrained-weights' / 'main_state_dict_paper.pth').as_posix(), type=str)
    parser.add_argument('--acquisition_function_config', default='BasicSafePredEntropyAllConfig', type=str)
    parser.add_argument("--optimize_acquisition_by_gradient", default=False, type=string2bool)
    parser.add_argument('--validation_type', default=['MAE', 'RMSE', 'NEG_LOG_LIKELI'], nargs='+', type=lambda name: ValidationType[name.upper()], choices=list(ValidationType))
    parser.add_argument("--oracle", default='BraninHoo', type=str)
    parser.add_argument("--constraint_on_y", default=True, type=string2bool)
    parser.add_argument("--plot_iterations", default=True, type=string2bool, help="this only works for 1 or 2 dim input though")
    parser.add_argument("--save_results", default=True, type=string2bool)
    # pool arguments for standard setting
    parser.add_argument("--n_pool", default=5000, type=int)
    parser.add_argument("--n_data_initial", default=10, type=int)
    parser.add_argument("--n_steps", default=50, type=int)
    parser.add_argument("--n_data_test", default=200, type=int)
    parser.add_argument("--query_noisy", default=True, type=string2bool)
    parser.add_argument("--noise_level", default=0.1, type=float, help="noise level (std)")
    # data arguments
    parser.add_argument("--safe_lower", default=[0.0], type=float, nargs='+')
    parser.add_argument("--safe_upper", default=[np.inf], type=float, nargs='+')
    args = parser.parse_args()
    return args

def experiment(args):
    exp_idx = args.experiment_idx
    
    n_data_initial = args.n_data_initial
    n_pool = args.n_pool
    n_steps = args.n_steps
    n_data_test = args.n_data_test
    # experiment settings
    safe_lower = args.safe_lower
    safe_upper = args.safe_upper
    acq_config_name = args.acquisition_function_config
    val_type = args.validation_type
    oracle_str = args.oracle
    constraint_on_y = args.constraint_on_y
    save_results = args.save_results
    
    # create an acquisition function
    function_config = ConfigPicker.pick_acquisition_function_config(acq_config_name)(
        alpha=0.05,
        safety_thresholds_lower=safe_lower,
        safety_thresholds_upper=safe_upper
    )
    acq_func = AcquisitionFunctionFactory.build(function_config)
    opt_acq_by_grad = args.optimize_acquisition_by_gradient and acq_func.support_gradient_based_optimization
    
    # need a pool here
    if constraint_on_y:      
        assert acq_func.number_of_constraints == 1  
        oracle = OracleNormalizer(
            getattr(oracles, oracle_str)(args.noise_level)
        )
        oracle.set_normalization_by_sampling( target_scale= np.sqrt(1.0 - args.noise_level**2) )
        pool = PoolFromOracle(oracle, seed=2024 + exp_idx, set_seed=True)
        pool.discretize_random(n_pool)
    else:
        if oracle_str in ['GasTransmissionCompressorDesign', 'LSQ', 'Simionescu', 'Townsend']:
            pool = getattr(pools, oracle_str + 'Pool')(args.noise_level, seed=2024 + exp_idx, set_seed=True)
            assert acq_func.number_of_constraints == len(pool.safety_oracle)
            pool.discretize_random(n_pool)
        elif oracle_str.startswith('HighPressureFluidSystem'):
            assert acq_func.number_of_constraints == 1
            pool = pools.HighPressureFluidSystemPool(args.noise_level, seed=2024 + exp_idx, set_seed=True, trajectory_sampler=True)
            pool.discretize_random(n_pool) # get n_pool time series trajectories
        elif oracle_str.lower() == 'lgbb':
            dataset = LGBB(base_path= Path(args.experiment_input_dir, 'lgbb'))
            dataset.load_data_set()
            dataset_z = LGBB(base_path= Path(args.experiment_input_dir, 'lgbb'))
            dataset_z.output_type = LGBB_OT.PITCH
            dataset_z.filter_outlier = False
            dataset_z.load_data_set()
            pool = PoolWithSafetyFromData(
                dataset.x,
                dataset.y,
                dataset_z.y,
                data_is_noisy=True,
                seed=2024 + exp_idx,
                set_seed=True
            )
            assert acq_func.number_of_constraints == 1
        elif oracle_str.lower().startswith('engine'):
            from alef.enums.data_sets_enums import InputPreprocessingType
            dataset = Engine2(Path(args.experiment_input_dir) / 'engine')
            dataset.input_preprocessing_type = InputPreprocessingType.MIN_MAX_NORMALIZATION
            dataset.constrain_input = False
            dataset.load_data_set()
            if oracle_str.lower() == 'engine':
                _x = dataset.x[:, :2]
            elif oracle_str.lower() == 'engine3d':
                _x = dataset.x[:, [0,1,3]]
            elif oracle_str.lower() == 'engine4d':
                _x = dataset.x
            pool = PoolWithSafetyFromData(
                _x,
                dataset.y[:, [4]],
                dataset.y[:, [1]] - 0.2,
                data_is_noisy=True,
                seed=2024 + exp_idx,
                set_seed=True
            )
            assert acq_func.number_of_constraints == 1
        elif oracle_str.lower() == 'powerplant':
            dataset = PowerPlant(base_path= Path(args.experiment_input_dir, 'power_plant'))
            dataset.load_data_set()
            pool = PoolWithSafetyFromData(
                dataset.x,
                dataset.y,
                - dataset.y,
                data_is_noisy=True,
                seed=2024 + exp_idx,
                set_seed=True
            )
            assert acq_func.number_of_constraints == 1
        else:
            oracle = OracleNormalizer(
                getattr(oracles, oracle_str)(args.noise_level)
            )
            oracle.set_normalization_by_sampling( target_scale= np.sqrt(1.0 - args.noise_level**2) )
            pool = PoolWithSafetyFromOracle(oracle, [oracle], seed=2024 + exp_idx, set_seed=True)
            assert acq_func.number_of_constraints == len(pool.safety_oracle)
            pool.discretize_random(n_pool)

    if opt_acq_by_grad:
        pool.set_query_non_exist(True)

    # initial data & test data
    print('generate constrained data')
    if oracle_str in CONSTRAINED_BOXES.keys():
        a, box_width = CONSTRAINED_BOXES[oracle_str]
        data_init = pool.get_random_constrained_data_in_box(n_data_initial, a, box_width, noisy=True, constraint_lower=np.array(safe_lower), constraint_upper=safe_upper)
    else:
        data_init = pool.get_random_constrained_data(n_data_initial, noisy=True, constraint_lower=np.array(safe_lower) + INITIAL_OUTPUT_HIGHER_BY, constraint_upper=safe_upper)
    data_test = pool.get_random_constrained_data(n_data_test, noisy=True, constraint_lower=safe_lower, constraint_upper=safe_upper)

    # create kernels and models
    model_config = PaperAmortizedStructuredConfig(checkpoint_path=args.model_path)
    model = ModelFactory.build(model_config)
    model.set_kernel_list([[BaseKernelTypes.SE]]*pool.get_dimension())
    if not constraint_on_y:
        safety_models = []
        for _ in range(acq_func.number_of_constraints):
            safety_model_config = deepcopy(model_config)
            safety_model = ModelFactory.build(safety_model_config)
            safety_model.set_kernel_list([[BaseKernelTypes.SE]]*pool.get_dimension())
            safety_models.append(safety_model)
    else:
        safety_models = None

    # save settings
    exp_path = Path(args.experiment_output_dir) / (
        '%s%s_PaperAmortizedStructuredConfig'%(
            'GradBased' if opt_acq_by_grad else '',
            acq_config_name
        )
     ) / oracle_str

    if not exp_path.is_dir():
        exp_path.mkdir(parents=True, exist_ok=True)
    # save experiment setup
    with open(exp_path / f'{exp_idx}_experiment_description.txt', mode='w') as fp:
        for name, content in args.__dict__.items():
            print(f'{name}  :  {content}', file=fp)

    # initialize optimizer
    learner = PoolSafeActiveLearner(
        acq_func,
        val_type,
        constraint_on_y = constraint_on_y,
        update_by_gradient = opt_acq_by_grad,
    )
    learner.set_pool(pool)
    learner.set_model(model, safety_models=safety_models)
    
    # perform the main experiment
    learner.set_train_set(*data_init)
    learner.set_test_set(*data_test[:2])

    learner.set_do_plotting(args.plot_iterations)
    if args.plot_iterations and save_results:
        learner.save_plots_to_path(exp_path) # will save plots to specified path
    if args.save_results: # will save results to specified file later
        learner.save_experiment_summary_to_path(exp_path, f'{exp_idx}_SafeAL_result.xlsx')

    regret, _, _ = learner.learn(n_steps)


if __name__ == "__main__":
    args = parse_args()
    experiment(args)