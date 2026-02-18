from pydantic.main import BaseConfig
import pytest
from alef import oracles
import gpflow
from alef.bayesian_optimization.bayesian_optimizer_factory import BayesianOptimizerFactory
from alef.pools.pool_from_oracle import PoolFromOracle
import numpy as np
from alef.configs.bayesian_optimization.bayesian_optimizer_configs import (
    BOExpectedImprovementConfig,
    BOGPUCBConfig,
    BOIntegratedExpectedImprovementConfig,
)
from alef.oracles import BraninHoo, GPOracle1D, OracleNormalizer
from alef.models.model_factory import ModelFactory
from alef.kernels.kernel_factory import KernelFactory
from alef.configs.models.gp_model_config import BasicGPModelConfig, GPModelFastConfig
from alef.configs.models.gp_model_marginalized_config import BasicGPModelMarginalizedConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.configs.kernels.hhk_configs import HHKFourLocalDefaultConfig
from alef.configs.active_learners.pool_active_learner_configs import PredVarPoolActiveLearnerConfig
from alef.oracles.safe_test_func import SafeTestFunc
from alef.configs.models.gp_model_laplace_config import BasicGPModelLaplaceConfig
from alef.oracles.exponential_2d import Exponential2D
from alef.bayesian_optimization.bayesian_optimizer import BayesianOptimizer
from alef.bayesian_optimization.hedge_bayesian_optimizer_oracle import HedgeBayesianOptimizerOracle
from alef.enums.bayesian_optimization_enums import ValidationType, AcquisitionFunctionType, AcquisitionOptimizationType
from alef.oracles.mnist_svm import MnistSVM
from alef.bayesian_optimization.bayesian_optimizer_constrained import ConstrainedBayesianOptimizer
from alef.acquisition_functions.acquisition_function_factory import AcquisitionFunctionFactory
from alef.configs.acquisition_functions.safe_acquisition_functions.safe_opt_config import BasicSafeOptConfig


@pytest.mark.parametrize("optimizer_config_class", (BOIntegratedExpectedImprovementConfig, BOExpectedImprovementConfig, BOGPUCBConfig))
def test_oracle_bayesian_optimization(optimizer_config_class):
    data_set_size = 4
    n_steps = 3
    oracle = SafeTestFunc(observation_noise=0.01)
    # oracle.load_data()
    optimizer_config = optimizer_config_class()
    kernel_config = RBFWithPriorConfig(input_dimension=1)
    if isinstance(optimizer_config, BOIntegratedExpectedImprovementConfig):
        model_config = BasicGPModelMarginalizedConfig(kernel_config=kernel_config, observation_noise=0.01)
    else:
        model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01)
    kernel = KernelFactory.build(kernel_config)
    model = ModelFactory.build(model_config)
    optimizer = BayesianOptimizerFactory.build(optimizer_config)
    optimizer.set_oracle(oracle)
    #optimizer.set_do_plotting(True)
    optimizer.sample_train_set(data_set_size, seed=100, set_seed=True)
    optimizer.set_max_value_for_validation(1.0)
    # active_learner.sample_from_oracle_to_find_max_value(10000)
    optimizer.set_model(model)
    regret, _ = optimizer.maximize(n_steps)


def test_hedge_oracle_bayesian_optimization():
    data_set_size = 3
    n_steps = 3
    oracle = SafeTestFunc(observation_noise=0.01)
    # oracle.load_data()
    kernel_config = BasicRBFConfig(input_dimension=1)
    model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01, train_likelihood_variance=False)
    kernel = KernelFactory.build(kernel_config)
    model = ModelFactory.build(model_config)
    kernel_config_2 = HHKFourLocalDefaultConfig(input_dimension=1)
    model_config_2 = GPModelFastConfig(kernel_config=kernel_config_2, observation_noise=0.01, train_likelihood_variance=False)
    kernel_2 = KernelFactory.build(kernel_config_2)
    model_2 = ModelFactory.build(model_config_2)
    optimizer = HedgeBayesianOptimizerOracle(
        AcquisitionFunctionType.GP_UCB, ValidationType.MAX_OBSERVED, AcquisitionOptimizationType.EVOLUTIONARY, False
    )
    optimizer.set_oracle(oracle)
    optimizer.sample_train_set(data_set_size, seed=120, set_seed=True)
    optimizer.set_max_value_for_validation(1.0)
    # active_learner.sample_from_oracle_to_find_max_value(10000)
    optimizer.set_models([model, model_2])
    regret, _ = optimizer.maximize(n_steps)


def test_safe_bayes_optimizer(tmp_path):
    oracle = OracleNormalizer(
        BraninHoo(0.1)
    )
    oracle.set_normalization_manually(60.088767740805736, 62.34134408167649)
    X, Y = oracle.get_random_data(100, noisy=False)
    Y_max = np.max(Y)
    Y_min = np.min(Y)
    safe_lower = 0.9 * Y_min + 0.1 * Y_max
    safe_upper = 0.1 * Y_min + 0.9 * Y_max

    pool = PoolFromOracle(oracle)
    pool.discretize_random(2000)
    acq_func = AcquisitionFunctionFactory.build(
        BasicSafeOptConfig(safety_thresholds_lower=safe_lower, safety_thresholds_upper=safe_upper)
    )
    model = ModelFactory.build(
        GPModelFastConfig(kernel_config = BasicRBFConfig(input_dimension=oracle.get_dimension()))
    )
    data_init = pool.get_random_data(10, noisy=True)
    
    optimizer = ConstrainedBayesianOptimizer(
        acq_func, ValidationType.SIMPLE_REGRET,
        do_plotting=False,
        query_noisy=True,
        constraint_on_y=True,
        save_results=True,
        experiment_path=tmp_path
    )
    optimizer.set_pool(pool)
    optimizer.set_model(model, safety_models=None)
    optimizer.set_max_value_for_validation(Y_max)
    
    # perform the main experiment
    optimizer.set_train_set(*data_init)
    _, _, _ = optimizer.learn(5)


if __name__ == "__main__":
    test_oracle_bayesian_optimization()
