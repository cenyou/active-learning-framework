import pytest
from alef.active_learners.active_learner_factory import ActiveLearnerFactory
from alef.configs.active_learners.oracle_active_learner_configs import (
    PredEntropyOracleActiveLearnerConfig,
    PredVarOracleActiveLearnerConfig,
    PredSigmaOracleActiveLearnerConfig,
    RandomOracleActiveLearnerConfig,
)
from alef.configs.active_learners.pool_active_learner_configs import (
    PredEntropyPoolActiveLearnerConfig,
    PredVarPoolActiveLearnerConfig,
    PredSigmaPoolActiveLearnerConfig,
    RandomPoolActiveLearnerConfig,
)
from alef.configs.active_learners.pool_active_learner_batch_configs import EntropyBatchPoolActiveLearnerConfig, RandomBatchPoolActiveLearnerConfig
from alef.enums.global_model_enums import PredictionQuantity
from alef.pools import PoolFromData, PoolFromOracle, PoolWithSafetyFromOracle, LSQPool
import numpy as np
from alef.oracles import (
    GPOracle1D,
    BraninHoo,
    OracleNormalizer,
)
from alef.models.model_factory import ModelFactory
from alef.kernels.kernel_factory import KernelFactory
from alef.configs.models.gp_model_config import GPModelFastConfig
from alef.configs.models.gp_model_marginalized_config import BasicGPModelMarginalizedConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.configs.active_learners.pool_active_learner_configs import PredVarPoolActiveLearnerConfig
from alef.enums.active_learner_enums import (
    ModelSelectionType,
    ValidationType,
    OracleALAcquisitionOptimizationType,
)
from alef.oracles.exponential_2d import Exponential2D
from alef.enums.active_learner_enums import ValidationType

from alef.active_learners.pool_safe_active_learner import PoolSafeActiveLearner, PoolSafeActiveLearnerWithCCLMetric

from alef.acquisition_functions.acquisition_function_factory import AcquisitionFunctionFactory
from alef.configs.acquisition_functions.safe_acquisition_functions.safe_pred_entropy_config import BasicSafePredEntropyAllConfig

@pytest.mark.parametrize(
    "active_learner_config_class", (
        PredEntropyPoolActiveLearnerConfig,
        PredVarPoolActiveLearnerConfig,
        PredSigmaPoolActiveLearnerConfig,
        RandomPoolActiveLearnerConfig
    )
)
def test_pool_active_learner(tmp_path, active_learner_config_class):
    n_steps = 3
    kernel_config = BasicRBFConfig(input_dimension=1)
    model_config = GPModelFastConfig(kernel_config=kernel_config)
    al_config = active_learner_config_class(validation_at=[0, n_steps-1])
    model = ModelFactory.build(model_config)
    gp_oracle = GPOracle1D(kernel_config, 0.01)
    gp_oracle.initialize(0, 1, 50)
    pool = PoolFromOracle(gp_oracle)
    pool.discretize_random(300)
    x_test, y_test = gp_oracle.get_random_data(40)
    x_init, y_init = pool.get_random_data(5, noisy=True)
    active_learner = ActiveLearnerFactory.build(al_config)
    active_learner.set_model(model)
    active_learner.set_pool(pool)
    active_learner.set_train_set(x_init, y_init)
    active_learner.set_test_set(x_test, y_test)
    # set saving
    #active_learner.set_do_plotting(True)
    #active_learner.save_plots_to_path(tmp_path)
    active_learner.save_experiment_summary_to_path(tmp_path, 'AL_result.xlsx')
    val, x_queries = active_learner.learn(n_steps)
    assert len(val) == n_steps


def test_active_learner_log_likeli():
    kernel_config = BasicRBFConfig(input_dimension=1)
    model_config = GPModelFastConfig(kernel_config=kernel_config)
    al_config = PredVarPoolActiveLearnerConfig(validation_type=ValidationType.NEG_LOG_LIKELI)
    model = ModelFactory.build(model_config)
    gp_oracle = GPOracle1D(kernel_config, 0.01)
    gp_oracle.initialize(0, 1, 50)
    pool = PoolFromOracle(gp_oracle)
    pool.discretize_random(300)
    x_test, y_test = gp_oracle.get_random_data(40)
    x_init, y_init = pool.get_random_data(20, noisy=True)
    active_learner = ActiveLearnerFactory.build(al_config)
    active_learner.set_model(model)
    active_learner.set_pool(pool)
    active_learner.set_train_set(x_init, y_init)
    active_learner.set_test_set(x_test, y_test)
    val, x_queries = active_learner.learn(3)
    assert not val.isnull().values.any()
    assert len(val) == 3


@pytest.mark.parametrize("active_learner_config_class", (EntropyBatchPoolActiveLearnerConfig, RandomBatchPoolActiveLearnerConfig))
def test_batch_active_learner(tmp_path, active_learner_config_class):
    data_set_size = 5
    batch_size = 5
    n_steps = 3
    kernel_config = RBFWithPriorConfig(input_dimension=1)
    model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01, prediction_quantity=PredictionQuantity.PREDICT_F)
    model = ModelFactory.build(model_config)
    gp_oracle = GPOracle1D(kernel_config, 0.01)
    gp_oracle.initialize(-10, 10, 30)
    pool = PoolFromOracle(gp_oracle)
    pool.discretize_random(300)
    x_test, y_test = gp_oracle.get_random_data(40)
    x_init, y_init = pool.get_random_data(data_set_size, noisy=True)
    active_learner = ActiveLearnerFactory.build(active_learner_config_class(validation_at=[0, n_steps-1]))
    active_learner.set_model(model)
    active_learner.set_pool(pool)
    active_learner.set_train_set(x_init, y_init)
    active_learner.set_test_set(x_test, y_test)
    # set saving
    active_learner.set_do_plotting(True)
    active_learner.save_plots_to_path(tmp_path)
    active_learner.save_experiment_summary_to_path(tmp_path, 'Batch_AL_result.xlsx')
    val, x_queries = active_learner.learn(n_steps)
    print(val)
    assert len(val) == n_steps
    assert len(x_queries) == data_set_size + n_steps * batch_size


@pytest.mark.parametrize(
    "active_learner_config_class", (
        PredEntropyOracleActiveLearnerConfig,
        PredVarOracleActiveLearnerConfig,
        PredSigmaOracleActiveLearnerConfig,
        RandomOracleActiveLearnerConfig
    )
)
def test_oracle_active_learner(tmp_path, active_learner_config_class):
    data_set_size = 3
    test_set_size = 200
    n_steps = 3
    oracle = Exponential2D(0.01)
    x_data, y_data = oracle.get_random_data(data_set_size)
    x_test, y_test = oracle.get_random_data(test_set_size)
    kernel_config = RBFWithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01)
    model = ModelFactory.build(model_config)
    active_learner = ActiveLearnerFactory.build(active_learner_config_class(validation_at=[0, n_steps-1]))
    # active_learner.set_do_plotting(True)
    active_learner.set_oracle(oracle)
    #active_learner.sample_train_set(data_set_size, set_seed=True)
    active_learner.sample_train_set_in_box(data_set_size, 0.2, 0.8, set_seed=True)
    active_learner.set_test_set(x_data, y_data)
    active_learner.set_model(model)
    # set saving
    active_learner.set_do_plotting(True)
    active_learner.save_plots_to_path(tmp_path)
    active_learner.save_experiment_summary_to_path(tmp_path, 'AL_result.xlsx')
    active_learner.learn(n_steps)

def test_oracle_active_learner_marginalized(tmp_path):
    data_set_size = 3
    test_set_size = 200
    n_steps = 1
    # oracle = GPOracle1D(gpflow.kernels.RBF(lengthscales=0.2),0.01)
    # oracle.initialize(0,1,2000)
    oracle = Exponential2D(0.01)
    x_data, y_data = oracle.get_random_data(data_set_size)
    x_test, y_test = oracle.get_random_data(test_set_size)
    kernel_config = RBFWithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = BasicGPModelMarginalizedConfig(kernel_config=kernel_config, observation_noise=0.01)
    model = ModelFactory.build(model_config)
    active_learner_config = PredVarOracleActiveLearnerConfig()
    active_learner_config.validation_type = [ValidationType.RMSE, ValidationType.NEG_LOG_LIKELI]
    active_learner_config.acquisiton_optimization_type = OracleALAcquisitionOptimizationType.RANDOM_SHOOTING
    active_learner = ActiveLearnerFactory.build(active_learner_config)
    active_learner.set_oracle(oracle)
    active_learner.sample_train_set(data_set_size, set_seed=True)
    active_learner.set_test_set(x_data, y_data)
    active_learner.set_model(model)
    active_learner.set_do_plotting(True)
    active_learner.save_plots_to_path(tmp_path)
    active_learner.save_experiment_summary_to_path(tmp_path, 'AL_result.xlsx')
    active_learner.learn(n_steps)

def test_pool_safe_active_learner(tmp_path):
    n_steps = 5
    oracle = OracleNormalizer(
        BraninHoo(0.1)
    )
    oracle.set_normalization_manually(60.088767740805736, 62.34134408167649)
    X, Y = oracle.get_random_data(100, noisy=False)
    Y_max = np.max(Y)
    Y_min = np.min(Y)
    safe_lower = 0.9 * Y_min + 0.1 * Y_max
    safe_upper = 0.1 * Y_min + 0.9 * Y_max

    pool = PoolWithSafetyFromOracle(oracle, [oracle])
    pool.discretize_random(2000)
    acq_func = AcquisitionFunctionFactory.build(
        BasicSafePredEntropyAllConfig(safety_thresholds_lower=safe_lower, safety_thresholds_upper=safe_upper)
    )
    model = ModelFactory.build(
        GPModelFastConfig(kernel_config = BasicRBFConfig(input_dimension=oracle.get_dimension()))
    )
    smodel = ModelFactory.build(
        GPModelFastConfig(kernel_config = BasicRBFConfig(input_dimension=oracle.get_dimension()))
    )
    data_init = pool.get_random_data(10, noisy=True)
    data_test = pool.get_random_data(100, noisy=True)

    for twsdo in [False, True]:
        learner = PoolSafeActiveLearner(
            acq_func, [ValidationType.RMSE, ValidationType.NEG_LOG_LIKELI],
            validation_at=[0, n_steps-1],
            constraint_on_y=False,
            train_with_safe_data_only=[twsdo],
        )
        learner.set_pool(pool)
        learner.set_model(model, safety_models=smodel)
        
        # perform the main experiment
        learner.set_train_set(*data_init)
        learner.set_test_set(*data_test[:2])
        learner.set_do_plotting(True)
        learner.save_plots_to_path(tmp_path)
        learner.save_experiment_summary_to_path(tmp_path, 'SafeAL_result.xlsx')
        
        _, _, _ = learner.learn(n_steps)



def test_pool_safe_active_learner_with_CCL(tmp_path):
    pool = LSQPool(0.1, set_seed=True, seed=2024)
    pool.discretize_random(2000)

    acq_func = AcquisitionFunctionFactory.build(
        BasicSafePredEntropyAllConfig(safety_thresholds_lower=[0.0]*2, safety_thresholds_upper=[100]*2)
    )
    model = ModelFactory.build(
        GPModelFastConfig(kernel_config = BasicRBFConfig(input_dimension=pool.get_dimension()))
    )
    smodel = [ModelFactory.build(
        GPModelFastConfig(kernel_config = BasicRBFConfig(input_dimension=pool.get_dimension()))
    ) for _ in range(2)]
    data_init = pool.get_random_constrained_data(10, noisy=True)
    data_test = pool.get_random_constrained_data(100, noisy=True)
    data_grid = pool.get_grid_data(30, noisy=False)

    for twsdo in [False, True]:
        learner = PoolSafeActiveLearnerWithCCLMetric(
            acq_func, ValidationType.RMSE,
            query_noisy=True,
            constraint_on_y=False,
            train_with_safe_data_only=[twsdo],
        )
        learner.set_pool(pool)
        learner.set_model(model, safety_models=smodel)
        
        # perform the main experiment
        learner.set_train_set(*data_init)
        learner.set_test_set(*data_test[:2])
        learner.initialize_safe_area_measure(*data_grid)
        learner.set_do_plotting(True)
        learner.save_plots_to_path(tmp_path)
        learner.save_experiment_summary_to_path(tmp_path, 'SafeAL_result.xlsx')
        
        _, _, _ = learner.learn(5)


if __name__ == "__main__":
    test_batch_active_learner(RandomBatchPoolActiveLearnerConfig)
