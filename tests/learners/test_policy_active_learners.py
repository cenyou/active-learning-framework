from alef.active_learners.active_learner_factory import ActiveLearnerFactory
from alef.configs.active_learners.oracle_policy_active_learner_configs import PytestOraclePolicyActiveLearnerConfig
from alef.configs.active_learners.pool_policy_active_learner_configs import PytestPoolPolicyActiveLearnerConfig
from alef.configs.active_learners.oracle_policy_safe_active_learner_configs import PytestOraclePolicySafeActiveLearnerConfig
from alef.configs.active_learners.pool_policy_safe_active_learner_configs import PytestPoolPolicySafeActiveLearnerConfig

from alef.pools import PoolFromData, PoolFromOracle, PoolWithSafetyFromOracle, LSQPool
from alef.oracles import (
    GPOracle1D,
    BraninHoo,
    OracleNormalizer,
)
from alef.models.model_factory import ModelFactory
from alef.configs.models.gp_model_config import GPModelFastConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig



def test_oracle_amortized_active_learner(tmp_path):
    data_set_size = 3
    test_set_size = 200
    n_steps = 3
    oracle = BraninHoo(0.01)
    x_test, y_test = oracle.get_random_data(test_set_size)
    kernel_config = RBFWithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01)
    model = ModelFactory.build(model_config)
    al_config = PytestOraclePolicyActiveLearnerConfig(validation_at=[0, n_steps-1], policy_dimension=oracle.get_dimension())
    active_learner = ActiveLearnerFactory.build(al_config)
    # active_learner.set_do_plotting(True)
    active_learner.set_oracle(oracle)
    active_learner.sample_train_set(data_set_size, set_seed=True)
    active_learner.set_test_set(x_test, y_test)
    active_learner.set_model(model)
    # set saving
    active_learner.set_do_plotting(True)
    active_learner.save_plots_to_path(tmp_path)
    active_learner.save_experiment_summary_to_path(tmp_path, 'AL_result.xlsx')
    active_learner.learn(n_steps)

def test_pool_amortized_active_learner(tmp_path):
    data_set_size = 3
    test_set_size = 200
    n_steps = 3
    oracle = BraninHoo(0.01)
    x_init, y_init = oracle.get_random_data(data_set_size)
    x_test, y_test = oracle.get_random_data(test_set_size)
    x_plot, y_plot = oracle.get_grid_data(500)
    kernel_config = RBFWithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01)
    model = ModelFactory.build(model_config)
    al_config = PytestPoolPolicyActiveLearnerConfig(validation_at=[0, n_steps-1], policy_dimension=oracle.get_dimension())
    active_learner = ActiveLearnerFactory.build(al_config)
    pool = PoolFromOracle(oracle)
    pool.discretize_random(500)

    active_learner.set_pool(pool)
    active_learner.set_train_set(x_init, y_init)
    active_learner.set_test_set(x_test, y_test)
    active_learner.set_plot_data(x_plot, y_plot)
    active_learner.set_model(model)
    # set saving
    active_learner.set_do_plotting(True)
    active_learner.save_plots_to_path(tmp_path)
    active_learner.save_experiment_summary_to_path(tmp_path, 'AL_result.xlsx')
    active_learner.learn(n_steps)

def test_oracle_amortized_safe_active_learner(tmp_path):
    data_set_size = 3
    test_set_size = 200
    n_steps = 3
    oracle = BraninHoo(0.01)
    kernel_config = RBFWithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01)
    model = ModelFactory.build(model_config)
    for twsdo in [False, True]:
        al_config = PytestOraclePolicySafeActiveLearnerConfig(
            constraint_on_y=False,
            validation_at=[0, n_steps-1],
            train_with_safe_data_only=[twsdo],
            policy_dimension=oracle.get_dimension()
        )
        active_learner = ActiveLearnerFactory.build(al_config)
        # active_learner.set_do_plotting(True)
        active_learner.set_oracle(oracle, [oracle])
        active_learner.sample_constrained_train_set(data_set_size, set_seed=True, constraint_lower=0.0)
        active_learner.sample_constrained_test_set(test_set_size, set_seed=False, constraint_lower=0.0)
        active_learner.set_model(model)
        # set saving
        active_learner.set_do_plotting(True)
        active_learner.save_plots_to_path(tmp_path)
        active_learner.save_experiment_summary_to_path(tmp_path, 'AL_result.xlsx')
        active_learner.learn(n_steps)

def test_pool_amortized_safe_active_learner(tmp_path):
    data_set_size = 3
    test_set_size = 200
    n_steps = 3
    oracle = BraninHoo(0.01)
    x_init, y_init = oracle.get_random_data(data_set_size)
    z_init = y_init.copy()
    x_test, y_test = oracle.get_random_data(test_set_size)
    x_plot, y_plot = oracle.get_grid_data(500)
    z_plot = y_plot.copy()
    kernel_config = RBFWithPriorConfig(input_dimension=oracle.get_dimension())
    model_config = GPModelFastConfig(kernel_config=kernel_config, observation_noise=0.01)
    model = ModelFactory.build(model_config)
    for twsdo in [False, True]:
        al_config = PytestPoolPolicySafeActiveLearnerConfig(
            constraint_on_y=False,
            validation_at=[0, n_steps-1],
            train_with_safe_data_only=[twsdo],
            policy_dimension=oracle.get_dimension()
        )
        active_learner = ActiveLearnerFactory.build(al_config)
        pool = PoolWithSafetyFromOracle(oracle, [oracle])
        pool.discretize_random(500)
        # active_learner.set_do_plotting(True)
        active_learner.set_pool(pool)
        active_learner.set_train_set(x_init, y_init, z_init)
        active_learner.set_test_set(x_test, y_test)
        active_learner.set_plot_data(x_plot, y_plot, z_plot)
        active_learner.set_model(model)
        # set saving
        active_learner.set_do_plotting(True)
        active_learner.save_plots_to_path(tmp_path)
        active_learner.save_experiment_summary_to_path(tmp_path, 'AL_result.xlsx')
        active_learner.learn(n_steps)



