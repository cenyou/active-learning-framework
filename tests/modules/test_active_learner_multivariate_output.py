from alef.active_learners.pool_active_learner import PoolActiveLearner
from alef.pools import PoolMultioutputFromData
from alef.enums.active_learner_enums import ValidationType
from alef.utils.utils import calculate_multioutput_rmse, calculate_rmse
import numpy as np
from alef.active_learners.active_learner_factory import ActiveLearnerFactory
from alef.configs.active_learners.pool_active_learner_configs import (
    PredEntropyPoolActiveLearnerConfig,
    PredVarPoolActiveLearnerConfig,
    PredSigmaPoolActiveLearnerConfig,
    RandomPoolActiveLearnerConfig,
)


def test_pool_multivariate_output():
    x_data = np.array([[0, 0], [1, 1], [2, 2]])
    y_data = np.array([[1, 0], [1, 0], [3, 2]])
    pool = PoolMultioutputFromData( x_data, y_data, data_is_noisy=True )
    for i in range(0, x_data.shape[0]):
        y = pool.query(x_data[i])
        assert np.array_equal(y, y_data[i])
        assert y.shape[0] == 2
    pool.set_data(x_data, y_data)
    y = pool.query([0, 0])
    _, y_data_left_in_pool = pool.get_full_data()
    assert np.array_equal(y_data_left_in_pool[0], y_data[1])


def test_multivariate_rmse():
    y_data = np.array([[1, 0], [1, 0], [3, 2]])
    pred_y = np.array([[0.9, 0.1], [1.1, 0.2], [3.1, 2.15]])
    single_y = np.array([[1], [1], [3]])
    single_pred_y = np.array([[0.9], [1.1], [3.1]])
    rmses = calculate_multioutput_rmse(pred_y, y_data)
    print(rmses)
    assert len(rmses) == y_data.shape[1]
    single_rmse = calculate_rmse(single_pred_y, single_y)
    assert single_rmse == rmses[0]
