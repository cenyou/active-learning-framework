import numpy as np
from scipy.stats import tstd
from alef.oracles import Sinus, BraninHoo, CartPoleConstrained, Simionescu, OracleNormalizer
from alef.data_sets.pytest_set import PytestSet, PytestMOSet
from alef.pools import (
    PoolFromOracle, PoolWithSafetyFromOracle, PoolWithSafetyFromConstrainedOracle,
    PoolFromDataSet, PoolWithSafetyFromDataSet,
    PoolMultioutputFromData,
    TransferPoolFromPools, MultitaskPoolFromPools,
)
from alef.utils.utils import row_wise_compare, row_wise_unique

############
### PoolFromOracle
############
def test_pool_from_oracle_basic():
    oracle = Sinus(1e-6)
    pool = PoolFromOracle(oracle, seed=123, set_seed=True)
    pool.discretize_random(200)
    assert pool.get_dimension() == 1
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == 200
    y = pool.query(xx[0], noisy=False)
    assert y == oracle.query(xx[0], noisy=False)
    assert pool.possible_queries().shape[0] == 200

    pool.set_replacement(False)
    y = pool.query(xx[10], noisy=False)
    assert y == oracle.query(xx[10], noisy=False)
    assert pool.possible_queries().shape[0] == 199

    y = pool.batch_query(xx[20:30], noisy=False)
    assert y.shape == (10,)
    assert np.allclose(y, oracle.batch_query(xx[20:30], noisy=False))
    assert pool.possible_queries().shape[0] == 189

def test_pool_from_oracle_get_unconstrained_data():
    oracle = Sinus(1e-6)
    pool = PoolFromOracle(oracle, seed=123, set_seed=True)

    X, Y = pool.get_random_data(100, False)
    assert X.shape[0] == 100
    for i in range(100):
        assert Y[i] == oracle.query(X[i], False)
    
    X, Y = pool.get_random_data_in_box(10, -0.5, 1, False)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
    assert X.min() >= -0.5
    assert X.max() <= 0.5 # -0.5 + 1

def test_pool_from_oracle_get_constrained_data():
    oracle = Sinus(1e-6)
    pool = PoolFromOracle(oracle, seed=123, set_seed=True)

    X, Y = pool.get_random_constrained_data(10, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
    assert Y.min() >= -0.5
    assert Y.max() <= 0.6

    X, Y = pool.get_random_constrained_data_in_box(10, -0.1, 1, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
    assert X.min() >= -0.1
    assert X.max() <= 0.9
    assert Y.min() >= -0.5
    assert Y.max() <= 0.6


############
### PoolWithSafetyFromOracle
############
def test_pool_with_safety_from_oracle_basic():
    oracle = BraninHoo(1e-6)
    safety_oracles = [
        BraninHoo(1e-6, np.array([1, 1, 2, 4, 8, 1])),
        BraninHoo(1e-6, np.array([1.1, 2, 1.8, 3.4, 8.5, 2]))
    ]
    pool = PoolWithSafetyFromOracle(oracle, safety_oracles, seed=123, set_seed=True)
    pool.discretize_random(200)
    assert pool.get_dimension() == 2
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == 200
    y, z = pool.query(xx[0], noisy=False)
    assert y == oracle.query(xx[0], noisy=False)
    assert np.all(z == np.array([so.query(xx[0], noisy=False) for so in safety_oracles]))
    assert z.shape == (2,)
    assert pool.possible_queries().shape[0] == 200

    pool.set_replacement(False)
    y, z = pool.query(xx[10], noisy=False)
    assert y == oracle.query(xx[10], noisy=False)
    assert np.all(z == np.array([so.query(xx[10], noisy=False) for so in safety_oracles]))
    assert pool.possible_queries().shape[0] == 199

    y, z = pool.batch_query(xx[20:30], noisy=False)
    assert y.shape == (10,)
    assert z.shape == (10, 2)
    assert np.allclose(y, oracle.batch_query(xx[20:30], noisy=False))
    assert np.allclose(z[:,0], safety_oracles[0].batch_query(xx[20:30], noisy=False))
    assert np.allclose(z[:,1], safety_oracles[1].batch_query(xx[20:30], noisy=False))
    assert pool.possible_queries().shape[0] == 189
    
def test_pool_with_safety_from_oracle_get_unconstrained_data():
    oracle = BraninHoo(1e-6)
    safety_oracles = [
        BraninHoo(1e-6, np.array([1, 1, 2, 4, 8, 1])),
        BraninHoo(1e-6, np.array([1.1, 2, 1.8, 3.4, 8.5, 2]))
    ]
    pool = PoolWithSafetyFromOracle(oracle, safety_oracles, seed=123, set_seed=True)

    X, Y, Z = pool.get_random_data(100, False)
    assert X.shape[0] == 100
    for i in range(100):
        assert Y[i] == oracle.query(X[i], False)
        assert np.all(Z[i] == np.array([so.query(X[i], noisy=False) for so in safety_oracles]))
    
    X, Y, Z = pool.get_random_data_in_box(10, -0.5, 1, False)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
        assert np.all(Z[i] == np.array([so.query(X[i], noisy=False) for so in safety_oracles]))
    assert X.min() >= -0.5
    assert X.max() <= 0.5 # -0.5 + 1

def test_pool_with_safety_from_oracle_get_constrained_data():
    oracle = BraninHoo(1e-6)
    o1 = OracleNormalizer(BraninHoo(1e-6, np.array([1, 1, 2, 4, 8, 1])))
    o1.set_normalization_manually(0.0, 300)
    o2 = OracleNormalizer(BraninHoo(1e-6, np.array([1.1, 2, 1.8, 3.4, 8.5, 2])))
    o2.set_normalization_manually(0.0, 300)
    safety_oracles = [o1, o2]
    pool = PoolWithSafetyFromOracle(oracle, safety_oracles, seed=123, set_seed=True)

    X, Y, Z = pool.get_random_constrained_data(10, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
        assert np.all(Z[i] == np.array([so.query(X[i], noisy=False) for so in safety_oracles]))
    assert Z.min() >= -0.5
    assert Z.max() <= 0.6

    X, Y, Z = pool.get_random_constrained_data_in_box(10, -0.1, 1, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        assert Y[i] == oracle.query(X[i], False)
        assert np.all(Z[i] == np.array([so.query(X[i], noisy=False) for so in safety_oracles]))
    assert X.min() >= -0.1
    assert X.max() <= 0.9
    assert Z.min() >= -0.5
    assert Z.max() <= 0.6


############
### PoolWithSafetyFromConstrainedOracle
############
def test_pool_with_safety_from_oracle_basic():
    oracle = Simionescu(1e-6)
    pool = PoolWithSafetyFromConstrainedOracle(oracle, seed=123, set_seed=True)
    pool.discretize_random(200)
    assert pool.get_dimension() == 2
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == 200
    y, z = pool.query(xx[0], noisy=False)
    yo, zo = oracle.query(xx[0], noisy=False)
    assert y == yo
    assert np.all(z == zo)
    assert z.shape == (1,)
    assert pool.possible_queries().shape[0] == 200

    pool.set_replacement(False)
    y, z = pool.query(xx[10], noisy=False)
    yo, zo = oracle.query(xx[10], noisy=False)
    assert y == yo
    assert np.all(z == zo)
    assert pool.possible_queries().shape[0] == 199

    y, z = pool.batch_query(xx[20:30], noisy=False)
    yo, zo = oracle.batch_query(xx[20:30], noisy=False)
    assert y.shape == (10,)
    assert z.shape == (10, 1)
    assert np.allclose(y, yo)
    assert np.allclose(z, zo)
    assert pool.possible_queries().shape[0] == 189
    
def test_pool_with_safety_from_oracle_get_unconstrained_data():
    oracle = Simionescu(1e-6)
    pool = PoolWithSafetyFromConstrainedOracle(oracle, seed=123, set_seed=True)

    X, Y, Z = pool.get_random_data(100, False)
    assert X.shape[0] == 100
    for i in range(10):
        yo, zo = oracle.query(X[i], False)
        assert Y[i] == yo
        assert np.all(Z[i] == zo)
    
    X, Y, Z = pool.get_random_data_in_box(10, -0.5, 1, False)
    for i in range(10):
        yo, zo = oracle.query(X[i], False)
        assert Y[i] == yo
        assert np.all(Z[i] == zo)
    assert X.min() >= -0.5
    assert X.max() <= 0.5 # -0.5 + 1

def test_pool_with_safety_from_oracle_get_constrained_data():
    oracle = Simionescu(1e-6)
    pool = PoolWithSafetyFromConstrainedOracle(oracle, seed=123, set_seed=True)

    X, Y, Z = pool.get_random_constrained_data(10, False, constraint_lower=0.5, constraint_upper=1.5)
    for i in range(10):
        yo, zo = oracle.query(X[i], False)
        assert Y[i] == yo
        assert np.all(Z[i] == zo)
    assert Z.min() >= 0.5
    assert Z.max() <= 1.5

    X, Y, Z = pool.get_random_constrained_data_in_box(10, -0.1, 1, False, constraint_lower=0.5, constraint_upper=1.5)
    for i in range(10):
        yo, zo = oracle.query(X[i], False)
        assert Y[i] == yo
        assert np.all(Z[i] == zo)
    assert X.min() >= -0.1
    assert X.max() <= 0.9
    assert Z.min() >= 0.5
    assert Z.max() <= 1.5

############
### PoolFromDataSet
############
def test_pool_from_data_basic():
    dataset = PytestSet()
    dataset.load_data_set()
    pool = PoolFromDataSet(dataset, [0,1,2], data_is_noisy=False)
    assert pool.get_dimension() == 3
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == dataset.length
    y = pool.query(xx[0], noisy=False)
    assert y == dataset.y[0,0]
    assert pool.possible_queries().shape[0] == dataset.length

    pool.set_replacement(False)
    y = pool.query(xx[9], noisy=False)
    assert y == dataset.y[9,0]
    assert pool.possible_queries().shape[0] == dataset.length-1

    y = pool.batch_query(xx[1:6], noisy=False)
    assert y.shape == (5,)
    assert np.allclose(y, dataset.y[1:6,0])
    assert pool.possible_queries().shape[0] == dataset.length-6
    
def test_pool_from_data_get_unconstrained_data():
    dataset = PytestSet()
    dataset.load_data_set()
    pool = PoolFromDataSet(dataset, [0,1,2], data_is_noisy=False)
    
    xx, yy = pool.get_random_data(50, False)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y, yy)
    assert np.all(mask_x == mask_y)
    assert mask_x.sum() == 50
    for i in range(50):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask])

    xx, yy = pool.get_random_data_in_box(10, -0.5, 1, False)
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask])
    assert xx.min() >= -0.5
    assert xx.max() <= 0.5 # -0.5 + 1

def test_pool_from_data_get_constrained_data():
    dataset = PytestSet()
    dataset.load_data_set()
    pool = PoolFromDataSet(dataset, [0,1,2], data_is_noisy=False)
    
    xx, yy = pool.get_random_constrained_data(10, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask])
    assert yy.min() >= -0.5
    assert yy.max() <= 0.6

    xx, yy = pool.get_random_constrained_data_in_box(10, -0.5, 1, False, constraint_lower=-0.5, constraint_upper=0.6)
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask])
    assert xx.min() >= -0.5
    assert xx.max() <= 0.5
    assert yy.min() >= -0.5
    assert yy.max() <= 0.6


############
### PoolWithSafetyFromDataSet
############
def test_pool_with_safety_from_data_basic():
    dataset = PytestMOSet()
    dataset.load_data_set()
    pool = PoolWithSafetyFromDataSet(dataset, [0,1,2], [0], [1], data_is_noisy=False)
    assert pool.get_dimension() == 3
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == dataset.length
    y, z = pool.query(xx[0], noisy=False)
    assert y == dataset.y[0,0]
    assert z == dataset.y[0,1]
    assert pool.possible_queries().shape[0] == dataset.length

    pool.set_replacement(False)
    y, z = pool.query(xx[9], noisy=False)
    assert y == dataset.y[9,0]
    assert z == dataset.y[9,1]
    assert pool.possible_queries().shape[0] == dataset.length-1

    y, z = pool.batch_query(xx[1:6], noisy=False)
    assert y.shape == (5,)
    assert z.shape == (5, 1)
    assert np.allclose(y, dataset.y[1:6,0])
    assert np.allclose(z, dataset.y[1:6,1, None])
    assert pool.possible_queries().shape[0] == dataset.length-6
    
def test_pool_with_safety_from_data_get_unconstrained_data():
    dataset = PytestMOSet()
    dataset.load_data_set()
    pool = PoolWithSafetyFromDataSet(dataset, [0,1,2], [0], [1], data_is_noisy=False)
    
    xx, yy, zz = pool.get_random_data(50, False)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y[:,0,None], yy)
    mask_z = row_wise_compare(dataset.y[:,1,None], zz)
    assert np.all(mask_x == mask_y)
    assert np.all(mask_y == mask_z)
    assert mask_x.sum() == 50
    for i in range(50):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask,0])
        assert np.squeeze(zz[i]) == np.squeeze(dataset.y[mask,1])

    xx, yy, zz = pool.get_random_data_in_box(10, [-1, -0.5, 0], [0.5, 1, 0.5], False)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y[:,0,None], yy)
    mask_z = row_wise_compare(dataset.y[:,1,None], zz)
    assert np.all(mask_x == mask_y)
    assert np.all(mask_y == mask_z)
    assert mask_x.sum() == 10
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask,0])
        assert np.squeeze(zz[i]) == np.squeeze(dataset.y[mask,1])
    assert np.all(xx.min(axis=0) >= [-1, -0.5, 0])
    assert np.all(xx.max(axis=0) <= [-0.5, 0.5, 0.5])

def test_pool_with_safety_from_data_get_constrained_data():
    dataset = PytestMOSet()
    dataset.load_data_set()
    pool = PoolWithSafetyFromDataSet(dataset, [0,1,2], [0], [1], data_is_noisy=False)
    
    xx, yy, zz = pool.get_random_constrained_data(10, False, constraint_lower=-0.5, constraint_upper=0.6)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y[:,0,None], yy)
    mask_z = row_wise_compare(dataset.y[:,1,None], zz)
    assert np.all(mask_x == mask_y)
    assert np.all(mask_y == mask_z)
    assert mask_x.sum() == 10
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask,0])
        assert np.squeeze(zz[i]) == np.squeeze(dataset.y[mask,1])
    assert zz.min() >= -0.5
    assert zz.max() <= 0.6

    xx, yy, zz = pool.get_random_constrained_data_in_box(10, [-1, -0.5, 0], [0.5, 1, 0.5], False, constraint_lower=-0.5, constraint_upper=0.6)
    mask_x = row_wise_compare(dataset.x, xx)
    mask_y = row_wise_compare(dataset.y[:,0,None], yy)
    mask_z = row_wise_compare(dataset.y[:,1,None], zz)
    assert np.all(mask_x == mask_y)
    assert np.all(mask_y == mask_z)
    assert mask_x.sum() == 10
    for i in range(10):
        mask = row_wise_compare(dataset.x, xx[i, :])
        assert np.squeeze(yy[i]) == np.squeeze(dataset.y[mask,0])
        assert np.squeeze(zz[i]) == np.squeeze(dataset.y[mask,1])
    assert np.all(xx.min(axis=0) >= [-1, -0.5, 0])
    assert np.all(xx.max(axis=0) <= [-0.5, 0.5, 0.5])
    assert zz.min() >= -0.5
    assert zz.max() <= 0.6


############
### PoolMultioutputFromData
############
def test_pool_multioutput_from_data_basic():
    dataset = PytestMOSet()
    dataset.load_data_set()
    X, Y = dataset.get_complete_dataset()

    pool = PoolMultioutputFromData(X, Y, data_is_noisy=False)
    assert pool.get_dimension() == 3
    
    pool.set_replacement(True)
    xx = pool.possible_queries()
    assert xx.shape[0] == dataset.length
    y = pool.query(xx[0], noisy=False)
    assert np.allclose(y, Y[0, :])
    assert pool.possible_queries().shape[0] == dataset.length

    pool.set_replacement(False)
    y = pool.query(xx[9], noisy=False)
    assert np.allclose(y, Y[9, :])
    assert pool.possible_queries().shape[0] == dataset.length-1

    y = pool.batch_query(xx[1:6], noisy=False)
    assert y.shape == (5, 2)
    assert np.allclose(y, dataset.y[1:6, :])
    assert pool.possible_queries().shape[0] == dataset.length-6

def test_pool_multioutput_from_data_get_unconstrained_data():
    dataset = PytestMOSet()
    dataset.load_data_set()
    X, Y = dataset.get_complete_dataset()

    pool = PoolMultioutputFromData(X, Y, data_is_noisy=False)

    xx, yy = pool.get_random_data(50, False)
    mask_x = row_wise_compare(X, xx)
    mask_y = row_wise_compare(Y, yy)
    assert np.all(mask_x == mask_y)
    assert mask_x.sum() == 50
    for i in range(50):
        mask = row_wise_compare(X, xx[i, :])
        assert np.allclose(yy[i], Y[mask, :])

    xx, yy = pool.get_random_data_in_box(10, [-1, -0.5, 0], [0.5, 1, 0.5], False)
    mask_x = row_wise_compare(X, xx)
    mask_y = row_wise_compare(Y, yy)
    assert np.all(mask_x == mask_y)
    assert mask_x.sum() == 10
    for i in range(10):
        mask = row_wise_compare(X, xx[i, :])
        assert np.allclose(yy[i], Y[mask, :])
    assert np.all(xx.min(axis=0) >= [-1, -0.5, 0])
    assert np.all(xx.max(axis=0) <= [-0.5, 0.5, 0.5])


############
### TransferPoolFromPools
############
def test_transfer_pool():
    ps = PoolFromOracle(Sinus(0.01))
    pt = PoolFromOracle(Sinus(0.01))
    ps.discretize_random(300)
    pt.discretize_random(300)
    pool = TransferPoolFromPools(ps, pt)
    pool.set_task_mode(False)
    pool.set_replacement(True)
    x_pool = pool.possible_queries()
    y = pool.query(x_pool[0], noisy=False)
    assert pool.possible_queries().shape[0] == 300
    pool.set_replacement(False)
    x_pool = pool.possible_queries()
    y = pool.query(x_pool[0], noisy=False)
    assert pool.possible_queries().shape[0] == 299

    pool.set_task_mode(True)
    pool.set_replacement(True)
    x_pool = pool.possible_queries()
    y = pool.query(x_pool[0], noisy=False)
    assert pool.possible_queries().shape[0] == 300
    pool.set_replacement(False)
    x_pool = pool.possible_queries()
    y = pool.query(x_pool[0], noisy=False)
    assert pool.possible_queries().shape[0] == 299


############
### MultitaskPoolFromPools
############
def test_multitask_pool():
    P = 4
    pool_list = [PoolFromOracle(Sinus(0.01)) for _ in range(P)]
    for p in pool_list:
        p.discretize_random(300)
    pool = MultitaskPoolFromPools(pool_list)
    for i in range(P):
        pool.set_task_mode(i)
        pool.set_replacement(True)
        x_pool = pool.possible_queries()
        y = pool.query(x_pool[0], noisy=False)
        assert pool.possible_queries().shape[0] == 300
        pool.set_replacement(False)
        x_pool = pool.possible_queries()
        y = pool.query(x_pool[0], noisy=False)
        assert pool.possible_queries().shape[0] == 299

if __name__ == "__main__":
    test_pool_from_oracle_basic()
    test_pool_from_oracle_get_unconstrained_data()
    test_pool_from_oracle_get_constrained_data()
    test_pool_with_safety_from_oracle_basic()
    test_pool_with_safety_from_oracle_get_unconstrained_data()
    test_pool_with_safety_from_oracle_get_constrained_data()
    test_pool_from_data_basic()
    test_pool_from_data_get_unconstrained_data()
    test_pool_from_data_get_constrained_data()
    test_pool_with_safety_from_data_basic()
    test_pool_with_safety_from_data_get_unconstrained_data()
    test_pool_with_safety_from_data_get_constrained_data()
    test_pool_multioutput_from_data_basic()