import numpy as np
import pytest
from functools import partial
from alef.configs.kernels.rbf_configs import RBFWithPriorConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
from alef.oracles import (
    StandardConstrainedOracle,
    GPOracle1D,
    GPOracle2D,
    Ackley3D, Ackley4D, Ackley,
    BraninHoo,
    Eggholder,
    Exponential2DExtended,
    Exponential2D,
    Griewangk,
    Hartmann3,
    Hartmann6,
    LSQMain, LSQConstraint1, LSQConstraint2, LSQ,
    Rosenbrock3D, Rosenbrock4D, Rosenbrock,
    SimionescuMain, SimionescuConstraint, Simionescu,
    Sinus,
    TownsendMain, TownsendConstraint, Townsend,
    CartPoleConstrained,
    OracleNormalizer
)


@pytest.mark.parametrize("oracle_class, kernel_config", [
    (GPOracle1D, RBFWithPriorConfig(input_dimension=1)),
    (GPOracle1D, RBFWithPriorPytorchConfig(input_dimension=1)),
    (GPOracle2D, RBFWithPriorConfig(input_dimension=2)),
    (GPOracle2D, RBFWithPriorPytorchConfig(input_dimension=2))
])
def test_gp_oracles(oracle_class, kernel_config):
    oracle = oracle_class(kernel_config=kernel_config, observation_noise=0.1)
    #oracle.draw_from_hyperparameter_prior()
    # the backend of oracle.draw_from_hyperparameter_prior is from gp_samplers, which are tested
    # we switch this draw off for numerical stability
    oracle.initialize(0, 1, 20)

    a, b = oracle.get_box_bounds()
    assert a==0
    assert b==1
    D = oracle.get_dimension()
    assert D == kernel_config.input_dimension
    X, Y = oracle.get_random_data(1, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 1
    assert X.shape[1] == D
    assert Y.shape[0] == 1
    assert np.all(X >= a)
    assert np.all(X <= b)

    length = 1.0
    a_set = 0.1
    b_set = 0.6001
    X, Y = oracle.get_random_data_in_box(10, a_set, length*0.4, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 10
    assert Y.shape[0] == 10
    assert np.all(X >= a_set)
    assert np.all(X <= b_set)


@pytest.mark.parametrize("oracle_class", [
    Ackley3D, Ackley4D, partial(Ackley, dimension=2),
    BraninHoo,
    Eggholder,
    Exponential2DExtended,
    Exponential2D,
    partial(Griewangk, dimension=2), partial(Griewangk, dimension=3),
    Hartmann3, Hartmann6,
    LSQMain, LSQConstraint1, LSQConstraint2,
    Rosenbrock3D, Rosenbrock4D, partial(Rosenbrock, dimension=2),
    SimionescuMain, SimionescuConstraint,
    Sinus,
    TownsendMain, TownsendConstraint,
])
def test_standard_oracles(oracle_class):
    oracle = oracle_class(observation_noise=0.1)

    a, b = oracle.get_box_bounds()
    D = oracle.get_dimension()
    X, Y = oracle.get_random_data(1, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 1
    assert X.shape[1] == D
    assert Y.shape[0] == 1
    assert np.all(X >= a)
    assert np.all(X <= b)

    length = b - a
    a_set = a + length * 0.1
    b_set = a + length * 0.5001
    X, Y = oracle.get_random_data_in_box(10, a_set, length*0.4, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 10
    assert Y.shape[0] == 10
    assert np.all(X >= a_set)
    assert np.all(X <= b_set)
    X, F = oracle.get_random_data(10, noisy=False)
    F_ref = oracle.batch_query(X, noisy=False)
    assert F.shape[-2] == F_ref.shape[-1], X.shape
    assert F_ref.shape[-1] == 10
    assert np.allclose(F, F_ref.reshape(10, 1))


def test_oracle_normalizer():
    oracle = OracleNormalizer(
        BraninHoo(observation_noise=0.1)
    )
    oracle.set_normalization_manually(2.0, 3.0)
    mu, std = oracle.get_normalization()
    assert mu ==2
    assert std ==3
    oracle.set_normalization_by_sampling()

    a, b = oracle.get_box_bounds()
    D = oracle.get_dimension()
    X, Y = oracle.get_random_data(1, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 1
    assert X.shape[1] == D
    assert Y.shape[0] == 1
    assert np.all(X >= a)
    assert np.all(X <= b)

    length = b - a
    a_set = a + length * 0.1
    b_set = a + length * 0.5001
    X, Y = oracle.get_random_data_in_box(10, a_set, length*0.4, noisy=True)
    assert X.ndim == 2
    assert Y.ndim == 2
    assert X.shape[0] == 10
    assert Y.shape[0] == 10
    assert np.all(X >= a_set)
    assert np.all(X <= b_set)


@pytest.mark.parametrize("oracle", [
    LSQ(observation_noise=0.1),
    Simionescu(observation_noise=0.1),
    Townsend(observation_noise=0.1),
])
def test_standard_constrained_oracles(oracle):

    assert np.allclose(oracle.get_box_bounds(), oracle.oracle.get_box_bounds())
    a, b = oracle.get_box_bounds()
    assert oracle.get_dimension() == oracle.oracle.get_dimension()
    D = oracle.get_dimension()
    X, Y, Z = oracle.get_random_data(50, noisy=False)
    assert X.shape == (50, D)
    assert Y.shape == (50, 1)
    assert Z.shape == (50, len(oracle.constraint_oracle))
    assert np.all(X >= a)
    assert np.all(X <= b)
    assert np.allclose(np.squeeze(Y), oracle.oracle.batch_query(X, noisy=False))
    for i, c in enumerate(oracle.constraint_oracle):
        assert np.allclose(Z[:, i], c.batch_query(X, noisy=False))

    length = b - a
    a_set = a + length * 0.1
    b_set = a + length * 0.5001
    X, Y, Z = oracle.get_random_data_in_box(10, a_set, length*0.4, noisy=False)
    assert X.shape == (10, D)
    assert Y.shape == (10, 1)
    assert Z.shape == (10, len(oracle.constraint_oracle))
    assert np.all(X >= a_set)
    assert np.all(X <= b_set)
    assert np.allclose(np.squeeze(Y), oracle.oracle.batch_query(X, noisy=False))
    for i, c in enumerate(oracle.constraint_oracle):
        assert np.allclose(Z[:, i], c.batch_query(X, noisy=False))

def test_cart_pole():
    oracle = CartPoleConstrained(0.01)
    a, b = oracle.get_box_bounds()
    D = oracle.get_dimension()
    X, Y, Z = oracle.get_random_data(50, noisy=False)
    assert X.shape == (50, D)
    assert Y.shape == (50, 1)
    assert Z.shape == (50, 1)
    assert np.all(X >= a)
    assert np.all(X <= b)

    length = b - a
    a_set = a + length * 0.1
    b_set = a + length * 0.5001
    X, Y, Z = oracle.get_random_data_in_box(10, a_set, length*0.4, noisy=False)
    assert X.shape == (10, D)
    assert Y.shape == (10, 1)
    assert Z.shape == (10, len(oracle.constraint_oracle))
    assert np.all(X >= a_set)
    assert np.all(X <= b_set)

if __name__=="__main__":
    test_standard_oracles(BraninHoo)
