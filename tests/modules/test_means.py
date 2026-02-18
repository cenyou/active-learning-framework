import logging
import numpy as np
import pytest
import torch
import gpytorch
import gpflow
from alef.means.mean_factory import MeanFactory
from alef.configs.means import (
    BaseMeanConfig,
    BasicZeroMeanConfig,
    BasicQuadraticMeanConfig,
    BasicPeriodicMeanConfig,
    BasicSechMeanConfig,
)
from alef.means.quadratic_mean import QuadraticMean
from alef.means.periodic_mean import PeriodicMean
from alef.means.sech_mean import SechMean
from alef.means.pytorch_means.pytorch_mean_factory import PytorchMeanFactory
from alef.configs.means.pytorch_means import (
    BaseMeanPytorchConfig,
    BasicZeroMeanPytorchConfig,
    BasicLinearMeanPytorchConfig,
    BasicQuadraticMeanPytorchConfig,
    BasicPeriodicMeanPytorchConfig,
    BasicSechMeanPytorchConfig,
)
from alef.means.pytorch_means.pytorch_linear_mean import LinearPytorchMean
from alef.means.pytorch_means.pytorch_quadratic_mean import QuadraticPytorchMean
from alef.means.pytorch_means.pytorch_periodic_mean import PeriodicPytorchMean
from alef.means.pytorch_means.pytorch_sech_mean import SechPytorchMean

def sech(x):
    return 2*np.exp(x) / (np.exp(2*x) + 1)


D = 3
def test_zero_mean():
    x = np.random.rand(10, 20, D)
    x_torch = torch.from_numpy(x).to(dtype=torch.get_default_dtype())
    ###
    mean_config = BasicZeroMeanConfig()
    m = MeanFactory.build(mean_config)
    assert isinstance(m, gpflow.mean_functions.Zero)
    assert np.allclose(m(x)[..., 0], np.zeros([10, 20]), rtol=1e-4, atol=1e-5)
    ###
    mean_config = BasicZeroMeanPytorchConfig(batch_shape=[])
    m = PytorchMeanFactory.build(mean_config)
    assert isinstance(m, gpytorch.means.ZeroMean)
    assert np.allclose(m(x_torch).detach().numpy(), np.zeros([10, 20]), rtol=1e-4, atol=1e-5)


def test_linear_mean():
    B, N = 10, 20
    x = np.random.rand(B, N, D)
    x_torch = torch.from_numpy(x).to(dtype=torch.get_default_dtype())
    ###
    mean_config = BasicLinearMeanPytorchConfig(
        input_dimension=D,
        scale=np.random.uniform(2.0, 3.0),
        bias=np.random.uniform(-1.5, -0.8),
        weights=np.linspace(1, 2, D).tolist()
    )
    m = PytorchMeanFactory.build(mean_config)
    assert isinstance(m, LinearPytorchMean)
    c = mean_config.scale
    b = mean_config.bias
    w = mean_config.weights
    ref = c*b + c*np.inner(x, w)
    assert np.allclose(m(x_torch).detach().numpy(), ref, rtol=1e-4, atol=1e-5)
    #
    mask = torch.ones([B, D], dtype=int)
    mask[..., -1] = torch.randint(0, 2, size=mask.shape[:-1], dtype=mask.dtype)
    mean = m(x_torch, mask=mask).detach().numpy() # [B, N]
    l = mask.sum(-1) # [B]
    for i in range(B):
        ref = c*b + c*np.inner(x[i, :, :l[i]], w[:l[i]])
        assert np.allclose(mean[i], ref, rtol=1e-4, atol=1e-5)


def test_quadratic_mean():
    B, N = 10, 20
    x = np.random.rand(B, N, D)
    x_torch = torch.from_numpy(x).to(dtype=torch.get_default_dtype())
    ###
    mean_config = BasicQuadraticMeanConfig(
        input_dimension=D,
        scale=np.random.uniform(2.0, 3.0),
        bias=np.random.uniform(-1.5, -0.8),
        center=np.random.uniform(0.4, 0.6),
        weights=np.linspace(1, 2, D).tolist()
    )
    m = MeanFactory.build(mean_config)
    assert isinstance(m, QuadraticMean)
    c = mean_config.scale
    b = mean_config.bias
    w = mean_config.weights
    center = mean_config.center
    ref = c*b + c*np.inner((x - center)**2, w)/D
    assert np.allclose(m(x)[..., 0], ref, rtol=1e-4, atol=1e-5)
    ###
    mean_config = BasicQuadraticMeanPytorchConfig(
        input_dimension=D,
        batch_shape=[],
        scale=np.random.uniform(2.0, 3.0),
        bias=np.random.uniform(-1.5, -0.8),
        center=np.random.uniform(0.4, 0.6),
        weights=np.linspace(1, 2, D).tolist()
    )
    m = PytorchMeanFactory.build(mean_config)
    assert isinstance(m, QuadraticPytorchMean)
    c = mean_config.scale
    b = mean_config.bias
    w = mean_config.weights
    center = mean_config.center
    ref = c*b + c*np.inner((x - center)**2, w)/D
    assert np.allclose(m(x_torch).detach().numpy(), ref, rtol=1e-4, atol=1e-5)
    #
    mask = torch.ones([B, D], dtype=int)
    mask[..., -1] = torch.randint(0, 2, size=mask.shape[:-1], dtype=mask.dtype)
    mean = m(x_torch, mask=mask).detach().numpy() # [B, N]
    l = mask.sum(-1) # [B]
    for i in range(B):
        ref = c*b + c*np.inner((x[i, :, :l[i]] - center)**2, w[:l[i]])/l[i]
        assert np.allclose(mean[i], ref, rtol=1e-4, atol=1e-5)


def test_periodic_mean():
    B, N = 10, 20
    x = np.random.rand(B, N, D)
    x_torch = torch.from_numpy(x).to(dtype=torch.get_default_dtype())
    ###
    mean_config = BasicPeriodicMeanConfig(
        input_dimension=D,
        scale=np.random.uniform(2.0, 3.0),
        bias=np.random.uniform(-1.5, -0.8),
        center=np.random.uniform(0.4, 0.6),
        weights=np.linspace(1, 2, D).tolist()
    )
    m = MeanFactory.build(mean_config)
    assert isinstance(m, PeriodicMean)
    c = mean_config.scale
    b = mean_config.bias
    w = mean_config.weights
    center = mean_config.center
    ref = c*b + c*np.cos( np.inner((x - center)**2, w)/D )
    assert np.allclose(m(x)[..., 0], ref, rtol=1e-4, atol=1e-5)
    ###
    mean_config = BasicPeriodicMeanPytorchConfig(
        input_dimension=D,
        batch_shape=[],
        scale=np.random.uniform(2.0, 3.0),
        bias=np.random.uniform(-1.5, -0.8),
        center=np.random.uniform(0.4, 0.6),
        weights=np.linspace(1, 2, D).tolist()
    )
    m = PytorchMeanFactory.build(mean_config)
    assert isinstance(m, PeriodicPytorchMean)
    c = mean_config.scale
    b = mean_config.bias
    w = mean_config.weights
    center = mean_config.center
    ref = c*b + c*np.cos( np.inner((x - center)**2, w)/D )
    assert np.allclose(m(x_torch).detach().numpy(), ref, rtol=1e-4, atol=1e-5)
    #
    mask = torch.ones([B, D], dtype=int)
    mask[..., -1] = torch.randint(0, 2, size=mask.shape[:-1], dtype=mask.dtype)
    mean = m(x_torch, mask=mask).detach().numpy() # [B, N]
    l = mask.sum(-1) # [B]
    for i in range(mean.shape[0]):
        ref = c*b + c*np.cos( np.inner((x[i, :, :l[i]] - center)**2, w[:l[i]]) / l[i] )
        assert np.allclose(mean[i], ref, rtol=1e-4, atol=1e-5)


def test_sech_mean():
    B, N = 10, 20
    x = np.random.rand(B, N, D)
    x_torch = torch.from_numpy(x).to(dtype=torch.get_default_dtype())
    ###
    mean_config = BasicSechMeanConfig(
        input_dimension=D,
        scale=np.random.uniform(2.0, 3.0),
        bias=np.random.uniform(-1.5, -0.8),
        center=np.random.uniform(0.4, 0.6),
        weights=np.linspace(1, 2, D).tolist()
    )
    m = MeanFactory.build(mean_config)
    assert isinstance(m, SechMean)
    c = mean_config.scale
    b = mean_config.bias
    w = mean_config.weights
    center = mean_config.center
    ref = c*b + c*sech( np.inner((x - center)**2, w)/D )
    assert np.allclose(m(x)[..., 0], ref, rtol=1e-4, atol=1e-5)
    ###
    mean_config = BasicSechMeanPytorchConfig(
        input_dimension=D,
        batch_shape=[],
        scale=np.random.uniform(2.0, 3.0),
        bias=np.random.uniform(-1.5, -0.8),
        center=np.random.uniform(0.4, 0.6),
        weights=np.linspace(1, 2, D).tolist()
    )
    m = PytorchMeanFactory.build(mean_config)
    assert isinstance(m, SechPytorchMean)
    c = mean_config.scale
    b = mean_config.bias
    w = mean_config.weights
    center = mean_config.center
    ref = c*b + c*sech( np.inner((x - center)**2, w)/D )
    assert np.allclose(m(x_torch).detach().numpy(), ref, rtol=1e-4, atol=1e-5)
    #
    mask = torch.ones([B, D], dtype=int)
    mask[..., -1] = torch.randint(0, 2, size=mask.shape[:-1], dtype=mask.dtype)
    mean = m(x_torch, mask=mask).detach().numpy() # [B, N]
    l = mask.sum(-1) # [B]
    for i in range(mean.shape[0]):
        ref = c*b + c*sech( np.inner((x[i, :, :l[i]] - center)**2, w[:l[i]]) / l[i] )
        assert np.allclose(mean[i], ref, rtol=1e-4, atol=1e-5)

