import numpy as np
import torch
import pytest
from pyro.distributions import MultivariateNormal, Categorical
from alef.utils.pyro_distributions import CategoricalWithValues, MultivariateNormalSVD, GPMatheron, GPMatheronSVD
N_batch = 20
expand_batch = [30, 20]
D_prior = 10
D_post = 5
D = D_prior + D_post

L = np.random.uniform(0, 1, [D, D])
cov = L @ L.T + np.eye(D)
cov = torch.from_numpy(cov).to(torch.get_default_dtype())
loc = np.random.standard_normal([N_batch, D])
loc = torch.from_numpy(loc).to(cov.dtype)

test_values = torch.from_numpy(
    np.random.standard_normal([N_batch, D_post])
).to(torch.get_default_dtype())
test_values_after_expand = torch.from_numpy(
    np.random.standard_normal(expand_batch + [D_post])
).to(torch.get_default_dtype())


tmp = MultivariateNormal(loc, covariance_matrix=cov)
Y = tmp.sample()
L_prior = torch.linalg.cholesky(cov[..., :D_prior, :D_prior])
u, s, vh = torch.linalg.svd(cov[..., :D_prior, :D_prior])
K_prior_new = cov[..., :D_prior, D_prior:]
K_new = cov[..., D_prior:, D_prior:]


@pytest.mark.parametrize("dist1,dist2",[
        (
            MultivariateNormal(loc[..., D_prior:], covariance_matrix=K_new),
            MultivariateNormalSVD(loc[..., D_prior:], covariance_matrix=K_new)
        ),
        #(
        #    GPMatheron(loc[..., D_prior:], loc[..., :D_prior], Y[..., :D_prior], L_prior, K_prior_new, K_new),
        #    GPMatheronSVD(loc[..., D_prior:], loc[..., :D_prior], Y[..., :D_prior], u, torch.sqrt(s), K_prior_new, K_new)
        #),
    ])
def test_equivalent_distribution(dist1, dist2):
    def run_assert(dist1, dist2, eval_tensor):
        assert dist1.sample().size() == dist2.sample().size()
        assert dist1.sample([2, 3,]).size() == dist2.sample([2, 3,]).size()
        assert torch.allclose(dist1.mean, dist2.mean, rtol=1e-4)
        assert torch.allclose(dist1.covariance_matrix, dist2.covariance_matrix, rtol=1e-4)
        assert torch.allclose(dist1.variance, dist2.variance, rtol=1e-4)
        assert torch.allclose(dist1.entropy(), dist2.entropy(), rtol=1e-4)
        assert torch.allclose(dist1.log_prob(eval_tensor), dist2.log_prob(eval_tensor), rtol=1e-4)
    run_assert(dist1, dist2, test_values)
    run_assert(dist1.expand(expand_batch), dist2.expand(expand_batch), test_values_after_expand)


def test_categorical():
    probs = torch.rand([D])
    values = torch.rand_like(probs)
    ref = Categorical(probs=probs)
    dist = CategoricalWithValues(values, probs=probs)

    test_idx = torch.randint(0, D, size=[100])
    test_val = values[test_idx]
    assert torch.allclose(ref.log_prob(test_idx), dist.log_prob(test_val))
    torch.manual_seed(0)
    sample_idx = ref.sample([100])
    torch.manual_seed(0)
    sample_val = dist.sample([100])
    assert torch.allclose(values[sample_idx], sample_val)
    
    probs = torch.rand(expand_batch + [D])
    values = torch.rand_like(probs)
    ref = Categorical(probs=probs)
    dist = CategoricalWithValues(values, probs=probs)

    test_idx = torch.randint(0, D, size=[100]+expand_batch)
    test_val = torch.gather(
        values.expand([100]+expand_batch+[D]),
        dim=-1,
        index=test_idx.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.allclose(ref.log_prob(test_idx), dist.log_prob(test_val))
    torch.manual_seed(0)
    sample_idx = ref.sample([100])
    torch.manual_seed(0)
    sample_val = dist.sample([100])
    assert torch.allclose(
        torch.gather(
            values.expand([100]+expand_batch+[D]),
            dim=-1,
            index=sample_idx.unsqueeze(-1)
        ).squeeze(-1),
        sample_val
    )
    

