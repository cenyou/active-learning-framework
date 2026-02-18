import numpy as np
import torch
import pytest
import gpytorch
from pyro import poutine, condition
from pyro.distributions import Categorical, Normal, MultivariateNormal
from alef.models.gp_model_pytorch import ExactGPModel
from alef.configs.initial_data_conditions import INITIAL_OUTPUT_HIGHER_BY, INITIAL_OUTPUT_LOWER_BY
from alef.configs.base_parameters import NUMERICAL_POSITIVE_LOWER_BOUND
from alef.configs.active_learners.amortized_policies.policy_configs import (
    ContinuousGPPolicyConfig,
    SafetyAwareContinuousGPPolicyConfig,
    ContinuousGPFlexDimPolicyConfig,
    SafetyAwareContinuousGPFlexDimPolicyConfig,
)

# need GP model
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
from alef.configs.means.pytorch_means import BasicZeroMeanPytorchConfig, BasicPeriodicMeanPytorchConfig
from alef.configs.kernels.pytorch_kernels.elementary_kernels_pytorch_configs import RBFWithPriorPytorchConfig
from alef.enums.active_learner_amortized_policy_enums import DomainWarpperType, SafetyProbability, SafetyProbabilityWrapper

from alef.active_learners.amortized_policies.amortized_policy_factory import AmortizedPolicyFactory
from alef.active_learners.amortized_policies.simulated_processes import (
    PytestSequentialGaussianProcessContinuousDomain,
    PytestSequentialSafeGaussianProcessContinuousDomain,
)
from alef.active_learners.amortized_policies.losses import (
    PriorContrastiveEstimation,
    GPEntropy1Loss,
    GPEntropy2Loss,
    GPMutualInformation1Loss,
    GPMutualInformation2Loss,
    GPSafetyEntropyWrapLoss,
    GPSafetyMIWrapLoss,
)
from alef.active_learners.amortized_policies.utils.gp_computers import GaussianProcessComputer
from alef.active_learners.amortized_policies.utils.safety_probabilities import SigmoidSafetyProbability
from alef.active_learners.amortized_policies.global_parameters import MAX_DIMENSION

from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)

n_init_min = 5
n_init_max = 10
T = 2 # please keep T > 1
D = MAX_DIMENSION
device = 'cpu'
assert device=='cpu', 'do not support gpu test at the moment, pyro site control will be wrong (only wrong in test)'
## test loss functions
design_net = AmortizedPolicyFactory.build(
    ContinuousGPFlexDimPolicyConfig(
        encoding_dim = 8,
        num_self_attention_layer = 1,
        attend_sequence_first = True,
        hidden_dim_budget_encoder = [32],
        hidden_dim_emitter = [32],
        domain_warpper=DomainWarpperType.TANH,
        forward_with_budget=True,
        device=device
    )
)
# AmortizedPolicyFactory.build(
#     ContinuousGPPolicyConfig(
#         input_dim=D,
#         hidden_dim_encoder = [32],
#         encoding_dim = 8,
#         hidden_dim_emitter = [32],
#         num_self_attention_layer=1,
#         domain_warpper=DomainWarpperType.TANH,
#         forward_with_budget=True,
#         device=device
#     )
# )
process = PytestSequentialGaussianProcessContinuousDomain(
    design_net,
    kernel_config=RBFWithPriorPytorchConfig(
        input_dimension=D,
        base_lengthscale=0.2
    ),
    mean_config=BasicPeriodicMeanPytorchConfig(
        input_dimension=D,
        batch_shape=[],
        center=0.5,
        weights=8,
        scale=1.0,
        bias=0.0
    ),
    n_initial_min=n_init_min,
    n_initial_max=n_init_max,
    n_steps_min=None,
    n_steps_max=T,
    sample_gp_prior=True,
    random_subsequence=True,
    split_subsequence=True,
    device=device
)
## Safety aware
safe_design_net = AmortizedPolicyFactory.build(
    SafetyAwareContinuousGPFlexDimPolicyConfig(
        encoding_dim = 8,
        num_self_attention_layer = 1,
        attend_sequence_first = True,
        hidden_dim_budget_encoder = [32],
        hidden_dim_emitter = [32],
        domain_warpper=DomainWarpperType.TANH,
        forward_with_budget=True,
        device=device
    )
)
# AmortizedPolicyFactory.build(
#     SafetyAwareContinuousGPPolicyConfig(
#         input_dim=D,
#         hidden_dim_encoder = [32],
#         encoding_dim = 8,
#         hidden_dim_emitter = [32],
#         num_self_attention_layer=1,
#         domain_warpper=DomainWarpperType.TANH,
#         forward_with_budget=True,
#         device=device
#     )
# )
safe_process = PytestSequentialSafeGaussianProcessContinuousDomain(
    safe_design_net,
    kernel_config=RBFWithPriorPytorchConfig(
        input_dimension=D,
        base_lengthscale=0.2
    ),
    mean_config=BasicPeriodicMeanPytorchConfig(
        input_dimension=D,
        batch_shape=[],
        center=0.5,
        weights=8,
        scale=1.0,
        bias=0.0
    ),
    n_initial_min=n_init_min,
    n_initial_max=n_init_max,
    n_steps_min=None,
    n_steps_max=T,
    sample_gp_prior=True,
    random_subsequence=True,
    split_subsequence=True,
    device=device
)

Nk, Nf, B = 4, 5, 3
num_splits = 2 # would chunk Nk into 2 runs in sequence
X_init = torch.rand([1, Nk, Nf, n_init_max, D]).expand([B, Nk, Nf, n_init_max, D])
X_query = torch.rand([B, Nk, Nf, T + n_init_max - n_init_min, D])
Y_init = torch.rand([1, Nk, Nf, n_init_max]).expand([B, Nk, Nf, n_init_max])
Y_query = torch.rand([B, Nk, Nf, T + n_init_max - n_init_min])
Z_init = torch.rand([1, Nk, Nf, n_init_max]).expand([B, Nk, Nf, n_init_max]).abs() + INITIAL_OUTPUT_HIGHER_BY
Z_query = torch.rand([B, Nk, Nf, T + n_init_max - n_init_min]).abs()

n_grid = 2
X_grid = torch.zeros([B, Nk, Nf, 2, D])
X_grid[:, :, :, 1, :] = 1
Y_grid = torch.ones([B, Nk, Nf, 2])

_ = process(B, Nk, 1, sample_domain_grid_points=False)
primary_blm_ref = process.gp_dist.bayesian_linear_model

ref_mean_list, ref_kernel_list, ref_noise_var, \
ref_mean_list_safety, ref_kernel_list_safety, ref_noise_var_safety, \
mask_dim, \
n_init, n_query1, n_query2, \
_, _, _, _, _, _ = \
    safe_process( B, Nk, Nf, sample_domain_grid_points=False )

sample_reference = {
    'Trial_0_n_init': n_init[:, :2, :],
    'Trial_1_n_init': n_init[:, 2:, :],
    'Trial_0_t1': n_query1[:, :2, :],
    'Trial_1_t1': n_query1[:, 2:, :],
    'Trial_0_t2': n_query2[:, :2, :],
    'Trial_1_t2': n_query2[:, 2:, :],
    'Trial_0_dimension': mask_dim[0, :2].sum(-1) if not mask_dim is None else None,
    'Trial_1_dimension': mask_dim[0, 2:].sum(-1) if not mask_dim is None else None,
    **{f'Trial_0.y_sampler.gp_mean.{nk}.index': torch.zeros(1, dtype=int, device=device) for nk in range(2)},
    **{f'Trial_1.y_sampler.gp_mean.{nk}.index': torch.zeros(1, dtype=int, device=device) for nk in range(2)},
    **{f'Trial_0.y_sampler.gp_mean.{nk}.center_prior': ref_mean_list[nk].center for nk in range(2)},
    **{f'Trial_1.y_sampler.gp_mean.{nk}.center_prior': ref_mean_list[2+nk].center for nk in range(2)},
    **{f'Trial_0.y_sampler.gp_mean.{nk}.weights_prior': ref_mean_list[nk].weights for nk in range(2)},
    **{f'Trial_1.y_sampler.gp_mean.{nk}.weights_prior': ref_mean_list[2+nk].weights for nk in range(2)},
    'Trial_0.y_sampler.kernels.0.outputscale_prior': safe_process.gp_dist.kernel.kernel.outputscale[:2],
    'Trial_1.y_sampler.kernels.0.outputscale_prior': safe_process.gp_dist.kernel.kernel.outputscale[2:],
    'Trial_0.y_sampler.kernels.0.base_kernel.lengthscale_prior': safe_process.gp_dist.kernel.kernel.base_kernel.lengthscale[:2,...],
    'Trial_1.y_sampler.kernels.0.base_kernel.lengthscale_prior': safe_process.gp_dist.kernel.kernel.base_kernel.lengthscale[2:,...],
    **{f'Trial_0.z_sampler.gp_mean.{nk}.index': torch.zeros(1, dtype=int, device=device) for nk in range(2)},
    **{f'Trial_1.z_sampler.gp_mean.{nk}.index': torch.zeros(1, dtype=int, device=device) for nk in range(2)},
    **{f'Trial_0.z_sampler.gp_mean.{nk}.center_prior': ref_mean_list_safety[nk].center for nk in range(2)},
    **{f'Trial_1.z_sampler.gp_mean.{nk}.center_prior': ref_mean_list_safety[2+nk].center for nk in range(2)},
    **{f'Trial_0.z_sampler.gp_mean.{nk}.weights_prior': ref_mean_list[nk].weights for nk in range(2)},
    **{f'Trial_1.z_sampler.gp_mean.{nk}.weights_prior': ref_mean_list[2+nk].weights for nk in range(2)},
    'Trial_0.z_sampler.kernels.0.outputscale_prior': safe_process.safety_gp_dist.kernel.kernel.outputscale[:2],
    'Trial_1.z_sampler.kernels.0.outputscale_prior': safe_process.safety_gp_dist.kernel.kernel.outputscale[2:],
    'Trial_0.z_sampler.kernels.0.base_kernel.lengthscale_prior': safe_process.safety_gp_dist.kernel.kernel.base_kernel.lengthscale[:2,...],
    'Trial_1.z_sampler.kernels.0.base_kernel.lengthscale_prior': safe_process.safety_gp_dist.kernel.kernel.base_kernel.lengthscale[2:,...],
    'Trial_0_x_init': X_init[:1, :2, :],
    'Trial_0_y_init': Y_init[:1, :2, :],
    'Trial_0_z_init': Z_init[:1, :2, :],
    'Trial_1_x_init': X_init[:1, 2:, :],
    'Trial_1_y_init': Y_init[:1, 2:, :],
    'Trial_1_z_init': Z_init[:1, 2:, :],
    **{f'Trial_0_x{i+1}': X_query[:, :2, :, i, None, :] for i in range(T)},
    **{f'Trial_0_y{i+1}': Y_query[:, :2, :, i, None] for i in range(T)},
    **{f'Trial_0_z{i+1}': Z_query[:, :2, :, i, None] for i in range(T)},
    **{f'Trial_1_x{i+1}': X_query[:, 2:, :, i, None, :] for i in range(T)},
    **{f'Trial_1_y{i+1}': Y_query[:, 2:, :, i, None] for i in range(T)},
    **{f'Trial_1_z{i+1}': Z_query[:, 2:, :, i, None] for i in range(T)},
}

test_sample_reference = {
    **{f'Trial_0.y_sampler.gp_mean.{nk}.index': torch.zeros(1, dtype=int, device=device) for nk in range(Nk)},
    **{f'Trial_0.y_sampler.gp_mean.{nk}.center_prior': ref_mean_list[nk].center for nk in range(Nk)},
    **{f'Trial_0.y_sampler.gp_mean.{nk}.weights_prior': ref_mean_list[nk].weights for nk in range(Nk)},
    'Trial_0.y_sampler.kernels.0.outputscale_prior': safe_process.gp_dist.kernel.kernel.outputscale,
    'Trial_0.y_sampler.kernels.0.base_kernel.lengthscale_prior': safe_process.gp_dist.kernel.kernel.base_kernel.lengthscale,
    'Trial_0_x_init': X_init[:1, :, :, :n_init_min, :],
    'Trial_0_y_init': Y_init[:1, :, :, :n_init_min],
    'Trial_0_z_init': Z_init[:1, :, :, :n_init_min],
    **{f'Trial_0_x{i+1}': X_query[:, :, :, i, None, :] for i in range(T + n_init_max - n_init_min)},
    **{f'Trial_0_y{i+1}': Y_query[:, :, :, i, None] for i in range(T + n_init_max - n_init_min)},
    **{f'Trial_0_z{i+1}': Z_query[:, :, :, i, None] for i in range(T + n_init_max - n_init_min)},
}

# GaussianProcessComputer is heavily used in our losses, make sure everything is correct
def test_gp_computer_cholesky():
    gp_helper = GaussianProcessComputer()
    kernel = ref_kernel_list[0]
    noise_var = 0.01
    xx = torch.rand([10, D])
    xnew = torch.rand([5, D])
    # test cholesky of prior
    K = kernel(xx, xx).to_dense() + noise_var * torch.eye(10)
    L = gp_helper.compute_cholesky(K)
    assert torch.allclose( L, torch.linalg.cholesky(K) )
    # test covariance computation
    K_cross = kernel(xx, xnew).to_dense()
    Knew = kernel(xnew, xnew).to_dense() + noise_var * torch.eye(5)
    Kall = kernel(
        torch.concat([xx, xnew], dim=-2),
        torch.concat([xx, xnew], dim=-2)
    ).to_dense() + noise_var * torch.eye(15)
    assert torch.allclose(
        gp_helper.compute_cholesky_update(
            L, K_cross, Knew),
        torch.linalg.cholesky(Kall),
        rtol=1e-4, atol=1e-5
    ), gp_helper.compute_cholesky_update( L, K_cross, Knew ) - torch.linalg.cholesky(Kall)
    assert torch.allclose(
        gp_helper.compute_cholesky_inv_B(
            L, K_cross),
        torch.linalg.solve_triangular(L, K_cross, upper=False),
        rtol=1e-4, atol=1e-5
    ), gp_helper.compute_cholesky_inv_B( L, K_cross ) - torch.linalg.solve_triangular(L, K_cross, upper=False)

# GaussianProcessComputer is heavily used in our losses, make sure everything is correct
def test_gp_computer_posterior():
    gp_helper = GaussianProcessComputer()
    kernel = ref_kernel_list[0]
    noise_var = 0.01
    xx = torch.rand([10, D])
    xnew = torch.rand([5, D])
    mu = torch.arange(10)
    yy = mu + torch.rand([10])
    munew = torch.arange(10, 15)
    ynew = munew + torch.rand([5])
    #
    K = kernel(xx, xx).to_dense() + noise_var * torch.eye(10)
    L = torch.linalg.cholesky(K)
    # test covariance computation
    K_cross = kernel(xx, xnew).to_dense()
    Knew = kernel(xnew, xnew).to_dense() + noise_var * torch.eye(5)
    # test GP posterior
    mu_post, cov_post = gp_helper.compute_gaussian_process_posterior(K, K_cross, Knew, yy, mu, munew, return_mu=True, return_cov=True)
    B = torch.linalg.solve_triangular(L, K_cross, upper=False)
    mu_post_ref = munew + torch.matmul(
        B.T,
        torch.linalg.solve_triangular(L, (yy-mu).unsqueeze(-1), upper=False)
    ).squeeze(-1)
    cov_post_ref = Knew - torch.matmul(B.T, B)
    assert torch.allclose( mu_post, mu_post_ref, rtol=1e-4, atol=1e-5 ), mu_post - mu_post_ref
    assert torch.allclose( cov_post, cov_post_ref, rtol=1e-4, atol=1e-5 ), cov_post - cov_post_ref
    # test entropy
    ent = gp_helper.compute_gaussian_entropy(Knew)
    ent_ref = MultivariateNormal(munew, Knew).entropy()
    assert torch.allclose( ent, ent_ref, rtol=1e-4, atol=1e-5 ), ent - ent_ref
    # test log likelihood
    log_py = gp_helper.compute_gaussian_log_likelihood(ynew, munew, Knew)
    log_py_ref = MultivariateNormal(munew, Knew).log_prob(ynew)
    assert torch.allclose( log_py, log_py_ref, rtol=1e-4, atol=1e-5 ), log_py - log_py_ref

# GaussianProcessComputer is heavily used in our losses, make sure everything is correct
def test_gp_computer_subsequence():
    gp_helper = GaussianProcessComputer()
    kernel = ref_kernel_list[0]
    noise_var = 0.01
    xnew = torch.rand([5, D])
    munew = torch.arange(10, 15)
    ynew = munew + torch.rand([5])
    Knew = kernel(xnew, xnew).to_dense() + noise_var * torch.eye(5)
    # test entropy and log likelihood of subsequences
    mask_start = Categorical(probs=torch.ones([xnew.shape[-2] - 1])).sample(xnew.shape[:-2] + (1, ))
    probs = torch.where(
        torch.arange(xnew.shape[-2]+1) > mask_start,
        torch.ones([xnew.shape[-2]+1]),
        torch.zeros([xnew.shape[-2]+1])
    )
    mask_end = Categorical(probs=probs).sample().unsqueeze(-1)
    mask_idx = torch.cat([mask_start, mask_end], dim=-1)
    # entropy
    ent = gp_helper.compute_gaussian_entropy(Knew, mask_idx=mask_idx)
    ent_ref = MultivariateNormal(
        munew[:mask_idx[1]],
        Knew[:mask_idx[1], :mask_idx[1]]
    ).entropy()
    ent_ref -= 0 if mask_start==0 else MultivariateNormal(
        munew[:mask_idx[0]],
        Knew[:mask_idx[0], :mask_idx[0]]
    ).entropy()
    assert torch.allclose( ent, ent_ref, rtol=1e-4, atol=1e-5 ), ent - ent_ref
    # log likelihood
    log_py = gp_helper.compute_gaussian_log_likelihood(ynew, munew, Knew, mask_idx=mask_idx)
    log_py_ref = MultivariateNormal(
        munew[:mask_idx[1]],
        Knew[:mask_idx[1], :mask_idx[1]]
    ).log_prob(ynew[:mask_idx[1]])
    log_py_ref -= 0 if mask_start==0 else MultivariateNormal(
        munew[:mask_idx[0]],
        Knew[:mask_idx[0], :mask_idx[0]]
    ).log_prob(ynew[:mask_idx[0]])
    assert torch.allclose( log_py, log_py_ref, rtol=1e-4, atol=1e-5 ), log_py - log_py_ref


# GaussianProcessComputer is heavily used in our losses, make sure everything is correct
def test_gp_computer_random_prior():
    gp_helper = GaussianProcessComputer()
    kernel = ref_kernel_list[0]
    noise_var = 0.01
    n_batch, n_prior, n_post = 50, 10, 5
    xx = torch.rand([n_batch, n_prior, D])
    xnew = torch.rand([n_batch, n_post, D])
    mu = torch.arange(n_prior).unsqueeze(0).expand([n_batch, n_prior])
    yy = mu + torch.rand([n_batch, n_prior])
    munew = torch.arange(n_prior, n_prior + n_post).unsqueeze(0).expand([n_batch, n_post])
    ynew = munew + torch.rand([n_batch, n_post])
    used_priors = torch.randint(1, n_prior + 1, size=[n_batch])
    #
    K = kernel(xx, xx).to_dense() + noise_var * torch.eye(n_prior)
    K_cross = kernel(xx, xnew).to_dense()
    Knew = kernel(xnew, xnew).to_dense() + noise_var * torch.eye(n_post)
    # test GP posterior
    mu_post, cov_post = gp_helper.compute_gaussian_process_posterior(K, K_cross, Knew, yy, mu, munew, prior_mask_idx=used_priors, return_mu=True, return_cov=True)
    for b in range(n_batch):
        np = used_priors[b]
        L = torch.linalg.cholesky(K[b, :np, :np])
        B = torch.linalg.solve_triangular(L, K_cross[b, :np], upper=False)        
        mu_post_ref = munew[b] + torch.matmul(
            B.T,
            torch.linalg.solve_triangular(L, (yy[b, :np] - mu[b, :np]).unsqueeze(-1), upper=False)
        ).squeeze(-1)
        cov_post_ref = Knew[b] - torch.matmul(B.T, B)
        assert torch.allclose( mu_post[b], mu_post_ref, rtol=1e-4, atol=1e-5 ), mu_post[b] - mu_post_ref
        assert torch.allclose( cov_post[b], cov_post_ref, rtol=1e-4, atol=1e-5 ), cov_post[b] - cov_post_ref


###
###
###
def test_amortized_al_dad_loss():
    loss = PriorContrastiveEstimation(
        B, Nk, Nf,
        data_source=iter([{# primary rollout take this
            **{f'Default.y_sampler.gp_mean.{nk}.index': torch.zeros(1, dtype=int, device=device) for nk in range(Nk)},
            **{f'Default.y_sampler.gp_mean.{nk}.center_prior': ref_mean_list[nk].center for nk in range(Nk)},
            **{f'Default.y_sampler.gp_mean.{nk}.weights_prior': ref_mean_list[nk].weights for nk in range(Nk)},
            'Default.y_sampler.kernels.0.outputscale_prior': safe_process.gp_dist.kernel.kernel.outputscale,
            'Default.y_sampler.kernels.0.base_kernel.lengthscale_prior': safe_process.gp_dist.kernel.kernel.base_kernel.lengthscale,
            'Default.y_sampler.BayesianLinearModel.omega': primary_blm_ref.omega,
            'Default.y_sampler.BayesianLinearModel.weight': primary_blm_ref.weight,
            'Default.y_sampler.BayesianLinearModel.bias': primary_blm_ref.bias,
            'Default_x_init': X_init[:1, :, 0, None, :n_init_min, :],
            'Default_y_init': Y_init[:1, :, 0, None, :n_init_min],
            **{f'Default_x{i+1}': X_query[:, :, 0, None, i, None, :] for i in range(T)},
            **{f'Default_y{i+1}': Y_query[:, :, 0, None, i, None] for i in range(T)},
            'Default_n_init': n_init_min*torch.ones([B, Nk, 1], dtype=int),
            'Default_dimension': D*torch.ones([Nk, 1], dtype=mask_dim.dtype, device=device) if not mask_dim is None else None,
        },
        {# test rollout take this
            **{f'Default.y_sampler.gp_mean.{nk}.index': torch.zeros(1, dtype=int, device=device) for nk in range(Nk)},
            **{f'Default.y_sampler.gp_mean.{nk}.center_prior': ref_mean_list[nk].center for nk in range(Nk)},
            **{f'Default.y_sampler.gp_mean.{nk}.weights_prior': ref_mean_list[nk].weights for nk in range(Nk)},
            'Default.y_sampler.kernels.0.outputscale_prior': safe_process.gp_dist.kernel.kernel.outputscale,
            'Default.y_sampler.kernels.0.base_kernel.lengthscale_prior': safe_process.gp_dist.kernel.kernel.base_kernel.lengthscale,
            'Default_x_init': X_init[:1, :, :, :n_init_min, :],
            'Default_y_init': Y_init[:1, :, :, :n_init_min],
            **{f'Default_x{i+1}': X_query[:, :, :, i, None, :] for i in range(T)},
            **{f'Default_y{i+1}': Y_query[:, :, :, i, None] for i in range(T)},
            'Default_dimension': D*torch.ones([Nk, Nf], dtype=mask_dim.dtype, device=device) if not mask_dim is None else None,
        }])
    )
    #
    process.random_subsequence = False
    #
    loss_value = loss.differentiable_loss(process) # this method execute process twice
    # 1st run get primary p(Y | f, X) with shape [B, Nk, 1], f = primary_blm_ref
    # 2nd run get contrastive p(Y) with shape [B, Nk, Nf]
    X = torch.cat([X_init[:, :, 0, None, :n_init_min, :], X_query[:, :, 0, None, :T, :]], dim=-2) # [B, Nk, 1, n_init_min + T, D]
    Y = torch.cat([Y_init[:, :, 0, None, :n_init_min], Y_query[:, :, 0, None, :T]], dim=-1) # [B, Nk, 1, n_init_min + T]
    gp_mean = torch.concat( [m(X[:, i, None]) for i, m in enumerate(ref_mean_list)], dim=1 ) # [B, Nk, 1, n_init_min + T]
    f_primary = gp_mean + primary_blm_ref(X) # [B, Nk, 1, n_init + T]
    scale_primary = torch.sqrt( ref_noise_var ).reshape([1, Nk, 1, 1]).expand([B, Nk, 1, n_init_min + T])
    primary_log_prob = Normal( f_primary, scale_primary ).log_prob(Y).sum(-1) # [B, Nk, 1]
    logprob = Normal( f_primary, scale_primary ).log_prob(Y)

    gp_mean = torch.concat( [m(X[:, i, None, 0, None, ...]) for i, m in enumerate(process.gp_dist.mean_list)], dim=1 )
    f_contrastive = gp_mean + process.gp_dist.bayesian_linear_model(X.expand([B, Nk, Nf, n_init_min + T, D])) # [B, Nk, Nf, n_init + T]
    scale_contrastive = torch.sqrt( ref_noise_var ).reshape([1, Nk, 1, 1]).expand([B, Nk, Nf, n_init_min + T])
    contrastive_log_prob = Normal( f_contrastive, scale_contrastive ).log_prob(Y.expand([B, Nk, Nf, n_init_min + T])).sum(-1) # [B, Nk, Nf]

    log_prob_combined = torch.concat([primary_log_prob, contrastive_log_prob], dim=-1).logsumexp(-1) # [B, Nk]
    loss_ref = (log_prob_combined - primary_log_prob.squeeze(-1)).mean()
    assert torch.isclose( loss_value, loss_ref, rtol=1e-4, atol=1e-5 )

    # then test cross validation
    rmse_mean, rmse_stderr = loss.validation(process)

    # compute ref value
    rmse_ref = []
    X = torch.cat([X_init[..., :n_init_min, :], X_query[..., :T, :]], dim=-2) # [B, Nk, Nf, n_init_min + T, D]
    Y = torch.cat([Y_init[..., :n_init_min], Y_query[..., :T]], dim=-1) # [B, Nk, Nf, n_init_min + T]
    for i, k in enumerate(ref_kernel_list):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = ref_noise_var[i]
        model = ExactGPModel(X[:, i, ...], Y[:, i, ...], likelihood, ref_mean_list[i], k)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            mu_pred = model(X_grid[:, i, ...]).mean
            rmse = (mu_pred - Y_grid[:, i]).pow(2).mean(-1).sqrt()
            rmse_ref.append(rmse.flatten())
    with torch.no_grad():
        rmse_ref = torch.concat(rmse_ref)
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-4)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-4)
    #
    process.random_subsequence = True
    #


######################################################
# write some helpers to make our life easier later
def _helper_mean(
    ref_mean_list,
    X_prior,
    X_post,
    mask=None,
):
    """
    :param ref_mean_list: [Nk]
    :param X_prior: [B, Nk, Nf, n_prior, D] tensor
    :param X_post: [B, Nk, Nf, n_post, D] tensor
    :param mask: None or [B, Nk, Nf, D] tensor
    :return: mu_prior [B, Nk, Nf, n_prior], mu_post [B, Nk, Nf, n_post]
    """
    mu_prior = torch.concat([
        m(X_prior[:, None, i, ...], mask=mask[:, None, i] if not mask is None else None) for i, m in enumerate(ref_mean_list)
    ], dim=1) # [B, Nk, Nf, n_prior]

    mu_post = torch.concat([
        m(X_post[:, None, i, ...], mask=mask[:, None, i] if not mask is None else None) for i, m in enumerate(ref_mean_list)
    ], dim=1) # [B, Nk, Nf, n_post]
    return mu_prior, mu_post

def _helper_covariance(
    ref_kernel_list,
    ref_noise_var,
    X_prior,
    X_post,
    mask=None,
):
    """
    :param ref_kernel_list: [Nk]
    :param ref_noise_var: [Nk]
    :param X_prior: [B, Nk, Nf, n_prior, D] tensor
    :param X_post: [B, Nk, Nf, n_post, D] tensor
    :param mask: None or [B, Nk, Nf, D] tensor
    :return: K_prior [B, Nk, Nf, n_prior, n_prior], K_cross [B, Nk, Nf, n_prior, n_post], K_post [B, Nk, Nf, n_post, n_post]
    """
    K_prior = torch.concat([
        kernel(X_prior[:, None, i, ...], mask=mask[:, None, i] if not mask is None else None).to_dense() + ref_noise_var[i] * torch.eye(X_prior.shape[-2], device=X_prior.device)
        for i, kernel in enumerate(ref_kernel_list)
    ], dim=1) # [B, Nk, Nf, n_prior, n_prior]

    K_cross = torch.concat([
        kernel(X_prior[:, None, i, ...], X_post[:, None, i, ...], mask=mask[:, None, i] if not mask is None else None).to_dense()
        for i, kernel in enumerate(ref_kernel_list)
    ], dim=1) # [B, Nk, Nf, n_prior, n_post]

    K_post = torch.concat([
        kernel(X_post[:, None, i, ...], mask=mask[:, None, i] if not mask is None else None).to_dense() + ref_noise_var[i] * torch.eye(X_post.shape[-2], device=X_init.device)
        for i, kernel in enumerate(ref_kernel_list)
    ], dim=1) # [B, Nk, Nf, n_post, n_post]

    return K_prior, K_cross, K_post

def _helper_posterior(
    mu_prior,
    mu_post,
    y_prior,
    cov_prior,
    cov_cross,
    cov_post,
    n_prior,
    *,
    return_mu: bool=True
):
    """
    compute GP posterior mean & cov.
    :param mu_prior: [*batch, n_prior.max()] tensor
    :param mu_post: [*batch, n_posterior] tensor
    :param y_prior: [*batch, n_prior.max()] tensor
    :param cov_prior: [*batch, n_prior.max(), n_prior.max()] tensor
    :param cov_cross: [*batch, n_prior.max(), n_posterior] tensor
    :param cov_post: [*batch, n_posterior, n_posterior] tensor
    :param n_prior: [*batch] of int tensor
    :return: [*batch, n_posterior] tensor, [*batch, n_posterior, n_posterior] tensor
    """
    if return_mu:
        assert not mu_prior is None
        assert not mu_post is None
        assert not y_prior is None
    assert n_prior.shape == cov_prior.shape[:-2]

    if return_mu:
        mu = torch.empty_like(mu_post)
    cov = torch.empty_like(cov_post)
    # filter out masked prior points
    batch_shape = cov.shape[:-2]
    _n_prior = n_prior.flatten()
    if return_mu:
        mu = mu.flatten(0, -2)
        _y_prior, _mu_prior, _mu_post = y_prior.flatten(0, -2), mu_prior.flatten(0, -2), mu_post.flatten(0, -2)
    cov = cov.flatten(0, -3)
    _cov_prior, _cov_cross, _cov_post = cov_prior.flatten(0, -3), cov_cross.flatten(0, -3), cov_post.flatten(0, -3)
    for i in range(cov.shape[0]):
        _n = _n_prior[i]
        cov_prior_inv = torch.linalg.inv(_cov_prior[i, :_n, :_n])
        if return_mu:
            mu[i] = _mu_post[i] + (
                (_cov_cross[i, :_n, :].transpose(-1, -2) @ cov_prior_inv) @ (
                    _y_prior[i, :_n, None] - _mu_prior[i, :_n, None])
            ).squeeze(-1)
        cov[i] = _cov_post[i] - (
            _cov_cross[i, :_n, :].transpose(-1, -2) @ cov_prior_inv
        ) @ _cov_cross[i, :_n, :]
    if return_mu:
        return mu.reshape(batch_shape + mu.shape[-1:]), cov.reshape(batch_shape + cov.shape[-2:])
    else:
        return cov.reshape(batch_shape + cov.shape[-2:])

def _helper_entropy(
    cov_prior,
    cov_cross,
    cov_post,
    n_prior,
    id_start,
    id_len,
    *,
    n_unnormalized_prior=0,
    reuse_prestart_samples: bool=False
):
    """
    :param cov_prior: [*batch, n_prior.max(), n_prior.max()] tensor
    :param cov_cross: [*batch, n_prior.max(), n_posterior] tensor
    :param cov_post: [*batch, n_posterior, n_posterior] tensor
    :param n_prior: [*batch] of int tensor
    :param id_start: [*batch] of int tensor, 0 <= each value < n_posterior
    :param id_len: [*batch] of int tensor, each value <= n_posterior,
        meaning we compute the entropy of id_start:id_start+id_len elements
    :param n_unnormalized_prior: int, num in prior points not taken into account in normalization factor
    :param reuse_prestart_samples: flag, whether we use n_init:n_init+n_query1 data as well (this points have gradient as well)
    :return: [*batch] tensor
    """
    if torch.all(n_prior <= 0):
        mu = torch.zeros_like(cov_post[..., 0]) # mean doesn't matter to entropy, just make the shape right
        cov = cov_post
    else:
        assert n_prior.shape == cov_prior.shape[:-2]
        mu = torch.zeros_like(cov_post[..., 0]) # mean doesn't matter to entropy, just make the shape right
        cov = _helper_posterior(None, None, None, cov_prior, cov_cross, cov_post, n_prior, return_mu=False)
    max_T = mu.shape[-1]
    cholesky = torch.linalg.cholesky(cov)
    entropy = MultivariateNormal( mu[..., :1], scale_tril= cholesky[..., :1,:1]).entropy()
    entropy_pre = torch.where(
        id_start==1, entropy, torch.zeros_like(entropy)
    )
    # y_q1 = y_post[:id_start]
    # y_q2 = y_post[id_start:id_start+id_len]
    # (n_prior + id_start + id_len)/(n_prior + id_start) H( y_q1 | y_prior)
    #   + H( y_q2 | y_prior, y_q1)
    # = id_len/(n_prior + id_start) H( y_q1 | y_prior)
    #   + H( y_q1, y_q2 | y_prior)
    pre_factor = id_len / (n_prior - n_unnormalized_prior + id_start).clamp(min=1) if reuse_prestart_samples else -1
    for t in range(2, max_T+1):
        ent_t = MultivariateNormal( mu[..., :t], scale_tril= cholesky[..., :t,:t]).entropy()
        entropy = torch.where(
            id_start+id_len==t, ent_t, entropy
        )
        entropy_pre = torch.where(
            id_start==t+1, ent_t, entropy_pre
        )
    return entropy + pre_factor * entropy_pre

def _helper_log_prob(
    mu_prior,
    mu_post,
    y_prior,
    y_post,
    cov_prior,
    cov_cross,
    cov_post,
    n_prior,
    id_start,
    id_len,
    *,
    n_unnormalized_prior=0,
    reuse_prestart_samples: bool=False
):
    """
    :param mu_prior: [*batch, n_prior.max()] tensor
    :param mu_post: [*batch, n_posterior] tensor
    :param y_prior: [*batch, n_prior.max()] tensor
    :param y_post: [*batch, n_posterior] tensor
    :param cov_prior: [*batch, n_prior.max(), n_prior.max()] tensor
    :param cov_cross: [*batch, n_prior.max(), n_posterior] tensor
    :param cov_post: [*batch, n_posterior, n_posterior] tensor
    :param n_prior: [*batch] of int tensor
    :param full_cov: [*batch, N, N] tensor
    :param n_prior: int, N = n_prior + n_posterior
    :param id_start: [*batch] of int tensor, 0 <= each value < n_posterior
    :param id_len: [*batch] of int tensor, each value <= n_posterior,
        meaning we compute the entropy of id_start:id_start+id_len elements
    :param n_unnormalized_prior: int, num in prior points not taken into account in normalization factor
    :param reuse_prestart_samples: flag, whether we use n_init:n_init+n_query1 data as well (this points have gradient as well)
    :return: [*batch] tensor
    """
    if torch.all(n_prior <= 0):
        mu = mu_post
        cov = cov_post
        y_test = y_post
    else:
        y_test = y_post
        mu, cov = _helper_posterior(
            mu_prior, mu_post, y_prior, cov_prior, cov_cross, cov_post, n_prior,
            return_mu=True)
    max_T = mu.shape[-1]
    cholesky = torch.linalg.cholesky(cov)
    log_prob = MultivariateNormal( mu[..., :1], scale_tril= cholesky[..., :1,:1]).log_prob(y_test[..., :1])
    log_prob_pre = torch.where(
        id_start==1, log_prob, torch.zeros_like(log_prob)
    )
    # y_q1 = y_post[:id_start]
    # y_q2 = y_post[id_start:id_start+id_len]
    # (n_prior + id_start + id_len)/(n_prior + id_start) log p( y_q1 | y_prior)
    #   + log p( y_q2 | y_prior, y_q1)
    # = id_len/(n_prior + id_start) log p( y_q1 | y_prior)
    #   + log p( y_q1, y_q2 | y_prior)
    pre_factor = id_len / (n_prior - n_unnormalized_prior + id_start).clamp(min=1) if reuse_prestart_samples else -1
    for t in range(2, max_T+1):
        log_p_t = MultivariateNormal( mu[..., :t], scale_tril= cholesky[..., :t,:t]).log_prob(y_test[..., :t])
        log_prob = torch.where(
            id_start+id_len==t, log_p_t, log_prob
        )
        log_prob_pre = torch.where(
            id_start==t+1, log_p_t, log_prob_pre
        )
    return log_prob + pre_factor * log_prob_pre

def _helper_observation_mask(y, id_start, id_len):
    """
    take y values from id_start:id_start+id_len, and set the remaining to zeros
    
    :param y: [*batch, N] tensor
    :param id_start: [*batch] of int tensor, 0 <= each value < N
    :param id_len: [*batch] of int tensor, each value <= N
    :return: [*batch, N] tensor
    """
    N_max = y.shape[-1]
    idx = torch.arange(0, N_max).expand(y.shape[:-1] + (N_max, ))
    out = torch.where(idx>= (id_start + id_len).unsqueeze(-1), torch.zeros_like(y), y)
    out = torch.where(idx < id_start.unsqueeze(-1), torch.zeros_like(y), out)
    return out

def _helper_ref_rmse(X, Y, X_test, Y_test, mean_list, kernel_list, noise_var):
    """
    :param X: [B, Nk, Nf, n_prior, D] tensor
    :param Y: [B, Nk, Nf, n_prior] tensor
    :param X_test: [B, Nk, Nf, n_test, D] tensor
    :param Y_test: [B, Nk, Nf, n_test] tensor
    :param mean_list: [Nk] list of mean classes
    :param kernel_list: [Nk] list of kernel classes
    :param noise_var: [Nk] tensor of noise variance
    :return: [Nk * B * Nf] tensor
    """
    rmse_ref = []
    for i, k in enumerate(kernel_list):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise_var[i]
        model = ExactGPModel(X[:, i, ...], Y[:, i, ...], likelihood, mean_list[i], k)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            mu_pred = model(X_test[:, i, ...]).mean
            rmse = (mu_pred - Y_test[:, i]).pow(2).mean(-1).sqrt()
            rmse_ref.append(rmse.flatten())
    with torch.no_grad():
        rmse_ref = torch.concat(rmse_ref)
    return rmse_ref
######################################################

#@pytest.mark.skip(reason="debugging")
@pytest.mark.parametrize(
    "loss, process, use_safety_probability", [
        (GPEntropy1Loss(B, Nk, Nf, num_splits), process, False),
        (GPSafetyEntropyWrapLoss(
            GPEntropy1Loss(B, Nk, Nf, num_splits),
            SafetyProbability.TRIVIAL,
            (0.05, -0.05),
            SafetyProbabilityWrapper.NONE,
            safety_discount_ratio = 1.0,
            ), safe_process, False),
        (GPSafetyEntropyWrapLoss(
            GPEntropy1Loss(B, Nk, Nf, num_splits),
            SafetyProbability.SIGMOID,
            (0.05, -0.05),
            SafetyProbabilityWrapper.LOGCONDITION,
            safety_discount_ratio = 1.0,
            ), safe_process, True),
    ]
)
def test_amortized_al_entropy1_loss(loss, process, use_safety_probability: bool):
    trace = poutine.trace(
        condition(loss.differentiable_loss,
            data=sample_reference),
        graph_type='flat').get_trace(process)
    loss_value = trace.nodes['_RETURN']['value']

    # compute reference value
    K_init, K_cross, K_query = _helper_covariance(ref_kernel_list, ref_noise_var, X_init, X_query, mask=mask_dim)
    # [B, Nk, Nf, n_init_max, n_init_max], [..., n_init_max, T], [..., T, T]
    entropy = _helper_entropy(
        K_init, K_cross, K_query,
        n_init, n_query1, n_query2,
        reuse_prestart_samples=loss.loss_computer.reuse_prestart_samples
    )
    if use_safety_probability:
        p_z = SigmoidSafetyProbability(0.05, -0.05)(Z_query)
        log_pz = _helper_observation_mask(
            (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log(), n_query1, n_query2
        ) # [B, Nk, Nf, T]
        if loss.loss_computer.reuse_prestart_samples:
            log_pz += _helper_observation_mask(
                (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log(), torch.zeros_like(n_query1), n_query1
            ) * ( (n_query2 + n_query1 + n_init) / (n_init + n_query1).clamp(min=1)).unsqueeze(-1)
        entropy = entropy + log_pz.sum(-1)

    assert torch.isclose(
        - loss_value,
        (entropy / (n_init + n_query1 + n_query2)).mean(),
        rtol=1e-4,
        atol=1e-5
    )

    # then test cross validation
    trace = poutine.trace(
        condition(loss.validation,
            data=test_sample_reference),
        graph_type='flat').get_trace(process)
    rmse_mean, rmse_stderr = trace.nodes['_RETURN']['value']

    # compute ref value
    rmse_ref = _helper_ref_rmse(
        torch.cat([X_init[..., :n_init_min, :], X_query], dim=-2),
        torch.cat([Y_init[..., :n_init_min], Y_query], dim=-1),
        X_grid, Y_grid,
        ref_mean_list, ref_kernel_list, ref_noise_var
    )
    with torch.no_grad():
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-6)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-6)


#@pytest.mark.skip(reason="debugging")
@pytest.mark.parametrize(
    "loss, process, use_safety_probability", [
        (GPEntropy2Loss(B, Nk, Nf, num_splits), process, False),
        (GPSafetyEntropyWrapLoss(
            GPEntropy2Loss(B, Nk, Nf, num_splits),
            SafetyProbability.TRIVIAL,
            (0.05, -0.05),
            SafetyProbabilityWrapper.NONE,
            safety_discount_ratio = 1.0,
            ), safe_process, False),
        (GPSafetyEntropyWrapLoss(
            GPEntropy2Loss(B, Nk, Nf, num_splits),
            SafetyProbability.SIGMOID,
            (0.05, -0.05),
            SafetyProbabilityWrapper.LOGCONDITION,
            safety_discount_ratio = 1.0,
            ), safe_process, True),
    ]
)
def test_amortized_al_entropy2_loss(loss, process, use_safety_probability: bool):
    trace = poutine.trace(
        condition(loss.differentiable_loss,
            data=sample_reference),
        graph_type='flat').get_trace(process)
    loss_value = trace.nodes['_RETURN']['value']

    # compute reference value
    mu_init, mu_query = _helper_mean(ref_mean_list, X_init, X_query, mask=mask_dim) # [B, Nk, Nf, n_init_max], [..., T]
    K_init, K_cross, K_query = _helper_covariance(ref_kernel_list, ref_noise_var, X_init, X_query, mask=mask_dim)
    # [B, Nk, Nf, n_init_max, n_init_max], [..., n_init_max, T], [..., T, T]
    log_pY = _helper_log_prob(
        mu_init, mu_query,
        Y_init, Y_query,
        K_init, K_cross, K_query,
        n_init, n_query1, n_query2,
        reuse_prestart_samples=loss.loss_computer.reuse_prestart_samples
    )

    if use_safety_probability:
        p_z = SigmoidSafetyProbability(0.05, -0.05)(Z_query)
        log_pz = _helper_observation_mask(
            (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log(), n_query1, n_query2
        ) # [B, Nk, Nf, T]
        if loss.loss_computer.reuse_prestart_samples:
            log_pz += _helper_observation_mask(
                (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log(), torch.zeros_like(n_query1), n_query1
            ) * ( (n_query2 + n_query1 + n_init) / (n_init + n_query1).clamp(min=1)).unsqueeze(-1)
        log_pY = log_pY - log_pz.sum(-1)# * p_z.log().mean(-1).exp()

    assert torch.isclose(
        loss_value,
        (log_pY / (n_init + n_query1 + n_query2)).mean(),
        rtol=1e-4,
        atol=1e-5
    )

    # then test cross validation
    trace = poutine.trace(
        condition(loss.validation,
            data=test_sample_reference),
        graph_type='flat').get_trace(process)
    rmse_mean, rmse_stderr = trace.nodes['_RETURN']['value']

    # compute ref value
    rmse_ref = _helper_ref_rmse(
        torch.cat([X_init[..., :n_init_min, :], X_query], dim=-2),
        torch.cat([Y_init[..., :n_init_min], Y_query], dim=-1),
        X_grid, Y_grid,
        ref_mean_list, ref_kernel_list, ref_noise_var
    )
    with torch.no_grad():
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-6)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-6)


#@pytest.mark.skip(reason="debugging")
@pytest.mark.parametrize(
    "loss, process, use_safety_probability", [
        (GPMutualInformation1Loss(B, Nk, Nf, 2, num_splits), process, False),
        (GPSafetyMIWrapLoss(
            GPMutualInformation1Loss(B, Nk, Nf, 2, num_splits),
            SafetyProbability.TRIVIAL,
            (0.05, -0.05),
            SafetyProbabilityWrapper.NONE,
            safety_discount_ratio = 1.0,
            ), safe_process, False),
        (GPSafetyMIWrapLoss(
            GPMutualInformation1Loss(B, Nk, Nf, 2, num_splits),
            SafetyProbability.SIGMOID,
            (0.05, -0.05),
            SafetyProbabilityWrapper.LOGCONDITION,
            safety_discount_ratio = 1.0,
            ), safe_process, True),
    ]
)
def test_amortized_al_gpmi1_loss(loss, process, use_safety_probability: bool):
    trace = poutine.trace(
        condition(loss.differentiable_loss,
            data=sample_reference),
        graph_type='flat').get_trace(process)
    loss_value = trace.nodes['_RETURN']['value']

    # compute reference value
    K_grid_init, K_grid_cross, K_query = _helper_covariance(ref_kernel_list, ref_noise_var, torch.cat( [X_grid, X_init], dim=-2 ), X_query, mask=mask_dim)
    # [B, Nk, Nf, n_grid + n_init_max, n_grid + n_init_max], [..., n_grid + n_init_max, T], [..., T, T]
    K_init = K_grid_init[..., n_grid:, n_grid:]# [B, Nk, Nf, n_init_max, n_init_max]
    K_cross = K_grid_cross[..., n_grid:, :]# [B, Nk, Nf, n_init_max, T]

    entropy = _helper_entropy(
        K_init, K_cross, K_query,
        n_init, n_query1, n_query2,
        reuse_prestart_samples=loss.loss_computer.reuse_prestart_samples
    )
    entropy_given_y_grid = _helper_entropy(
        K_grid_init, K_grid_cross, K_query,
        n_grid + n_init, n_query1, n_query2, n_unnormalized_prior=n_grid,
        reuse_prestart_samples=loss.loss_computer.reuse_prestart_samples
    )
    if use_safety_probability:
        p_z = SigmoidSafetyProbability(0.05, -0.05)(Z_query)
        log_pz = _helper_observation_mask(
            (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log(), n_query1, n_query2
        ) # [B, Nk, Nf, T]
        if loss.loss_computer.reuse_prestart_samples:
            log_pz += _helper_observation_mask(
                (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log(), torch.zeros_like(n_query1), n_query1
            ) * ( (n_query2 + n_query1 + n_init) / (n_init + n_query1).clamp(min=1)).unsqueeze(-1)
        entropy = entropy + log_pz.sum(-1)
        entropy_given_y_grid = entropy_given_y_grid

    assert torch.isclose(
        - loss_value,
        ( (entropy - entropy_given_y_grid) / (n_init + n_query1 + n_query2) ).mean(),
        rtol=1e-3,
        atol=1e-4
    )

    # then test cross validation
    trace = poutine.trace(
        condition(loss.validation,
            data=test_sample_reference),
        graph_type='flat').get_trace(process)
    rmse_mean, rmse_stderr = trace.nodes['_RETURN']['value']

    # compute ref value
    rmse_ref = _helper_ref_rmse(
        torch.cat([X_init[..., :n_init_min, :], X_query], dim=-2),
        torch.cat([Y_init[..., :n_init_min], Y_query], dim=-1),
        X_grid, Y_grid,
        ref_mean_list, ref_kernel_list, ref_noise_var
    )
    with torch.no_grad():
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-6)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-6)


#@pytest.mark.skip(reason="debugging")
@pytest.mark.parametrize(
    "loss, process, use_safety_probability", [
        (GPMutualInformation2Loss(B, Nk, Nf, 2, num_splits), process, False),
        (GPSafetyMIWrapLoss(
            GPMutualInformation2Loss(B, Nk, Nf, 2, num_splits),
            SafetyProbability.TRIVIAL,
            (0.05, -0.05),
            SafetyProbabilityWrapper.NONE,
            safety_discount_ratio = 1.0,
            ), safe_process, False),
        (GPSafetyMIWrapLoss(
            GPMutualInformation2Loss(B, Nk, Nf, 2, num_splits),
            SafetyProbability.SIGMOID,
            (0.05, -0.05),
            SafetyProbabilityWrapper.LOGCONDITION,
            safety_discount_ratio = 1.0,
            ), safe_process, True),
    ]
)
def test_amortized_al_gpmi2_loss(loss, process, use_safety_probability: bool):
    trace = poutine.trace(
        condition(loss.differentiable_loss,
            data=sample_reference),
        graph_type='flat').get_trace(process)
    loss_value = trace.nodes['_RETURN']['value']

    # compute reference value
    mu_grid_init, mu_query = _helper_mean(ref_mean_list, torch.cat( [X_grid, X_init], dim=-2 ), X_query, mask=mask_dim) # [B, Nk, Nf, n_grid + n_init_max], [..., T]
    mu_init = mu_grid_init[..., n_grid:] # [B, Nk, Nf, n_init_max]
    K_grid_init, K_grid_cross, K_query = _helper_covariance(ref_kernel_list, ref_noise_var, torch.cat( [X_grid, X_init], dim=-2 ), X_query, mask=mask_dim)
    # [B, Nk, Nf, n_grid + n_init_max, n_grid + n_init_max], [..., n_grid + n_init_max, T], [..., T, T]
    K_init = K_grid_init[..., n_grid:, n_grid:]# [B, Nk, Nf, n_init_max, n_init_max]
    K_cross = K_grid_cross[..., n_grid:, :]# [B, Nk, Nf, n_init_max, T]

    log_prob = _helper_log_prob(
        mu_init, mu_query,
        Y_init, Y_query,
        K_init, K_cross, K_query,
        n_init, n_query1, n_query2,
        reuse_prestart_samples=loss.loss_computer.reuse_prestart_samples
    )
    log_prob_given_y_grid = _helper_log_prob(
        mu_grid_init, mu_query,
        torch.cat([Y_grid, Y_init], dim=-1), Y_query,
        K_grid_init, K_grid_cross, K_query,
        n_grid + n_init, n_query1, n_query2, n_unnormalized_prior=n_grid,
        reuse_prestart_samples=loss.loss_computer.reuse_prestart_samples
    )

    if use_safety_probability:
        p_z = SigmoidSafetyProbability(0.05, -0.05)(Z_query)
        log_pz = _helper_observation_mask(
            (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log(), n_query1, n_query2
        ) # [B, Nk, Nf, T]
        if loss.loss_computer.reuse_prestart_samples:
            log_pz += _helper_observation_mask(
                (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log(), torch.zeros_like(n_query1), n_query1
            ) * ( (n_query2 + n_query1 + n_init) / (n_init + n_query1).clamp(min=1)).unsqueeze(-1)
        log_prob = log_prob - log_pz.sum(-1)
        log_prob_given_y_grid = log_prob_given_y_grid

    assert torch.isclose(
        loss_value,
        ( (log_prob - log_prob_given_y_grid) / (n_init + n_query1 + n_query2) ).mean(),
        rtol=1e-3,
        atol=1e-4
    )

    # then test cross validation
    trace = poutine.trace(
        condition(loss.validation,
            data=test_sample_reference),
        graph_type='flat').get_trace(process)
    rmse_mean, rmse_stderr = trace.nodes['_RETURN']['value']

    # compute ref value
    rmse_ref = _helper_ref_rmse(
        torch.cat([X_init[..., :n_init_min, :], X_query], dim=-2),
        torch.cat([Y_init[..., :n_init_min], Y_query], dim=-1),
        X_grid, Y_grid,
        ref_mean_list, ref_kernel_list, ref_noise_var
    )
    with torch.no_grad():
        rmse_ref_mean = rmse_ref.mean().numpy()
        rmse_ref_stderr = torch.sqrt(rmse_ref.var() / rmse_ref.shape[0]).numpy()
    assert np.isclose(rmse_mean.detach().numpy(), rmse_ref_mean, atol=1e-6)
    assert np.isclose(rmse_stderr.detach().numpy(), rmse_ref_stderr, atol=1e-6)


alpha = 0.2

#@pytest.mark.skip(reason="debugging")
@pytest.mark.parametrize(
    "loss, process", [
        (GPSafetyEntropyWrapLoss(
            GPEntropy2Loss(B, Nk, Nf, num_splits),
            SafetyProbability.GP_POSTERIOR,
            (alpha, ),
            SafetyProbabilityWrapper.LOGCONDITION,
            safety_discount_ratio = 1.0,
            ),
         safe_process),
    ]
)
def test_amortized_al_gp_entropy2_gpsafety_loss(loss, process):
    trace = poutine.trace(
        condition(loss.differentiable_loss, data= sample_reference),
        graph_type='flat').get_trace(process)
    loss_value = trace.nodes['_RETURN']['value']

    # compute reference value
    mu_init, mu_query = _helper_mean(ref_mean_list, X_init, X_query, mask=mask_dim) # [B, Nk, Nf, n_init_max], [..., steps]
    K_init, K_cross, K_query = _helper_covariance(ref_kernel_list, ref_noise_var, X_init, X_query, mask=mask_dim)
    # [B, Nk, Nf, n_init_max, n_init_max], [..., n_init_max, steps], [..., steps, steps]
    log_pY = _helper_log_prob(
        mu_init, mu_query,
        Y_init, Y_query,
        K_init, K_cross, K_query,
        n_init, n_query1, n_query2,
        reuse_prestart_samples=loss.loss_computer.reuse_prestart_samples
    )
    # add safety probability
    muz_init, muz_query = _helper_mean(ref_mean_list_safety, X_init, X_query, mask=mask_dim) # [B, Nk, Nf, n_init_max], [..., T]
    for t in range(T):
        X = torch.cat([X_query[..., :t+1, :].flip(-2), X_init], dim=-2) # [B, Nk, Nf, n_init + t + 1, D]
        Z = torch.cat([Z_query[..., :t+1].flip(-1), Z_init], dim=-1) # [B, Nk, Nf, n_init + t + 1]
        muz = torch.cat([muz_query[..., :t+1].flip(-1), muz_init], dim=-1) # [B, Nk, Nf, n_init + t + 1]

        KS = torch.concat([
            kernel(X[:, None, i]).to_dense() + ref_noise_var_safety[i] * torch.eye(n_init_max + t + 1, device=X.device)
            for i, kernel in enumerate(ref_kernel_list_safety)
        ], dim=1) # [B, Nk, Nf, 1 + t + n_init, 1 + t + n_init]

        muz_pred, varz_pred = _helper_posterior(
            muz[..., 1:], muz[..., :1], Z[..., 1:], KS[..., 1:, 1:], KS[..., 1:, :1], KS[..., :1, :1], n_init + t, return_mu=True)

        safety_dist = Normal(
            muz_pred.squeeze(-1), # [B, Nk, Nf]
            varz_pred.squeeze(-1).squeeze(-1).sqrt()
        )
        p_z = torch.clamp(
            1 - safety_dist.cdf(torch.zeros_like(Z[..., 1])), # [B, Nk, Nf]
            max = (1 - alpha)
        )
        log_pz = (p_z + NUMERICAL_POSITIVE_LOWER_BOUND).log()
        if not loss.loss_computer.reuse_prestart_samples:
            log_pz = torch.where(
                torch.logical_and(n_query1 <= t, t < n_query1 + n_query2),
                log_pz,
                torch.zeros_like(log_pz)
            )
        else:
            log_pz = torch.where(
                t < n_query1 + n_query2,
                log_pz,
                torch.zeros_like(log_pz)
            )
            log_pz = torch.where(
                t < n_query1,
                log_pz * (n_query2 + n_query1 + n_init) / (n_init + n_query1).clamp(min=1),
                log_pz
            )
        log_pY = log_pY - log_pz

    ref_loss = ( log_pY / (n_init + n_query1 + n_query2) ).mean()
    assert torch.isclose(
        loss_value,
        ref_loss,
        rtol=1e-4,
        atol=1e-5
    ), f'{loss_value} vs true value {log_pY.mean()}'





