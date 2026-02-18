import numpy as np
import gpflow
from alef.kernels.multi_output_kernels.coregionalization_kernel import CoregionalizationMOKernel
from alef.kernels.multi_output_kernels.latent_kernel_enum import LatentKernel
from alef.models.mo_gpr import MOGPR

N = 50
D = 3
P = 2

# data
X = np.random.normal(size=[N,D])
Y = np.random.normal(size=[N,P]) * 3

# kernel
kernel = CoregionalizationMOKernel([1.3,0.8], [0.9,0.75], D, P, latent_kernel=LatentKernel.MATERN52, add_prior=True, lengthscale_prior_parameters= (1, 9), variance_prior_parameters= (1, 0.3), active_on_single_dimension=False, active_dimension=None, name=None)
# mean function
m = gpflow.mean_functions.Constant()
# model
model = MOGPR((X, Y), kernel, mean_function=m, noise_variance=[0.2, 0.3])
W_value = np.eye(P)
for l in range(P):
    model.kernel.kernel.W_list[l].assign(W_value[:, l])

# comparison model
k1 = gpflow.kernels.Matern52(1.3, 0.9)
k2 = gpflow.kernels.Matern52(0.8, 0.75)
mm1 = gpflow.models.GPR((X, Y[..., 0, None]), k1, mean_function=m, noise_variance=0.2)
mm2 = gpflow.models.GPR((X, Y[..., 1, None]), k2, mean_function=m, noise_variance=0.3)

def test_log_marginal_likelihood():
    MO_log_marg_lik = model.log_marginal_likelihood().numpy()
    SO1_log_marg_lik = mm1.log_marginal_likelihood().numpy()
    SO2_log_marg_lik = mm2.log_marginal_likelihood().numpy()

    assert np.allclose(MO_log_marg_lik, SO1_log_marg_lik + SO2_log_marg_lik)


def test_predict_f():
    Xt = np.random.normal(size=[10, D])

    mu1, cov1 = mm1.predict_f( Xt, full_cov=True)
    mu2, cov2 = mm2.predict_f( Xt, full_cov=True)

    mu, cov = model.predict_f( Xt, full_cov=True, full_output_cov=True)
    assert np.allclose(mu[..., 0, None], mu1)
    assert np.allclose(mu[..., 1, None], mu2)
    assert np.allclose(cov[..., 0, :, 0], cov1[0,...])
    assert np.allclose(cov[..., 1, :, 1], cov2[0,...])

    mu, cov = model.predict_f( Xt, full_cov=True, full_output_cov=False)
    assert np.allclose(mu[..., 0, None], mu1)
    assert np.allclose(mu[..., 1, None], mu2)
    assert np.allclose(cov[0], cov1[0,...])
    assert np.allclose(cov[1], cov2[0,...])

    mu1, var1 = mm1.predict_f( Xt, full_cov=False)
    mu2, var2 = mm2.predict_f( Xt, full_cov=False)
    mu, var = model.predict_f( Xt, full_cov=False, full_output_cov=False)

    assert np.allclose(mu[..., 0, None], mu1)
    assert np.allclose(mu[..., 1, None], mu2)
    assert np.allclose(var[..., 0, None], var1)
    assert np.allclose(var[..., 1, None], var2)

def test_predict_y():
    Xt = np.random.normal(size=[10, D])

    mu1, var1 = mm1.predict_y( Xt, full_cov=False)
    mu2, var2 = mm2.predict_y( Xt, full_cov=False)
    mu, var = model.predict_y( Xt, full_cov=False, full_output_cov=False)

    assert np.allclose(mu[..., 0, None], mu1)
    assert np.allclose(mu[..., 1, None], mu2)
    assert np.allclose(var[..., 0, None], var1)
    assert np.allclose(var[..., 1, None], var2)


if __name__ == '__main__':
    test_log_marginal_likelihood()
    test_predict_f()
    test_predict_y()
