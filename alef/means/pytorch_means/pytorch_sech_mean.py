# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from torch.nn.utils import parametrize
from gpytorch.priors import Prior
from alef.means.pytorch_means.gpytorch_mean_wrapper import WrapMean

from alef.configs.base_parameters import INPUT_DOMAIN

def sech(x):
    return 2*torch.exp(x) / (1.0 + torch.exp(2*x))

class SechPytorchMean(WrapMean):
    def __init__(
        self,
        input_size,
        batch_shape=torch.Size(),
        name: str='',
        *,
        center_prior: Prior=None,
        weights_prior: Prior=None,
        scale_prior: Prior=None,
        bias_prior: Prior=None,
        **kwargs
    ):
        """
        m(x) = c * ( b + sech( 1/d sum_d w_d(x_d - center_d)**2 ) )
        """
        super().__init__()
        pass_kwargs = {'device': kwargs['device']} if kwargs.get('device') else {}
        self.name = name
        center = (INPUT_DOMAIN[1] + INPUT_DOMAIN[0]) / 2 * torch.ones([*batch_shape, input_size], **pass_kwargs)
        self.register_parameter(
            name='center',
            parameter=torch.nn.Parameter(center)
        )
        if not center_prior is None:
            self.register_prior(
                "center_prior",
                center_prior.expand([*batch_shape, input_size]),
                lambda m: m.center,
                lambda m, v: m._set_center(v)
            )
        self.register_parameter(
            name='weights',
            parameter=torch.nn.Parameter( 10*torch.ones([*batch_shape, input_size], **pass_kwargs) )
        )
        if not weights_prior is None:
            self.register_prior(
                "weights_prior",
                weights_prior.expand([*batch_shape, input_size]),
                lambda m: m.weights,
                lambda m, v: m._set_weights(v)
            )
        self.register_parameter(
            name='scale',
            parameter=torch.nn.Parameter( 3*torch.ones([*batch_shape], **pass_kwargs) )
        )
        if not scale_prior is None:
            self.register_prior(
                "scale_prior",
                scale_prior.expand([*batch_shape]),
                lambda m: m.scale,
                lambda m, v: m._set_scale(v)
            )
        self.register_parameter(
            name='bias',
            parameter=torch.nn.Parameter( -1/2*torch.ones([*batch_shape], **pass_kwargs) )
        )
        if not bias_prior is None:
            self.register_prior(
                "bias_prior",
                bias_prior.expand([*batch_shape]),
                lambda m: m.bias,
                lambda m, v: m._set_bias(v)
            )

    def _set_center(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.center)
        self.center.data = value

    def _set_weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.weights)
        self.weights.data = value

    def _set_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.scale)
        self.scale.data = value

    def _set_bias(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.bias)
        self.bias.data = value

    def forward(self, input, mask=None):
        """
        :param input: [*batch_shape, N, D] torch.Tensor
        :param mask: None or [*batch_shape, D] torch.Tensor
        :return: [*batch_shape] torch.Tensor
        """
        # input is [*B, N, D], center / weights is [*B, D], up to broadcast permit
        shape = torch.broadcast_shapes(input.shape, self.center.unsqueeze(-2).shape)
        shifted_input = input.expand(shape) - self.center.unsqueeze(-2).expand(shape)
        if mask is None:
            q = sech( (self.weights.unsqueeze(-2).expand(shape) * shifted_input**2).mean(-1) )
        else:
            m = mask.unsqueeze(-2).expand(shape)
            shifted_input = shifted_input.masked_fill(m==0, 0)
            q = sech(
                (self.weights.unsqueeze(-2).expand(shape) * shifted_input**2).sum(-1) / m.sum(-1)
            )
        return self.scale.unsqueeze(-1).expand(shape[:-1]) * ( q + self.bias.unsqueeze(-1).expand(shape[:-1]) )

class SechRotatedPytorchMean(SechPytorchMean):
    def __init__(
        self,
        input_size,
        batch_shape=torch.Size(),
        name: str='',
        *,
        center_prior: Prior=None,
        weights_prior: Prior=None,
        axis_prior: Prior=None,
        scale_prior: Prior=None,
        bias_prior: Prior=None,
        **kwargs
    ):
        super().__init__(input_size, batch_shape, name, center_prior=center_prior, weights_prior=weights_prior, scale_prior=scale_prior, bias_prior=bias_prior)
        pass_kwargs = {'device': kwargs['device']} if kwargs.get('device') else {}
        self.register_parameter(
            name='eigenvectors',
            parameter=torch.nn.Parameter(
                torch.eye(input_size, **pass_kwargs).expand([*batch_shape, input_size, input_size])
            )
        )
        if not axis_prior is None:
            self.register_prior(
                "axis_prior",
                axis_prior.expand([*batch_shape, input_size, input_size]),
                lambda m: m.eigenvectors,
                lambda m, v: m._set_eigenvectors(v)
            )

    def _set_eigenvectors(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.eigenvectors)
        
        """
        # torch.linalg.householder_product needs nvidia GPU & cuSOLVER
        h, tau = torch.geqrf(value)
        Q_hp = torch.linalg.householder_product(h.to('cpu'), tau.to('cpu')).to(self.eigenvectors.device)
        self.eigenvectors.data = Q_hp
        """
        Q, R = torch.linalg.qr(value, mode= 'complete')
        self.eigenvectors.data = Q

    def forward(self, input, mask=None):
        """
        :param input: [*batch_shape, N, D] torch.Tensor
        :param mask: None or [*batch_shape, D] torch.Tensor
        :return: [*batch_shape] torch.Tensor
        """
        # input is [*B, N, D], center / weights is [*B, D], up to broadcast permit
        shape = torch.broadcast_shapes(input.shape, self.center.unsqueeze(-2).shape) # [*B, N, D]
        shifted_input = input.expand(shape) - self.center.unsqueeze(-2).expand(shape) # [*B, N, D]
        eigenvectors = self.eigenvectors # [*B, D, D]
        E = self.weights.unsqueeze(-2).sqrt() * eigenvectors # [*B, 1, D] * [*B, D, D], weighted columns
        if mask is None:
            Ex = torch.matmul(shifted_input, E) # [*B, N, D]
            q = sech( (Ex**2).mean(-1) )
        else:
            m = mask.unsqueeze(-2).expand(shape)
            shifted_input = shifted_input.masked_fill(m==0, 0)
            Ex = torch.matmul(shifted_input, E) # [*B, N, D]
            q = sech( (Ex**2).sum(-1) / m.sum(-1) )
        return self.scale.unsqueeze(-1).expand(shape[:-1]) * ( q + self.bias.unsqueeze(-1).expand(shape[:-1]) )

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np
    D = 2
    B = 3
    m = SechRotatedPytorchMean(D, [])
    m._set_weights(
        [10, 20]
        #15/np.array([0.48, 0.83, 0.6, 0.3, 0.44])
        #5/np.array([0.48, 3.18, 0.83, 0.6, 0.3, 3.4, 0.44])
    )
    #m._set_eigenvectors(
    #    np.array([[0.8, 0.6], [-0.6, 0.8]])
    #)
    #m._set_center([0.65]*D)
    m._set_scale(3.2)
    m._set_bias(-0.47)
    x = torch.rand([10**(D+1), D])
    y = m(x)
    print(m.state_dict())
    print(y.shape)
    print('y >= 0: %.1f %%'%( (y >= 0).sum() / x.shape[0] * 100 ) )
    print( y.mean(), y.var() )
    if D==2:
        fig, axs = plt.subplots(1, 1)
        p = axs.tricontourf(
            x[:,0].numpy(), x[:,1].numpy(), y.detach().numpy(),
            levels=np.linspace(-5, 5, 100),
            cmap='seismic',
            alpha=1.0
        )
        fig.colorbar(p, ax=axs)
        plt.show()
    elif D==1:
        fig, axs = plt.subplots(1, 1)
        axs.plot( x[:,0].numpy(), y.detach().numpy(), 'o')
        axs.set_ylim([-5, 5])
        plt.show()
    
    print('center [0.4, 0.6]')
    x = torch.rand([10**(D+1), D])*0.2 + 0.4
    y = m(x)
    print('y >= 0: %.1f %%'%( (y >= 0).sum() / x.shape[0] * 100 ) )
        
    