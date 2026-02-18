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
from gpytorch.priors import Prior
from alef.means.pytorch_means.gpytorch_mean_wrapper import WrapMean

from alef.configs.base_parameters import INPUT_DOMAIN

class LinearPytorchMean(WrapMean):
    def __init__(
        self,
        input_size,
        batch_shape=torch.Size(),
        name: str='',
        *,
        weights_prior: Prior=None,
        scale_prior: Prior=None,
        bias_prior: Prior=None,
        **kwargs
    ):
        """
        m(x) = c * ( b + sum_d w_d x_d )
        """
        super().__init__()
        pass_kwargs = {'device': kwargs['device']} if kwargs.get('device') else {}
        self.name = name
        
        center = (INPUT_DOMAIN[1] + INPUT_DOMAIN[0]) / 2
        self.register_parameter(
            name='weights',
            parameter=torch.nn.Parameter( torch.ones([*batch_shape, input_size], **pass_kwargs) )
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
            parameter=torch.nn.Parameter( 2.5*torch.ones([*batch_shape], **pass_kwargs) )
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
            parameter=torch.nn.Parameter( -2.5*center*torch.ones([*batch_shape], **pass_kwargs) )
        )
        if not bias_prior is None:
            self.register_prior(
                "bias_prior",
                bias_prior.expand([*batch_shape]),
                lambda m: m.bias,
                lambda m, v: m._set_bias(v)
            )

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
        # input is [*B, N, D], scale is [*B, D], up to broadcast permit
        x = input
        if not mask is None:
            x = input.masked_fill(mask.unsqueeze(-2)==0, 0)
        shape = torch.broadcast_shapes(x.shape, self.weights.unsqueeze(-2).shape)
        product = x.expand(shape) * self.weights.unsqueeze(-2).expand(shape)
        return self.scale.unsqueeze(-1).expand(shape[:-1]) * (
            product.sum(dim=-1) + self.bias.unsqueeze(-1).expand(shape[:-1])
        )

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np
    D = 2
    B = 3
    m = LinearPytorchMean(D, [])
    x = torch.rand([4, B, 10, D])
    print(m(x).shape)
    x = torch.rand([1000, D])
    print(m.state_dict())
    fig, axs = plt.subplots(1, 1)
    p = axs.tricontourf(
        x[:,0].numpy(), x[:,1].numpy(), m(x).detach().numpy(),
        levels=np.linspace(-5, 5, 100),
        cmap='seismic',
        alpha=1.0
    )
    fig.colorbar(p, ax=axs)
    plt.show()
    