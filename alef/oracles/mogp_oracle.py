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

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Union, Sequence
from functools import partial
from scipy import interpolate
from gpflow.utilities import tabulate_module_summary, leaf_components

from alef.utils.utils import create_grid
from alef.oracles.base_oracle import StandardOracle

class MOGPOracle(StandardOracle):
    def __init__(
        self,
        kernel_list,
        input_dimension: int,
        W: np.ndarray,
        observation_noise: float,
        specified_output: int
    ):
        D = input_dimension
        if not kernel_list is None:
            assert np.all([kern.input_dimension == D for kern in kernel_list])
        super().__init__(observation_noise, -np.inf, np.inf, D)
        self.observation_noise = observation_noise
        self.W = W
        self.kernel_list = kernel_list
        self.__specified_output = specified_output

    def set_specified_output(self, specified_output:int):
        self.__specified_output = specified_output

    def draw_from_hyperparameter_prior(self):
        print("-Draw from hyperparameter prior")
        for i, kernel in enumerate(self.kernel_list):
            for parameter in kernel.trainable_parameters:
                # old_value=parameter.numpy()
                # print("Old value")
                # print(old_value)
                print(parameter)
                new_value = parameter.prior.sample()
                parameter.assign(new_value)
            print(f'Latent kernel {i}')
            print_summary(kernel)

    def initialize(self, a, b, n, normalized_output: bool=False):
        self._export_kernel()
        D = self.get_dimension()
        obs_noise = self.observation_noise
        super().__init__(obs_noise, a, b, D)
        grid, function_values = self._gp_sample(a, b, n, normalized_output=normalized_output)
        points, values = self._turn_grid_to_mesh(grid, function_values, n)
        self.__mapping = [points, values]
    
    def initialize_from_txt(self, file_path, output_name='function_values.txt'):
        D = self.get_dimension()
        obs_noise = self.observation_noise
        a, b = np.loadtxt(os.path.join(file_path, 'box_bounds.txt'))
        super().__init__(obs_noise, a, b, D)

        grid = np.loadtxt(os.path.join(file_path, 'grid.txt'))
        function_values = np.loadtxt(os.path.join(file_path, output_name))

        N = grid.shape[0]
        n_per_dim = int(N**(1/D))
        points, values = self._turn_grid_to_mesh(grid, function_values, n_per_dim)
        self.__mapping = [points, values]
    
    def save_gp_initialization_to_txt(self, a, b, n, file_path, output_name='function_values.txt', normalized_output: bool=False):
        self._export_kernel(path=os.path.join(file_path, 'kernel_documentation.json'))
        
        grid, function_values = self._gp_sample(a, b, n, normalized_output=normalized_output)

        np.savetxt(os.path.join(file_path, 'box_bounds.txt'), [a, b])
        np.savetxt(os.path.join(file_path, 'grid.txt'), grid)
        np.savetxt(os.path.join(file_path, output_name), function_values)

    def _export_kernel(self, path: str=None):
        if path is None:
            print(f'W: {self.W}\n')
            print('\nlatent kernels')
            for LatKern in self.kernel_list:
                #print(f'name: {LatKern.__class__.__name__}\n')
                print(tabulate_module_summary(LatKern, None))
        else:
            if path.endswith('.txt'):
                with open(path, mode='w') as fp:
                    print(f'W: {self.W}\n', file=fp)
                    print('\nlatent kernels', file=fp)
                    for LatKern in self.kernel_list:
                        #print(f'name: {LatKern.__class__.__name__}\n', file=fp)
                        print(tabulate_module_summary(LatKern, None), file=fp)
            elif path.endswith('.json'):
                kernel = {
                    'W': self.W.tolist(),
                    'latent_k': {
                        i:{
                            var_name: var_value.numpy().tolist() for var_name, var_value in leaf_components(k).items()
                        } for i, k in enumerate(self.kernel_list)
                    }
                }
                with open(path, 'w') as f:
                    json.dump(kernel, f)

    def _gp_sample(self, a, b, n_per_dim, normalized_output: bool=False): 
        """
        return grid (inputs), gp samples of the grid
        """
        D = self.get_dimension()
        grid = create_grid(a, b, n_per_dim, D)
        K_list = [kern(grid, grid).numpy() for kern in self.kernel_list]
        n = grid.shape[0]
        L, P, R = self.W.shape
        
        Chol_stack = [np.kron(
            np.linalg.cholesky( self.W[l, :, :] @ self.W[l,:,:].T ),
            np.linalg.cholesky( K_list[l] )
            ) for l in range(L)] # L elements, each [Pn, Pn]
        
        samples = np.random.standard_normal(size=[P*n, L])
        F_latent = np.concatenate(
            [ np.reshape(Chol_stack[l] @ samples[:, l, None], [P, n, 1]) for l in range(L) ],
            axis=-1
        ) # [P, n, L]
        F = F_latent.sum(axis=-1) # [P, n]
        F = F.T # [n, P]
        
        #
        if normalized_output:
            return grid, (F - F.mean(axis=0))/ np.sqrt(F.var(axis=0) )
        else:
            return grid, F

    def _turn_grid_to_mesh(self, X, F, n_per_dim):
        """
        from grid to mesh
        X: [N, D]
        F: [N, P]

        return
        (X1, ..., XD): recover from X = np.meshgrid(X1, X2, ... XD, indexing='ij'), Xi = np.linspace(a[i], b[i], n_i)
        mesh F: [n_1, ..., n_D x P], the corresponding P dim F values

        our X was generated by create_grid, which is the same as [X1[i1, ..., iD], ..., XD[i1, ..., iD] for iD in range(n_D) for ... for i1 in range(n_1)] when each Xi = np.linspace(a[i], b[i], n_i)
        """
        D = self.get_dimension()
        P = F.shape[1]
        # mesh grids
        n_unflattened = [n_per_dim] * D
        mesh_grid_tuple = tuple([X.reshape(n_unflattened + [D])[..., -i] for i in range(1, D+1)])
        X_list = [mesh_grid_tuple[0].reshape(n_per_dim, -1)[:,0]]
        for i in range(1, D):
            X_list.append(np.moveaxis(mesh_grid_tuple[i], i, 0).reshape(n_per_dim, -1)[:,0])

        mesh_F = F.reshape(n_unflattened + [P])
        return tuple(X_list), mesh_F

    def f(self, x):
        p = self.__specified_output
        points, values = self.__mapping
        if p >= 0:
            return interpolate.interpn(points, values[..., p], x, method="linear")
        else:
            P = values.shape[-1]
            return np.concatenate([
                interpolate.interpn(points, values[..., i], x, method="linear") for i in range(P)
            ])

    def query(self, x, noisy=True):
        p = self.__specified_output
        if p >=0:
            function_value = np.squeeze(self.f(x))
            if noisy:
                epsilon = np.random.normal(0, self.observation_noise, 1)[0]
                function_value += epsilon
        else:
            function_value = np.reshape(self.f(x), -1)
            if noisy:
                epsilon = np.random.normal(0, self.observation_noise, len(function_value))
                function_value += np.reshape(epsilon, -1)
        return function_value

if __name__=="__main__":
    from matplotlib import pyplot as plt
    from alef.kernels.kernel_factory import KernelFactory
    from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
    from alef.configs.kernels.matern52_configs import BasicMatern52Config, Matern52WithPriorConfig


    def sample_kernel(D=1, P=2, R=2, L=2):
        kernel_list = [
            KernelFactory.build(BasicMatern52Config(
                input_dimension=D,
                base_lengthscale=list(np.random.uniform(0.1, 1, size=[D])),
                base_variance=1
            )) for _ in range(L)
        ]
        W = np.zeros([L, P, R], dtype=float)
        for l in range(L):
            w_l = np.random.uniform(-1, 1, size=[P, R])
            w_norm = np.linalg.norm(w_l, axis=1).reshape([P,1])
            w_l = w_l/w_norm
            W[l,:,:] = w_l
        
        return kernel_list, W
    D = 2
    kernel_list, W = sample_kernel(D=D, P=2, R=2, L=2)
    oracle = MOGPOracle(kernel_list, D, W, 0.01, 0)

    oracle.initialize(-1, 1, 10)
    oracle.set_specified_output(1)
    X, Y = oracle.get_random_data(5000, noisy=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], Y, marker=".")
    plt.show()
