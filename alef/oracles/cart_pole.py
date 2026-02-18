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

import numpy as np
import os
import logging
from typing import List, Union
from scipy.integrate import solve_ivp
from alef.oracles.base_oracle import StandardOracle
from alef.oracles.base_constrained_oracle import StandardConstrainedOracle
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from alef.utils.custom_logging import getLogger
logger = getLogger(__name__)
"""
ref:
Marc Peter Deisenroth, Dieter Fox, and Carl Edward Rasmussen,
Gaussian Processes for Data-Efficient Learning in Robotics and Control,
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015

Marc Peter Deisenroth,
Efficient Reinforcement Learning using Gaussian Processes,
PhD thesis, KIT

Jens Schreiter, Duy Nguyen-Tuong, Mona Eberts, Bastian Bischoff, Heiner Markert, and Marc Toussaint
Safe Exploration for Active Learning with Gaussian Processes,
ECML 2015
"""

gravity_acc = 9.82 # m/s^2, acceleration of gravity
batch_size = 1 # controller is applied to a batch of random initial state to get one performance score
control_freq = 50 # Hz, frequency of control
record_freq = max(10, control_freq)
time_window = 4 # s, each trajectory last for 10 seconds
friction_coeff = 0.1 # Ns/m, coeff. of friction between cart and ground
default_parameters = np.array([
    0.5,# kg, mass of cart
    0.5,# kg, mass of pendulum
    0.5,# m, length of pendulum
])

class CartPole(StandardOracle):
    def __init__(
        self,
        observation_noise: float=0.1,
        constants: np.ndarray = default_parameters,
        reduce_dimension: bool=False,
    ):
        """
        :param observation_noise: standard deviation of the observation noise (Gaussian noise)
        :param constants: (m_c, m_p, l, f, freq)
            m_c: mass of cart (kg),
            m_p: mass of pendulum (kg),
            l: length of pendulum (m),
        )
        :param reduce_dimension: if False, our linear contoller is x*s + x4 (5 dim), otherwise x*s (4 dim)
        """
        self._constants = np.squeeze(constants)
        assert self._constants.shape == default_parameters.shape
        D = 4 if reduce_dimension else 5
        super().__init__(observation_noise, 0.0, 1.0, D)
        self.reduce_dimension = reduce_dimension
        self.reset(noisy=False)

    def _centralize_state(self, state, shift: bool):
        """
        state: [..., 4] array
        shift: if True, return state - target_state
        """
        loc = state[...,0]
        loc_dot = state[..., 1]
        theta = state[..., 2] % (2*np.pi)
        theta_dot = state[..., 3]
        new_state = np.concatenate(
            [
                loc[..., None],
                loc_dot[..., None],
                theta[..., None],
                theta_dot[..., None]
            ],
            axis=-1)
        if shift:
            return new_state - self.target_state
        else:
            return new_state

    @property
    def state(self):
        """
        see Deisenroth et al. 2015
        
        loc, velocity, angle, angular speed
            loc: location of cart (m)
            velocity: of cart (m/s)
            angle: of pendulum, counter-clockwise from bottom (rad),
            angular speed: of pendulum (rad/s)
        """
        return self._state

    @property
    def target_state(self):
        return np.array([0, 0, np.pi, 0])

    def reset(self, noisy:bool):
        """
        see Schreiter et al. 2015
        """
        self._state = np.tile(
            self.target_state[None,...],# loc, velocity, angle, angular speed
            [batch_size, 1]
        )
        noise = np.random.normal(0, self.observation_noise, [batch_size, 4]) if noisy else np.zeros([batch_size, 4])
        self.update_state(noise)
        self.update_trajectory(np.array([0]), self.state[None, ...])

    def update_state(self, ds):
        new_state = self.state + ds
        self._state = self._centralize_state(new_state, shift=False)

    def update_trajectory(self, t:np.ndarray, states:np.ndarray):
        """
        t: [T,] array
        states: [T, batch_size, 4] array
        """
        self.trajectory_time_map = t
        self.trajectory = self._centralize_state(states, shift=False)

    def controller(self, x:np.ndarray, state) -> float:
        """
        see Schreiter et al. 2015
        
        linear controller with loc=0, theta=pi as reference
        return: force u= x[:4]^T @ state + x[4]
        """
        assert x.ndim == 1
        if self.reduce_dimension:
            assert len(x) == 4
            u = np.inner(x, self._centralize_state(state, shift=True))
        else:
            assert len(x) == 5
            u = np.inner(x[:4], self._centralize_state(state, shift=True)) + x[-1]
        # set x to negative: loc, v too positive -> need negative force
        #                           ... neg...   -> ...  positive ...
        #                    theta neg -> need to push right (positive force)
        #                      ... pos -> ...................(neg...        )
        return np.clip(u, -10, 10)

    def difference_target_tip(self, state):
        """
        compute tip loc from state, then compute the diff vector of tip loc to target tip loc
        
        return (tip_x, tip_y) - (0, l), l is the pendulum length
        """
        new_state = self._centralize_state(state, shift=True)
        _, _, l = self._constants
        loc = new_state[..., 0]
        theta = new_state[..., 2]
        # loc, theta are diff to target states
        # so we just compare (loc + l sin(theta), l cos(theta)) against (0, l)
        diff_x = loc - l*np.sin(theta)
        diff_y = l*(np.cos(theta) - 1)
        return np.concatenate((diff_x[...,None], diff_y[..., None]), axis=-1)

    def loss(self, state):
        """
        see Deisenroth et al. 2015
        desired: state=[[0, doesn't matter, \pi, doesn't matter]]
        """
        _, _, l = self._constants
        tip_diff = np.linalg.norm(
            self.difference_target_tip(state),
            ord=2, axis=-1
        )
        loss = np.mean(
            1 - np.exp( - (25/l * tip_diff)**2)
        ) # [0, 1]
        return loss

    def c(self, state):
        """
        see Schreiter et al. 2015
        
        unsafe when fall, so we take angle as constraint value
        """
        tip_diff = np.linalg.norm(
            self.difference_target_tip(state),
            ord=2, axis=-1
        )
        # see if tip is around target, tolerance radius l/2 meter
        # meaning at target loc, rad can't exceed ~0.5 rad (28.65 degree)
        _, _, l = self._constants
        dist = np.min(
            0.5 - 1/l * tip_diff
        )
        return 2*( -1 + np.exp(dist) )

    def state_derivative(self, state, u):
        """
        see Deisenroth PhD thesis, appendix C
        
        Args:
            state: [N, 4] array
            u: force, [N,] array
        return:
            d_state: [N, 4]
        """
        m_c, m_p, l = self._constants
        f = friction_coeff
        new_state = self._centralize_state(state, shift=False)
        v = new_state[...,1]
        theta = new_state[...,2]
        a_theta = new_state[...,3]

        d_v = (
            2*m_p*l*(a_theta**2)*np.sin(theta) + \
            3*m_p*gravity_acc*np.sin(theta)*np.cos(theta) + \
            4*u - 4*f*v
        ) / (
            4*(m_c + m_p) - 3*m_p*(np.cos(theta)**2)
        )
        d_a_theta = (
            -3*m_p*l*(a_theta**2)*np.sin(theta)*np.cos(theta) + \
            -6*(m_c + m_p)*gravity_acc*np.sin(theta) + \
            -6*(u - f*v)*np.cos(theta)
        ) / (
            4*l*(m_c + m_p) - 3*m_p*l*(np.cos(theta)**2)
        )
        assert not np.any(np.isnan(d_v)), d_v
        assert not np.any(np.isnan(d_a_theta)), d_a_theta
        return np.hstack((
            v[..., None],
            d_v[..., None],
            a_theta[..., None],
            d_a_theta[..., None]
        ))

    def sample_constants(self):
        mc, mp, l = default_parameters
        margin = 0.05
        return np.array([
            np.random.uniform(low=(1.0-margin)*mc, high=(1.0+margin)*mc),
            np.random.uniform(low=(1.0-margin)*mp, high=(1.0+margin)*mp),
            np.random.uniform(low=(1.0-margin)*l, high=(1.0+margin)*l)
        ])

    def x_scale(self, x: np.ndarray):
        r"""
        rescale x=[x1, x2, x3, x4, x5] as if we are considering x in
            [0, 70] x
            [10, 20] x
            [-55, -50] x
            [-15, -5] x
            [150, 180]
        """
        assert x.shape[-1] == self.get_dimension(), x.shape
        desire_bound = np.array([
            [0, 70],
            [10, 20],
            [-55, -50],
            [-10, -5],
        ]) if self.reduce_dimension else np.array([
            [0, 70],
            [10, 20],
            [-55, -50],
            [-10, -5],
            [-0.5, 0.5],
        ])
        a, b = self.get_box_bounds()
        return (x - a) / (b - a) * (desire_bound[:,1] - desire_bound[:,0]) + desire_bound[:,0]

    def run_episode(self, x, *, noisy: bool):
        self.reset(noisy=noisy)
        time_stamp = np.linspace(0, time_window, record_freq*time_window + 1)
        u_log = {}
        def ode(t, y):
            state = y.reshape(batch_size, 4)
            u_idx = np.floor(control_freq * t)
            if u_idx in u_log:
                u = u_log[u_idx]
            else:
                u = self.controller(self.x_scale(x), state)
                u_log[u_idx] = u
            ds = self.state_derivative(state, u)
            return ds.reshape(-1)
        def hit_ground(t, y):
            state = self._centralize_state( y.reshape(batch_size, 4), shift=True )
            theta = state[..., 2]
            if np.all( np.abs(theta) >= np.pi/2 ):
                return 0
            else:
                return 1
        hit_ground.terminal = True
        #hit_ground.direction = -1

        sol = solve_ivp(
            ode,
            [0, time_window],
            self.trajectory[0].reshape(-1),
            t_eval=time_stamp,
            #events=hit_ground
        )
        if not sol.success:
            logger.info(f'trial with x={x} is terminated ealier due to failed ode solv')
        elif len(sol.t) < record_freq*time_window + 1:
            logger.debug(f'trial with x={x} is terminated ealier due to unsafe state')
            
        trajecs = sol.y.T.reshape(len(sol.t), batch_size, 4)
        self.update_trajectory(sol.t, trajecs)
        self.update_state(trajecs[-1])
        return True

    def query(self, x, noisy=True):
        assert self.run_episode(x, noisy=noisy)
        states = self.trajectory
        return self.loss(states)

    def plot(self):
        fig, ani = self._plot_trajectory(False)

    def save_plot(self, file_name: str = None, file_path: str = None):
        fig, ani = self._plot_trajectory(True, file_name=file_name, file_path=file_path)

    def _plot_trajectory(self, store: bool, file_name: str = None, file_path: str = None):
        theta_bin = np.linspace(0, 2*np.pi, 100)
        l = self._constants[2] # length of pendulum

        fig, axs = plt.subplots(2, 2, sharex='col')
        t = self.trajectory_time_map
        states = self._centralize_state(self.trajectory, shift=True)
        locs = states[...,0]
        thetas = states[...,2]
        axs[0,0].plot(np.tile(t[:,None], [1, batch_size]), locs)
        axs[1,0].plot(np.tile(t[:,None], [1, batch_size]), thetas)
        axs[0,0].set_title('cart location, target 0 (m)')
        axs[1,0].set_title('pendulum angle, target 0 (rad)')
        
        tip_diff = self.difference_target_tip(self.trajectory)
        axs[0,1].plot(tip_diff[..., 0], tip_diff[..., 1] + l)
        axs[0,1].plot([0], [l], 'o', color='black', label='target tip location (0, pendulum_length)')
        xlim=axs[0,1].get_xlim()
        ylim=axs[0,1].get_ylim()
        axs[0,1].hlines([0], [-2], [2], colors='black')
        axs[0,1].vlines([0], [-2], [2], colors='black')
        axs[0,1].legend()
        axs[0,1].set_xlim(xlim)
        axs[0,1].set_ylim([min(-0.1*l, ylim[0]), 1.1*ylim[1]])
        axs[0,1].set_title(f'loss %.5f, safety measure: %.5f'%(self.loss(self.trajectory), self.c(self.trajectory)))
        if store:
            fig.savefig(os.path.join(file_path, file_name))
        else:
            plt.show()
        return fig, None


    def _plot(self, store: bool, file_name: str = None, file_path: str = None):
        theta_bin = np.linspace(0, 2*np.pi, 100)
        l = self._constants[2] # length of pendulum

        fig, axs = plt.subplots(batch_size, 1, sharex='all', sharey='all')
        loc = self.trajectory[0,:,0]
        theta = self.trajectory[0,:,2]
        cart, pend_circle, pend = [], [], []
        for i in range(batch_size):
            cart.append(
                axs[i].plot([loc[i]], [0.0], "o", color="black", label='cart')[0]
            )
            pend_circle.append(
                axs[i].plot(loc[i] + l*np.sin(theta_bin), l*np.cos(theta_bin), '-', color='red')[0]
            )
            pend.append(
                axs[i].plot([loc[i], loc[i]+l*np.sin(theta[i])], [0, -l*np.cos(theta[i])], "-", color="black", label='pendulum')[0]
            )
            axs[i].hlines([0.0], [-20], [20], colors=['black'], linestyles=["dashed"])
            axs[i].vlines([0.0], [-2*l], [2*l], colors=['black'], linestyles=["dashed"])
            axs[i].set_xlim([-5, 5])
            axs[i].set_ylim([-1.1*l, 1.1*l])
            axs[i].legend()
        title = axs[0].set_title('t=0')
        def update(idx):
            t = self.trajectory_time_map[idx]
            state = self.trajectory[idx]
            loc = state[...,0]
            theta = state[...,2]
            for i in range(batch_size):                
                cart[i].set_xdata([loc[i]])
                cart[i].set_ydata([0.0])
                pend_circle[i].set_xdata(loc[i] + l*np.sin(theta_bin))
                pend_circle[i].set_ydata(l*np.cos(theta_bin))
                pend[i].set_xdata([loc[i], loc[i]+l*np.sin(theta[i])])
                pend[i].set_ydata([0, -l*np.cos(theta[i])])
            title.set_text('t=%0.3f'%t)
            return True
        ani = FuncAnimation(fig, update, [i for i in range(1, len(self.trajectory_time_map))], interval=1e3/record_freq, repeat_delay=1e3)
        if store:
            with PillowWriter(
                fps=record_freq,
                metadata=dict(artist='Me'),
                bitrate=1800
            ) as writer:
                ani.save(os.path.join(file_path, file_name+'.gif'), writer=writer)
        else:
            plt.show()
        return fig, ani

class CartPoleConstrained(StandardConstrainedOracle):
    def __init__(
        self,
        observation_noise: Union[float, List[float]],
        constants: np.ndarray = default_parameters,
        reduce_dimension: bool=False,
    ):
        """
        :param observation_noise: standard deviation of the observation noise (Gaussian noise)
            specify all functions by passing a float or specify each individually by passing a list of 2 floats
        :param constants: (m_c, m_p, l, f, freq)
            m_c: mass of cart (kg),
            m_p: mass of pendulum (kg),
            l: length of pendulum (m),
        )
        :param reduce_dimension: if False, our linear contoller is x*s + x4 (5 dim), otherwise x*s (4 dim)
        """
        if hasattr(observation_noise, '__len__'):
            if len(observation_noise) == 1:
                observation_noise = [observation_noise[0]] * 2
            elif len(observation_noise) == 2:
                assert np.isclose(*observation_noise)
            else:
                assert False, f'passing incorrect number of observation_noise to {self.__class__.__name__}.__init__ method'
        else:
            observation_noise = [observation_noise] * 2

        main = CartPole(observation_noise[0], constants, reduce_dimension=reduce_dimension)
        super().__init__(main, main)

    @property
    def reduce_dimension(self):
        return self.oracle.reduce_dimension

    def query(self, x, noisy: bool = True):
        assert self.oracle.run_episode(x, noisy=noisy)
        loss = self.oracle.loss(self.oracle.trajectory)
        c = self.oracle.c(self.oracle.trajectory)
        return loss, c[None]

    def batch_query(self, X, noisy: bool = True):
        Y = []
        Z = []
        for i in range(X.shape[-2]):
            y, z = self.query(X[..., i,:], noisy)
            Y.append(y)
            Z.append(z.reshape(-1, 1))
        return np.array(Y), np.vstack(Z)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(precision=2)
    np.random.seed(12345)
    oracle = CartPoleConstrained(0.01, reduce_dimension=True)
    #oracle = CartPoleConstrained(0.01, constants=oracle.oracle.sample_constants())

    N = 500
    T = 0.0
    X, Y, Z = oracle.get_random_data(N, noisy=True)
    mask = Z.reshape(-1) >= T # np.argsort( Z.reshape(-1) )[::-1]
    print( mask.mean() *100, '%')
    fig, axs = plt.subplots(3, 5)
    for i in range(5):
        if i==4 and oracle.reduce_dimension:
            break
        axs[0, i].hist(X[:,i], bins=100)
        axs[0, i].set_title(f'X_{i}')
    axs[1, 0].hist(Y, bins=100)
    axs[1, 1].hist(Z, bins=100)
    axs[1, 0].set_title('Y')
    axs[1, 1].set_title('Z')
    for i in range(5):
        if i==4 and oracle.reduce_dimension:
            break
        axs[2, i].hist(X[mask, i], bins=100)
        axs[2, i].set_title(f'X_{i}_safe')
    axs[1, 3].hist(Y[mask], bins=100)
    axs[1, 4].hist(Z[mask], bins=100)
    axs[1, 3].set_title('safe Y')
    axs[1, 4].set_title('safe Z')
    fig.suptitle(oracle.oracle._constants)
    plt.show()
    print(Y.mean(), Y.var())
    print(Z.mean(), Z.var())
    
    X_safe = X[mask]
    Y_safe = Y[mask]
    Z_safe = Z[mask]
    for i in range(min(X_safe.shape[0], 20)):
        print(f'controller pars: {oracle.oracle.x_scale(X_safe[i])}, loss: {Y_safe[i]}, safety_mesure: {Z_safe[i]}')
        y, z = oracle.query(X_safe[i])
        print(f'      query again, loss: {y}, safety_mesure: {z}')
        oracle.oracle.plot()
    
    """
    idx_loss = np.argsort(Y.reshape(-1))
    print('###'*10)
    for i in idx_loss[:10]:
        print('# x:', X[i])
        print( oracle.oracle.x_scale(X[i]) )
        print('y:', Y[i], 'z:', Z[i])
    idx_safe = np.argsort(Z.reshape(-1))
    print('###'*10)
    for i in idx_safe[-10:]:
        print('# x:', X[i])
        print( oracle.oracle.x_scale(X[i]) )
        print('y:', Y[i], 'z:', Z[i])
    """