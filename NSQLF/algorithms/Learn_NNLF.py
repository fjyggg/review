import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec


class LearnLyapunovFunction:
    def __init__(self, demonstration_set, nn_structure, max_norm=10, scale=0.05, epsilon=1e-10):
        '''
        Initialization info
        :param demonstration_set: a dictionary with keys "x_set", "dot_x_set" and "successive_x_set"
        :param nn_structure: structure info of the NN, an array [d_input=d_x, d_hidden>d_x, d_phi>d_hidden, q1>=1, q2>=1]
        :param max_norm: used to control the amplitude of the activation function tanh(x)
        :param scale: used to control the gradient of the activation function tanh(x)
        :param epsilon: used to ensure the PD of the W1 and W2
        '''
        self.demonstration_set = demonstration_set
        self.d_x = np.shape(demonstration_set['x_set'])[1]
        self.data_size = np.shape(demonstration_set['x_set'])[0]
        self.nn_structure = nn_structure
        self.max_norm = max_norm
        self.scale = scale
        self.epsilon = epsilon

    def forward_prop(self, para, x):
        '''
        :param para: para of the NN, array with the form (d_x*q1 + (d_hidden-d_x)*d_x + d_hidden*q2 + (d_phi-d_hidden)*d_hidden)
        :param x: the input of the RBFNN, an array of the form (d_x)
        :return: phi(x), an array of the form (d_phi>d_x)
        '''
        d_x, d_hidden, d_phi, q1, q2 = self.nn_structure
        epsilon = self.epsilon
        max_norm = self.max_norm
        scale = self.scale
        G11 = para[0: d_x * q1].reshape(q1, d_x)
        G12 = para[d_x * q1: (d_x * q1 + (d_hidden - d_x) * d_x)].reshape((d_hidden - d_x), d_x)
        W1 = np.vstack((G11.T.dot(G11) + epsilon * np.eye(d_x), G12))
        y1 = max_norm * np.tanh(scale * W1.dot(x))
        index = d_x * q1 + (d_hidden - d_x) * d_x
        G21 = para[index: (index + d_hidden * q2)].reshape(q2, d_hidden)
        G22 = para[(index + d_hidden * q2): (index + d_hidden * q2 +
                                             (d_phi-d_hidden) * d_hidden)].reshape(d_phi-d_hidden, d_hidden)
        W2 = np.vstack((G21.T.dot(G21) + epsilon * np.eye(d_hidden), G22))
        phi = max_norm * np.tanh(scale * W2.dot(y1))
        return phi

    def lyapunov_function(self, para, x):
        '''
        :param para: para of the NN, array with the form (d_x*q1 + (d_hidden-d_x)*d_x + d_hidden*q2 + (d_phi-d_hidden)*d_hidden)
        :param x: the input of the RBFNN, an array of the form (d_x)
        :return: Value of the Lyapunov function
        '''
        phi = self.forward_prop(para, x)
        return phi.dot(phi)

    def obj(self, para):
        '''
        :param para: para of the NN, array with the form (d_x*q1 + (d_hidden-d_x)*d_x + d_hidden*q2 + (d_phi-d_hidden)*d_hidden)
        :return: total cost function
        '''
        obj = 0
        x_set = self.demonstration_set['x_set']
        successive_x_set = self.demonstration_set['successive_x_set']
        for i in range(self.data_size):
            x = x_set[i]
            successive_x = successive_x_set[i]
            J = self.lyapunov_function(para, successive_x) - self.lyapunov_function(para, x)
            if J < 0:
                obj = obj + 0
            else:
                obj = obj + J
        return obj / self.data_size

    def learning(self, initial_para, learning_options, save_options=None):
        '''
        :param initial_para: initial para of the NN, array with the form (d_x*q1 + (d_hidden-d_x)*d_x + d_hidden*q2 + (d_phi-d_hidden)*d_hidden)
        :param learning_options: a dictionary with keys "max_iter", "disp", "ftol"
        :param save_options: a dictionary with keys "save_flag", "save_path"
        :return: trained para, array with the form (d_x*q1 + (d_hidden-d_x)*d_x + d_hidden*q2 + (d_phi-d_hidden)*d_hidden)
        '''
        max_iter = learning_options['max_iter']
        disp = learning_options['disp']
        ftol = learning_options['ftol']
        result = minimize(self.obj, initial_para, method='SLSQP', options={'disp': disp, 'maxiter': max_iter, 'ftol': ftol})
        trained_para = result.x
        if save_options is not None:
            save_flag = save_options['save_flag']
            if save_flag is True:
                save_path = save_options['save_path']
                np.savetxt(save_path, trained_para)
        return trained_para

    def show_learning_result(self, para, save_options=None):
        '''
        :param para: para of the NN, array with the form (d_x*q1 + (d_hidden-d_x)*d_x + d_hidden*q2 + (d_phi-d_hidden)*d_hidden)
        :param save_options: a dictionary with keys "save_flag", "save_path"
        '''
        # using for plot 2-D Lyapunov function
        mark_size = np.ones(self.data_size) * 10
        fig = plt.figure(figsize=(8, 8), dpi=100)
        gs = gridspec.GridSpec(4, 4)
        ax = fig.add_subplot(gs[0:4, 0:4], projection='3d')
        x_set = self.demonstration_set['x_set']
        successive_x_set = self.demonstration_set['successive_x_set']
        count = 0
        for i in range(self.data_size):
            if self.lyapunov_function(para, successive_x_set[i, :]) > self.lyapunov_function(para, x_set[i, :]):
                ax.scatter(successive_x_set[i, 0], successive_x_set[i, 1], successive_x_set[i, 2], c='blue', alpha=1.0,
                           s=mark_size, marker='x')
                count = count + 1
            else:
                ax.scatter(successive_x_set[i, 0], successive_x_set[i, 1], successive_x_set[i, 2], c='red', alpha=1.0,
                           s=mark_size, marker='o')
        print('the number of violated points are ', count)

        if save_options is not None:
            save_flag = save_options['save_flag']
            save_path = save_options['save_path']
            if save_flag is True:
                plt.savefig(save_path, dpi=300)
        plt.show()

    def dv_dx(self, para, x):
        '''
        :param para: para of the NN, array with the form (d_x*q1 + (d_hidden-d_x)*d_x + d_hidden*q2 + (d_phi-d_hidden)*d_hidden)
        :param x: the input of the NN, an array of the form (d_x)
        :return: the gradient of LF wst. the input
        '''
        d_x, d_hidden, d_phi, q1, q2 = self.nn_structure
        epsilon = self.epsilon
        max_norm = self.max_norm
        scale = self.scale
        G11 = para[0: d_x * q1].reshape(q1, d_x)
        G12 = para[d_x * q1: (d_x * q1 + (d_hidden - d_x) * d_x)].reshape((d_hidden - d_x), d_x)
        W1 = np.vstack((G11.T.dot(G11) + epsilon * np.eye(d_x), G12))
        y1 = max_norm * np.tanh(scale * W1.dot(x))
        index = d_x * q1 + (d_hidden - d_x) * d_x
        G21 = para[index: (index + d_hidden * q2)].reshape(q2, d_hidden)
        G22 = para[(index + d_hidden * q2): (index + d_hidden * q2 +
                                             (d_phi - d_hidden) * d_hidden)].reshape(d_phi - d_hidden, d_hidden)
        W2 = np.vstack((G21.T.dot(G21) + epsilon * np.eye(d_hidden), G22))
        y1_ = scale * W2.dot(y1)
        x_ = scale * W1.dot(x)
        dphi_dx = max_norm * max_norm * scale * scale * np.diag(1 - np.tanh(y1_) * np.tanh(y1_)).dot(W2).dot(np.diag(1 - np.tanh(x_) * np.tanh(x_))).dot(W1)
        phi = max_norm * np.tanh(scale * W2.dot(y1))
        dv_dphi = 2 * phi.reshape(1, -1)
        dv_dx = (dv_dphi.dot(dphi_dx)).reshape(-1)
        return dv_dx

    def plot_zero_gradient_area(self, para, area, epsilon):
        '''
        :param para: para of the NN, array with the form (d_x*q1 + (d_hidden-d_x)*d_x + d_hidden*q2 + (d_phi-d_hidden)*d_hidden)
        :param area: a dictionary with keys 'x_max', 'x_min', 'y_max', 'y_min' , 'z_max', 'z_min', and 'step'
        :param epsilon: used to control the size of zero_gradient areas
        :return: void
        '''
        x_ = np.arange(area['x_min'], area['x_max'], area['step'])
        y_ = np.arange(area['y_min'], area['y_max'], area['step'])
        z_ = np.arange(area['z_min'], area['z_max'], area['step'])
        x, y, z = np.meshgrid(x_, y_, z_)
        length_x_ = np.shape(x_)[0]
        length_y_ = np.shape(y_)[0]
        length_z_ = np.shape(z_)[0]
        # if the gradient is near zero, then the point is recorded
        print('recording near-zero-gradient points')
        zero_gradient_positions = []
        for i in range(length_y_):
            for j in range(length_x_):
                for k in range(length_z_):
                    position = np.array([x_[j], y_[i], z_[k]])
                    gradient = self.dv_dx(para, position)
                    print(np.sqrt(gradient.dot(gradient)))
                    if np.sqrt(gradient.dot(gradient)) <= epsilon:
                        zero_gradient_positions.append(position)
        zero_gradient_positions = np.array(zero_gradient_positions)
        fig = plt.figure(figsize=(6, 6), dpi=100)  # figsize=(6, 10), dpi=100
        plt.subplots_adjust(left=0.01, right=0.99, wspace=0.01, hspace=0.01, bottom=0.01, top=0.99)
        gs = gridspec.GridSpec(6, 6)
        ax = fig.add_subplot(gs[0:6, 0:6], projection='3d')
        # plot the near-zero-gradient points
        print('zero_gradient_positions are ', zero_gradient_positions)
        num_zero_gradient_positions = np.shape(zero_gradient_positions)[0]
        for i in range(num_zero_gradient_positions):
            ax.scatter(zero_gradient_positions[i, 0], zero_gradient_positions[i, 1], zero_gradient_positions[i, 2],
                       c='blue', alpha=0.5, s=2, marker='o')
        ax.scatter(0, 0, 0, c='black', alpha=1.0, s=20, marker='*')
        '''
        # record the gradients in zero-gradient areas
        print('recording gradients in near-zero-gradient areas')
        x_ = zero_gradient_positions[:, 0]
        y_ = zero_gradient_positions[:, 1]
        z_ = zero_gradient_positions[:, 2]
        x, y, z = np.meshgrid(x_, y_, z_)
        print(x, y, z)
        length_x_ = np.shape(x_)[0]
        length_y_ = np.shape(y_)[0]
        length_z_ = np.shape(z_)[0]
        u_lf = np.zeros((length_y_, length_x_, length_z_))
        v_lf = np.zeros((length_y_, length_x_, length_z_))
        omega_lf = np.zeros((length_y_, length_x_, length_z_))
        for i in range(length_y_):
            for j in range(length_x_):
                for k in range(length_z_):
                    position = np.array([x_[j], y_[i], z_[k]])
                    u_lf[i, j, k], v_lf[i, j, k], omega_lf[i, j, k] = -self.dv_dx(para, position)
        # plot the negative gradients in zero-gradient areas
        print('plotting gradients in near-zero-gradient areas')
        ax.quiver(x, y, z, u_lf, v_lf, omega_lf, length=0.006, normalize=True, color='red', alpha=0.8, linewidth=2)
        '''
        # plot the demonstration trajectories
        print('plotting demonstration trajectories')
        mark_size = np.ones(self.data_size) * 10
        successive_x_set = self.demonstration_set['successive_x_set']
        for i in range(self.data_size):
            ax.scatter(successive_x_set[i, 0], successive_x_set[i, 1], successive_x_set[i, 2], c='red', alpha=1.0,
                       s=mark_size, marker='o')

        ax.grid(color='grey', alpha=0)
        plt.show()

