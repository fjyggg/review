import numpy as np
from scipy.io import loadmat
from scipy.optimize import Bounds
import scipy.optimize as spo
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec


def call_back_function(w, result):
    # global iter
    # iter = iter + 1
    print('iter number is', result.nit)
    # print('The problem is  successfully solved ?', result.success)
    # print('termination message is', result.message)
    print('value is', result.fun)
    # print('constraint violation is', result.maxcv)
    print('-------------------------------')


class LearningSoSLf:
    def __init__(self, demonstration_set, M=2):
        '''
        :param demonstration_set: a dictionary with keys "x_set", "successive_x_set"
        :param delta: a constant to ensure the positive definite
        :param alpha: L2 regularization constant
        '''
        self.x_set = demonstration_set['x_set']
        self.successive_x_set = demonstration_set['successive_x_set']
        self.d_x = np.shape(self.x_set)[1]
        self.demonstration_set_size = np.shape(self.x_set)[0]
        self.M = M
        self.Mn = int(np.math.factorial(M + self.d_x) / np.math.factorial(self.d_x) / np.math.factorial(M) - 1)

    def j_(self, x, successive_x, l):
        return self.V(l, successive_x) - self.V(l, x)

    def g(self, j_):
        if j_ < 0:
            return 0
        else:
            return j_
        # return -np.log(np.exp(-j_) / (1 + np.exp(-j_)) + 1e-20)

    def j(self, x, successive_x, l):
        j_ = self.j_(x, successive_x, l)
        j = self.g(j_)
        return j

    def obj(self, l):
        obj = 0
        for i in range(self.demonstration_set_size):
            x = self.x_set[i, :]
            successive_x = self.successive_x_set[i, :]
            obj = obj + self.j(x, successive_x, l)
        obj = obj / self.demonstration_set_size
        return obj / 10

    def learning(self, l0, learning_options, cons=None, save_options=None):
        '''
        :param l0: initial parameter
        :param learning_options: a dictionary with keys "max_iter", "disp", "ftol"
        :param save_options: a dictionary with keys "save_flag", "save_path"
        :return:
        '''
        max_iter = learning_options['max_iter']
        disp = learning_options['disp']
        ftol = learning_options['ftol']
        result = minimize(self.obj, l0, method='SLSQP', options={'disp': disp, 'maxiter': max_iter, 'ftol': ftol},
                          constraints=cons)
        l = result.x
        if save_options is not None:
            save_flag = save_options['save_flag']
            save_path = save_options['save_path']
            if save_flag is True:
                np.savetxt(save_path, l)
        return l

    def V(self, l, x):
        L = l.reshape(self.Mn, self.Mn)
        P = L.dot(L.T)
        if self.d_x == 2:
            x1, x2 = x
            mx = np.array([x1, x1 * x2, x1 * x1, x2, x2 * x2])
        elif self.d_x == 3:
            x1, x2, x3 = x
            mx = np.array([x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 * x1, x2 * x2, x3 * x3])
        return mx.dot(P.dot(mx))

    def show_learning_result(self, w, save_options=None):
        '''
        :param w: parameter w, np.array form, shape:(d_x * d_H,)
        :param area: a dictionary with keys 'x_max', 'x_min', 'y_max', 'y_min' and 'step'
        :param save_options: a dictionary with keys "save_flag", "save_path"
        :return: void
        '''
        count = 0
        mark_size = np.ones(self.demonstration_set_size) * 10
        fig = plt.figure(figsize=(8, 8), dpi=100)
        gs = gridspec.GridSpec(4, 4)
        ax = fig.add_subplot(gs[0:4, 0:4], projection='3d')
        for i in range(self.demonstration_set_size):
            if self.V(w, self.successive_x_set[i, :]) > self.V(w, self.x_set[i, :]):
                ax.scatter(self.successive_x_set[i, 0], self.successive_x_set[i, 1], self.successive_x_set[i, 2],
                           c='blue', alpha=1.0, s=mark_size, marker='x')
                count = count + 1
            else:
                ax.scatter(self.successive_x_set[i, 0], self.successive_x_set[i, 1], self.successive_x_set[i, 2],
                           c='red', alpha=1.0, s=mark_size, marker='o')
        print('the number of violated points are ', count)
        # plot_handle.scatter(x_set[:, 0], x_set[:, 1], c='red', alpha=1.0, s=mark_size)
        if save_options is not None:
            save_flag = save_options['save_flag']
            save_path = save_options['save_path']
            if save_flag is True:
                ax.savefig(save_path, dpi=300)
        plt.show()

        print('the number of violated points are ', count)
        # plot_handle.scatter(x_set[:, 0], x_set[:, 1], c='red', alpha=1.0, s=mark_size)
        if save_options is not None:
            save_flag = save_options['save_flag']
            save_path = save_options['save_path']
            if save_flag is True:
                plt.savefig(save_path, dpi=400)
        plt.show()

    def dv_dx(self, l, x):
        L = l.reshape(self.Mn, self.Mn)
        P = L.dot(L.T)
        if self.d_x == 2:
            x1, x2 = x
            mx = np.array([x1, x1 * x2, x1 * x1, x2, x2 * x2])
            dmdx_T = np.array([[1, x2, 2 * x1, 0, 0], [0, x1, 0, 1, 2 * x2]])
        elif self.d_x == 3:
            x1, x2, x3 = x
            mx = np.array([x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 * x1, x2 * x2, x3 * x3])
            dmdx_T = np.array([[1, 0, 0, x2, x3, 0, 2 * x1, 0, 0], [0, 1, 0, x1, 0, x3, 0, 2 * x2, 0],
                               [0, 0, 1, 0, x1, x2, 0, 0, 2 * x3]])
        return 2 * (dmdx_T.dot(P.T).dot(mx.reshape(-1, 1))).reshape(-1)

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
        mark_size = np.ones(self.demonstration_set_size) * 10
        successive_x_set = self.successive_x_set
        for i in range(self.demonstration_set_size):
            ax.scatter(successive_x_set[i, 0], successive_x_set[i, 1], successive_x_set[i, 2], c='red', alpha=1.0,
                       s=mark_size, marker='o')

        ax.grid(color='grey', alpha=0)
        plt.show()