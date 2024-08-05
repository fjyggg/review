import numpy as np
from scipy.io import loadmat
from scipy.optimize import Bounds
import scipy.optimize as spo
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec


class LearningQLF:
    def __init__(self, demonstration_set, alpha=0):
        '''
        :param demonstration_set: a dictionary with keys "x_set", "successive_x_set"
        :param alpha: L2 regularization constant
        '''
        self.x_set = demonstration_set['x_set']
        self.successive_x_set = demonstration_set['successive_x_set']
        self.d_x = np.shape(self.x_set)[1]
        self.demonstration_set_size = np.shape(self.x_set)[0]
        self.alpha = alpha

    def j_(self, x, successive_x, l):
        return self.V(l, successive_x) - self.V(l, x)

    def g(self, j_):
        if j_ < 0:
            return 0
        else:
            return j_

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
        return obj + self.alpha * l.dot(l) #正则化参数

    def learning(self, l0, learning_options, save_options=None, cons=None):
        '''
        :param l0: initial parameters
        :param learning_options: a dictionary with keys "max_iter", "disp", "ftol"
        :param save_options: a dictionary with keys "save_flag", "save_path"
        :param cons: constrained functions
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
        L = l.reshape(self.d_x, self.d_x)
        P = L.dot(L.T)  # + self.delta * np.eye(self.d_x)
        return x.dot(P.dot(x))

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
                           c='blue', alpha=1.0, s=0.5, marker='x')
                count = count + 1
            else:
                ax.scatter(self.successive_x_set[i, 0], self.successive_x_set[i, 1], self.successive_x_set[i, 2],
                           c='red', alpha=1.0, s=0.5, marker='o')
        print('the number of violated points are ', count)
        # plot_handle.scatter(x_set[:, 0], x_set[:, 1], c='red', alpha=1.0, s=mark_size)
        if save_options is not None:
            save_flag = save_options['save_flag']
            save_path = save_options['save_path']
            if save_flag is True:
                ax.savefig(save_path, dpi=300)
        plt.show()

    def dv_dx(self, l, x):
        L = l.reshape(self.d_x, self.d_x)
        P = L.dot(L.T)
        return 2 * P.dot(x).reshape(-1)






