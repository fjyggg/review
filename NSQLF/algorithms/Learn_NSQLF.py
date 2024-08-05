import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix, spdiag, sqrt, div, exp, spmatrix, log, sparse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec


class Learn_LF():
    def __init__(self, demonstration_set, feature_parameters, P0=None, alpha=0.5, beta=2.0, overline_x=30):
        '''
        :param demonstration_set: a dictionary with keys "x_set", "dot_x_set" and "successive_x_set"
        :param feature_parameters: a dictionary with key "mu_set" and "sigma_set"
        :param P0: A based PD matrix, cvxopt form, shape:(d_x, d_x)
        :param alpha: the weight parameter
        '''
        self.x_set = demonstration_set['x_set']
        self.dot_x_set = demonstration_set['dot_x_set']
        self.successive_x_set = demonstration_set['successive_x_set']
        self.mu_set = feature_parameters['mu_set']
        self.sigma_set = feature_parameters['sigma_set']
        self.d_x = np.shape(self.x_set)[1]
        self.d_H = np.shape(self.mu_set)[0]
        self.demonstration_set_size = np.shape(self.x_set)[0]
        if P0 is None:
            self.P0 = spmatrix(1.0, range(self.d_x), range(self.d_x))
        else:
            self.P0 = P0
        self.alpha = alpha
        self.beta = beta
        self.overline_x = overline_x

    def feature_constructor_and_its_partial_derivatives(self, x):
        '''
        :param x: position, np.array form, shape:(d_x,)
        :return: (f, f_x), f: np.array form, shape:(d_H,); f_x: np.array form, shape:(d_H,d_x)
        '''
        f = np.zeros(self.d_H)
        f_pd = np.zeros((self.d_H, self.d_x))
        for i in range(self.d_H):
            temp = -0.5 * (x - self.mu_set[i]).reshape(1, -1).dot(np.diag(1 / np.power(self.sigma_set[i], 2))).dot(
                (x - self.mu_set[i]).reshape(-1, 1))
            f[i] = np.exp(temp.reshape(-1))
            f_pd[i, :] = f[i] * (np.diag(1 / np.power(self.sigma_set[i], 2)).dot(self.mu_set[i] - x)).reshape(-1)
        return f, f_pd

    def j_jpd_jpdpd(self, w, x, dot_x, successive_x):
        '''
        :param w: trained parameter, cvxopt.matrix form, shape:(d_x * d_H, 1)
        :param x: position, cvxopt.matrix form, shape:(d_x, 1)
        :param dot_x: corresponding velocity in demonstration_set, cvxopt.matrix form, shape:(d_x, 1)
        :param successive_x: corresponding successive position, cvxopt.matrix form, shape:(d_x, 1)
        :return: overline J(w), its derivative and Hessian for parameter w
        '''
        beta = self.beta
        W = matrix(w, (self.d_x, self.d_H))
        I = spmatrix(1.0, range(self.d_x), range(self.d_x))
        c1 = 2 * self.P0 * x
        c2 = c1 + dot_x
        f, f_pd = self.feature_constructor_and_its_partial_derivatives(np.array(x).reshape(-1))
        successive_f, _ = self.feature_constructor_and_its_partial_derivatives(np.array(successive_x).reshape(-1))
        f = matrix(f)
        successive_f = matrix(successive_f)
        f_pd = matrix(f_pd)
        j1 = matrix(0, (1, 1))
        jpd1 = matrix(0, (1, self.d_x * self.d_H))
        jpdpd1 = matrix(0, (self.d_x * self.d_H, self.d_x * self.d_H))
        j_2 = matrix(0, (1, 1))
        j_pd2 = matrix(0, (1, self.d_x * self.d_H))
        for i in range(self.d_x):
            x_i = x[i]
            successive_x_i = successive_x[i]
            e_i = I[:, i]
            temp_i = matrix(0, (self.d_H, self.d_x))
            for j in range(self.d_x):
                x_j = x[j]
                f_pd_i = f_pd[:, i]
                e_j = I[:, j]
                temp_i = temp_i + x_j * x_j * f_pd_i * e_j.T
            C_i1 = 2 * x_i * f * e_i.T + temp_i
            # since using the np.trace, hold j_ as the formulation as [scalar]
            val1 = matrix(matrix(0, (1, 1)) + np.trace(C_i1 * W) + c2[i], (1, 1))
            j1 = j1 + val1 * val1
            jpd1 = jpd1 + 2 * matrix(val1 * C_i1.T, (1, self.d_x * self.d_H))
            jpdpd1 = jpdpd1 + 2 * matrix(C_i1.T, (self.d_x * self.d_H, 1)) * matrix(C_i1.T, (1, self.d_x * self.d_H))
            C_i2 = successive_x_i ** 2 * successive_f * e_i.T - x_i ** 2 * f * e_i.T
            val2 = matrix(matrix(0, (1, 1)) + np.trace(C_i2 * W), (1, 1))
            j_2 = j_2 + val2
            j_pd2 = j_pd2 + matrix(C_i2.T, (1, self.d_x * self.d_H))

        v_norm_2 = dot_x.T * dot_x + 1e-20
        v_norm_2 = 1
        j_2 = j_2 + successive_x.T * self.P0 * successive_x - x.T * self.P0 * x
        '''
        print('j_2 is ', j_2)
        print('exp(-beta * j_2) is ', exp(-beta * j_2))
        '''
        if j_2[0] > 0:
            j2 = log(1 + exp(-beta * j_2)) + beta * j_2
            # j2 = -log(exp(-beta * j_2) / (exp(-beta * j_2) + 1) + 1e-50)  # -10
            jpd2 = beta / (exp(-beta * j_2) + 1) * j_pd2
            jpdpd2 = beta * beta * exp(-beta * j_2) / (exp(-beta * j_2) + 1) / (exp(-beta * j_2) + 1) * j_pd2.T * j_pd2
        else:
            j2 = log(1.0 + exp(beta * j_2))
            jpd2 = beta * exp(beta * j_2) / (exp(beta * j_2) + 1) * j_pd2
            jpdpd2 = beta * beta * exp(beta * j_2) / (exp(beta * j_2) + 1) / (exp(beta * j_2) + 1) * j_pd2.T * j_pd2
        j = sum(self.alpha * j1 / v_norm_2 + (1 - self.alpha) * j2)
        jpd = self.alpha * jpd1 / v_norm_2 + (1 - self.alpha) * jpd2
        jpdpd = self.alpha * jpdpd1 / v_norm_2 + (1 - self.alpha) * jpdpd2
        return j, jpd, jpdpd

    def F(self, w=None, z=None):
        '''
        parameter and return information can be found in
        https://cvxopt.org/userguide/solvers.html#problems-with-nonlinear-objectives
        '''
        if w is None:
            return 0, matrix(1e-2, (self.d_x * self.d_H, 1))
        J = 0
        Jpd = matrix(0.0, (1, self.d_x * self.d_H))
        Jpdpd = matrix(0.0, (self.d_x * self.d_H, self.d_x * self.d_H))
        for i in range(self.demonstration_set_size):
            x = matrix(self.x_set[i])
            dot_x = matrix(self.dot_x_set[i])
            successive_x = matrix(self.successive_x_set[i])
            j, jpd, jpdpd = self.j_jpd_jpdpd(w, x, dot_x, successive_x)
            J = J + j
            Jpd = Jpd + jpd
            Jpdpd = Jpdpd + jpdpd
        if z is None:
            return J / self.demonstration_set_size, Jpd / self.demonstration_set_size
        return J / self.demonstration_set_size, Jpd / self.demonstration_set_size, \
               z[0] * Jpdpd / self.demonstration_set_size

    def learning(self, training_options, save_options=None):
        '''
        :param training_options: a dictionary with keys "feastol", "abstol", "reltol", "maxiters" and "show_progress"
        :param save_options: a dictionary with keys "save_flag", "save_path"
        :return: parameter w, np.array form, shape:(d_x * d_H,)
        '''
        G1 = spmatrix(-1.0, range(self.d_H * self.d_x), range(self.d_H * self.d_x))
        G2 = spmatrix(1.0, range(self.d_H * self.d_x), range(self.d_H * self.d_x))
        G = sparse([G1, G2])
        h1 = matrix(-1e-10, (self.d_H * self.d_x, 1))
        h2 = matrix(1e5, (self.d_H * self.d_x, 1))
        h = matrix([h1, h2])
        solvers.options['feastol'] = training_options['feastol']
        solvers.options['abstol'] = training_options['abstol']
        solvers.options['reltol'] = training_options['reltol']
        solvers.options['maxiters'] = training_options['maxiters']
        solvers.options['show_progress'] = training_options['show_progress']
        w = solvers.cp(self.F, G=G, h=h)['x']
        if save_options is not None:
            if save_options['save_flag'] is True:
                np.savetxt(save_options['save_path'], np.array(w).reshape(-1))
        return np.array(w).reshape(-1)

    def V(self, w, x):
        '''
        :param w: parameter w, np.array form, shape:(d_x * d_H,)
        :param x: position, np.array form, shape:(d_x,)
        :return: LF value V(x), scalar
        '''
        W = np.array(matrix(w, (self.d_x, self.d_H)))
        feature, _ = self.feature_constructor_and_its_partial_derivatives(x)
        P = np.diag(W.dot(feature))
        V = x.reshape(1, -1).dot(P + self.P0).dot(x.reshape(-1, 1))
        return V[0, 0]

    def show_learning_result(self, w, save_options=None):
        '''
        :param w: parameter w, np.array form, shape:(d_x * d_H,)
        :param area: a dictionary with keys 'x_max', 'x_min', 'y_max', 'y_min' and 'step'
        :param save_options: a dictionary with keys "save_flag", "save_path"
        :return: void
        '''

        mark_size = np.ones(self.demonstration_set_size) * 10
        fig = plt.figure(figsize=(8, 8), dpi=100)
        gs = gridspec.GridSpec(4, 4)
        ax = fig.add_subplot(gs[0:4, 0:4], projection='3d')
        count = 0
        for i in range(self.demonstration_set_size):
            if self.V(w, self.successive_x_set[i, :]) > self.V(w, self.x_set[i, :]):
                ax.scatter(self.successive_x_set[i, 0], self.successive_x_set[i, 1],
                           self.successive_x_set[i, 2], c='blue', alpha=1.0, s=mark_size, marker='x')
                count = count + 1
            else:
                ax.scatter(self.successive_x_set[i, 0], self.successive_x_set[i, 1],
                           self.successive_x_set[i, 2], c='red', alpha=1.0, s=mark_size, marker='o')
        print('the number of violated points are ', count)
        if save_options is not None:
            save_flag = save_options['save_flag']
            save_path = save_options['save_path']

            if save_flag is True:
                plt.savefig(save_path, dpi=300)
        plt.show()

    def dv_dx(self, w, x):
        '''
        :param w: parameter w, np.array form, shape:(d_x * d_H,)
        :param x: position, np.array form, shape:(d_x,)
        :return: V_x, np.array form, shape(d_x,)
        '''
        w = matrix(w, (self.d_x * self.d_H, 1))
        x = matrix(x, (self.d_x, 1))
        W = matrix(w, (self.d_x, self.d_H))
        I = spmatrix(1.0, range(self.d_x), range(self.d_x))
        c2 = 2 * self.P0 * x
        f, f_pd = self.feature_constructor_and_its_partial_derivatives(np.array(x).reshape(-1))
        f = matrix(f)
        f_pd = matrix(f_pd)
        dv_dx = matrix(0.0, (1, self.d_x))
        for i in range(self.d_x):
            x_i = x[i]
            e_i = I[:, i]
            temp_i = matrix(0.0, (self.d_H, self.d_x))
            for j in range(self.d_x):
                x_j = x[j]
                f_pd_i = f_pd[:, i]
                e_j = I[:, j]
                temp_i = temp_i + x_j * x_j * f_pd_i * e_j.T
            C_i = 2 * x_i * f * e_i.T + temp_i
            # since using the np.trace, hold j_ as the formulation as [scalar]
            val = matrix(matrix(0, (1, 1)) + np.trace(C_i * W) + c2[i], (1, 1))
            dv_dx[0, i] = val
        return np.array(dv_dx).reshape(-1)
