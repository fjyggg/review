'''
Please find the paper "Neural Learning of Stable
Dynamical Systems based on Data-Driven Lyapunov Candidates"
for more details
'''

import numpy as np
from scipy.optimize import minimize
from autograd import grad, jacobian
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS

np.random.seed(5)


class LearnNILC:
    def __init__(self, demos, d_H, A=None, b=None):
        self.demos = demos
        # Choosing used tras of the LASA set (or self-made set with same structure with the Lasa set)
        self.used_tras = [1, 2, 3, 4, 5, 6]  # 4, 5, 6 for Lasa set
        # Gap of two successive points
        self.gap = 20  # 20
        # Start and end points of a trajectory
        self.start = 1  # 20
        self.end = -1  # -25
        # Get position, velocity and acceleration sets from the Lasa set
        self.x_set, self.dot_x_set = self.construct_demonstration_set()
        self.overline_x = np.max(np.sqrt(np.sum(self.x_set ** 2, axis=1)))
        self.d_H = d_H
        self.d_x = np.shape(self.x_set)[1]
        self.parameter_length = self.d_H
        self.data_size = np.shape(self.x_set)[0]
        self.L2 = 1.0 * 1e-8
        # Record the training loss trajectory
        self.loss_tra = []
        # Setting feature parameters (A, b)
        self.A, self.b = self.set_feature_parameters(A, b)

    def construct_demonstration_set(self):
        '''
        Obtain position, velocity and acceleration sets from the Lasa set
        :return: position, velocity and acceleration sets
        '''
        x_set = self.demos[self.used_tras[0]].pos[:, self.start:self.end:self.gap]
        print(x_set)
        dot_x_set = self.demos[self.used_tras[0]].vel[:, self.start:self.end:self.gap]
        # ddot_x_set = self.demos[self.used_tras[0]].acc[:, self.start:self.end:self.gap]
        for i in self.used_tras[1:]:
            x_set = np.hstack((x_set, self.demos[i].pos[:, self.start:self.end:self.gap]))
            dot_x_set = np.hstack((dot_x_set, self.demos[i].vel[:, self.start:self.end:self.gap]))
            # ddot_x_set = np.hstack((ddot_x_set, self.demos[i].acc[:, self.start:self.end:self.gap]))
        return x_set.T, dot_x_set.T  # , ddot_x_set.T

    def set_feature_parameters(self, A=None, b=None):
        '''
        Randomly initialize
        '''
        if (A is None) and (b is None):
            size = self.d_x * self.d_H + self.d_H
            feature_parameters = np.random.normal(0.0, 0.1, size)
            A = feature_parameters[0: (self.d_x * self.d_H)].reshape(self.d_H, self.d_x)
            b = feature_parameters[(self.d_x * self.d_H):(self.d_x * self.d_H + self.d_H)]
        return A, b

    def v(self, parameters, x):
        '''
        Compute the Lyapunov function value given by the parameters and position x
        '''
        def forward_prop(parameters, x):
            A = self.A
            b = self.b
            w = parameters
            f = 1 / (1 + np.exp(-(np.dot(A, x) + b)))
            output = np.dot(w, f)
            return output

        v = forward_prop(parameters, x)
        return v

    def v_for_set(self, parameters, X_set):
        '''
        Compute the Lyapunov function values
        for a position set X_set
        '''
        def forward_prop(parameters, X_set):
            A = self.A
            b = self.b
            w = parameters
            f = 1 / (np.ones((np.shape(X_set)[0], 1)) + np.exp(-(np.dot(X_set, A.T) + b)))  # (N, d_H)
            output = np.dot(f, w)
            return output

        v = forward_prop(parameters, X_set)
        return v  # (N,)

    def dvdx(self, parameters, x):
        A = self.A
        b = self.b
        w = parameters
        f = 1 / (1 + np.exp(-(np.dot(A, x) + b)))
        temp1 = w * f * (1 - f)
        dvdx = np.dot(A.T, temp1)

        '''
        dvdx2 = grad(self.v, 1)(parameters, x)
        max_diff = np.max(np.sqrt((dvdx - dvdx2) ** 2))
        print('max dvdx diff: ', max_diff)
        '''
        return dvdx

    def dvdx_for_set(self, parameters, X_set):
        '''
        Compute the dvdx for a position set X_set, (N, d_x)
        '''
        A = self.A
        b = self.b
        w = parameters
        f = 1 / (np.ones((np.shape(X_set)[0], 1)) + np.exp(-(np.dot(X_set, A.T) + b)))  # (N, d_H)
        temp1 = w.reshape(1, self.d_H) * (f * (1 - f))  # (N, d_H)
        return np.dot(temp1, A)  # (N, d_x)

    def dvdp(self, parameters, x):
        '''
        Compute the dvdp given a position x, (d_H,)
        '''
        A = self.A
        b = self.b
        f = 1 / (1 + np.exp(-(np.dot(A, x) + b)))
        return f

    def dvdp_for_set(self, parameters, X_set):
        '''
        Compute the dvdp given a position set X_set, (N, d_H)
        '''
        A = self.A
        b = self.b
        f = 1 / (np.ones((np.shape(X_set)[0], 1)) + np.exp(-(np.dot(X_set, A.T) + b)))  # (N, d_H)
        return f

    def ddvdxdp(self, parameters, x):
        A = self.A
        b = self.b
        w = parameters
        f = 1 / (1 + np.exp(-(np.dot(A, x) + b)))
        dfdx = (f * (1 - f)).reshape(-1, 1) * A
        ddvdxdp = dfdx.T

        '''
        ddvdxdp2 = jacobian(self.dvdx, 0)(parameters, x)
        max_diff = np.max(np.sqrt((ddvdxdp - ddvdxdp2) ** 2))
        print('max ddvdxdp diff: ', max_diff)
        '''
        return ddvdxdp

    def ddvdxdp_for_set(self, parameters, X_set):
        A = self.A
        b = self.b
        w = parameters
        f = 1 / (np.ones((np.shape(X_set)[0], 1)) + np.exp(-(np.dot(X_set, A.T) + b)))  # (N, d_H)
        ddvdxdp = A.T * np.expand_dims(f * (1 - f), 1)
        return ddvdxdp  # (N, dx, d_H)

    '''
    def obj_and_gradient_for_single_pair_points(self, parameters, x, dot_x, beta=10.0):
        dvdx = self.dvdx(parameters, x)
        dvdx_norm = np.sqrt(np.dot(dvdx, dvdx))
        dot_x_norm = np.sqrt(np.dot(dot_x, dot_x))
        if dvdx_norm == 0 or dot_x_norm == 0:
            # in this case, the pair of data points is omitted
            return None, None
        else:
            overline_j = np.dot(dvdx, dot_x) / np.sqrt(np.dot(dvdx, dvdx) * np.dot(dot_x, dot_x))
            djd_dvdx = (dot_x / dot_x_norm).reshape(1, -1).dot(dvdx_norm * np.eye(self.d_x) - dvdx.reshape(-1, 1).dot(dvdx.reshape(1, -1)) / dvdx_norm) / (dvdx_norm ** 2)
            ddvdxdp = self.ddvdxdp(parameters, x)

            overline_djdp = np.dot(djd_dvdx, ddvdxdp).reshape(-1)
            jk = np.tanh(beta * overline_j)
            djkdp = (1 - (np.tanh(beta * overline_j) ** 2)) * beta * overline_djdp
            return jk, djkdp
    '''

    def obj_and_gradient_for_single_pair_points(self, parameters, x, dot_x):
        dvdx = self.dvdx(parameters, x)
        jk = np.dot(dvdx + dot_x, dvdx + dot_x)
        ddvdxdp = self.ddvdxdp(parameters, x)
        djkdp = 2 * np.dot(ddvdxdp.T, dvdx + dot_x)
        return jk, djkdp

    def obj_and_gradient_for_data_set(self, parameters):
        J = 0
        dJdp = np.zeros(self.parameter_length)
        number_unsingular_points = 0
        for k in range(self.data_size):
            x, dot_x = self.x_set[k, :], self.dot_x_set[k, :]
            jk, djkdp = self.obj_and_gradient_for_single_pair_points(parameters, x, dot_x)
            if jk is not None:
                number_unsingular_points = number_unsingular_points + 1
                J = J + jk
                dJdp = dJdp + djkdp
        J = J / number_unsingular_points + self.L2 * np.sum(parameters ** 2)
        dJdp = dJdp / number_unsingular_points + 2 * self.L2 * parameters
        return J, dJdp

    def train(self, save_paths=None):
        def singular_indexes(Set):
            norms = np.sqrt(np.sum(Set ** 2, axis=1))
            singular_index = np.where(norms == 0)[0]
            return singular_index

        parameters = np.random.uniform(-0.5, 0.5, self.parameter_length)
        p = 0.95
        N_c = int(1e5)
        x_min = np.zeros(self.d_x) - self.overline_x  #  np.min(self.x_set, axis=0)
        x_max = np.zeros(self.d_x) + self.overline_x   # np.max(self.x_set, axis=0)

        # print(x_min, x_max, (N_c, self.d_x))
        X_samples = np.random.uniform(x_min, x_max, (N_c, self.d_x))
        dvdx_samples = self.dvdx_for_set(parameters, X_samples)  # (N_c, d_x)
        singular_index1 = singular_indexes(X_samples)
        singular_index2 = singular_indexes(dvdx_samples)
        singular_index = np.hstack((singular_index1, singular_index2))
        X_samples = np.delete(X_samples, singular_index, axis=0)
        dvdx_samples = np.delete(dvdx_samples, singular_index, axis=0)
        v_samples = self.v_for_set(parameters, X_samples)
        # Here is a modification on the sampling algorithm proposed in the paper
        # The condition (iv) is normalized by using <dvdx, -x> / (||dvdx||_2 * ||x||_2)
        # Otherwise, the value dvdt could be influenced by the amplitude
        X_sample_norms = np.sqrt(np.sum(X_samples**2, axis=1))
        dvdx_sample_norms = np.sqrt(np.sum(dvdx_samples**2, axis=1))
        dot_v_samples = np.sum(dvdx_samples * (-X_samples), axis=1) / (X_sample_norms * dvdx_sample_norms)

        v_1 = np.sum(v_samples > 0)  # The numbers of samples satisfying condition ii
        v_2 = np.sum(dot_v_samples < 0)  # The numbers of samples satisfying condition iv
        min_v_index = np.argmin(v_samples)
        max_dot_v_index = np.argmax(dot_v_samples)
        U_1 = []
        U_2 = []
        i = 0
        # Maximum iterations
        max_iteration = 50
        while ((v_1 < N_c * p) or (v_2 < N_c * p)) and (i < max_iteration):
            if v_1 < (N_c * p):
                U_1.append(X_samples[min_v_index])
            if v_2 < (N_c * p):
                U_2.append(X_samples[max_dot_v_index])

            def cons_f(parameters):
                cons_f1 = self.v(parameters, np.zeros(self.d_x))
                cons_f2 = self.dvdx(parameters, np.zeros(self.d_x))
                cons_f = np.hstack((cons_f1, cons_f2))
                if len(U_1) > 0:
                    U_1_arrary = np.array(U_1).reshape(-1, self.d_x)
                    cons_f3 = self.v_for_set(parameters, U_1_arrary)
                    cons_f = np.hstack((cons_f, cons_f3))
                if len(U_2) > 0:
                    U_2_arrary = np.array(U_2).reshape(-1, self.d_x)
                    cons_f4 = np.sum(self.dvdx_for_set(parameters, U_2_arrary) * U_2_arrary, axis=1)
                    cons_f = np.hstack((cons_f, cons_f4))
                return cons_f

            def cons_jac(parameters):
                dc1dp = self.dvdp(parameters, np.zeros(self.d_x)).reshape(1, self.parameter_length)
                dc2dp = self.ddvdxdp(parameters, np.zeros(self.d_x)).reshape(self.d_x, self.parameter_length)
                dcdp = np.vstack((dc1dp, dc2dp))
                if len(U_1) > 0:
                    U_1_arrary = np.array(U_1).reshape(-1, self.d_x)
                    dc3dp = self.dvdp_for_set(parameters, U_1_arrary)
                    dcdp = np.vstack((dcdp, dc3dp))
                if len(U_2) > 0:
                    U_2_arrary = np.array(U_2).reshape(-1, self.d_x)
                    dc4dp = np.sum(np.expand_dims(U_2_arrary, 2) * self.ddvdxdp_for_set(parameters, U_2_arrary), axis=1)
                    dcdp = np.vstack((dcdp, dc4dp))
                '''
                dcdp2 = jacobian(cons_f, 0)(parameters)
                max_diff = np.max(np.sqrt((dcdp2 - dcdp) ** 2))
                print('max dcdp diff: ', max_diff)
                '''
                return dcdp

            def cons_hvp(parameters, v):
                return np.zeros((self.parameter_length, self.parameter_length))

            cons_low_bound = np.hstack((np.zeros(1 + self.d_x), np.zeros(len(U_1)) + 1e-8, np.zeros(len(U_2)) + 1e-8))
            cons_up_bound = np.hstack((np.zeros(1 + self.d_x), np.zeros(len(U_1)) + np.inf, np.zeros(len(U_2)) + np.inf))
            nonlinear_constraint = NonlinearConstraint(cons_f, cons_low_bound, cons_up_bound, jac=cons_jac, hess=cons_hvp)
            res = minimize(self.obj_and_gradient_for_data_set, parameters, method='trust-constr', jac=True,
                           options={'disp': False, 'maxiter': 5000, 'xtol': 1e-50, 'gtol': 1e-20},
                           constraints=[nonlinear_constraint])
            parameters = res.x

            X_samples = np.random.uniform(x_min, x_max, (N_c, self.d_x))
            dvdx_samples = self.dvdx_for_set(parameters, X_samples)  # (N_c, d_x)
            singular_index1 = singular_indexes(X_samples)
            singular_index2 = singular_indexes(dvdx_samples)
            singular_index = np.hstack((singular_index1, singular_index2))
            X_samples = np.delete(X_samples, singular_index, axis=0)
            dvdx_samples = np.delete(dvdx_samples, singular_index, axis=0)

            v_samples = self.v_for_set(parameters, X_samples)
            # Here is a modification on the sampling algorithm proposed in the paper
            # The condition (iv) is normalized by using <dvdx, -x> / (||dvdx||_2 * ||x||_2)
            # Otherwise, the value dvdt could be influenced by the amplitude
            X_sample_norms = np.sqrt(np.sum(X_samples ** 2, axis=1))
            dvdx_sample_norms = np.sqrt(np.sum(dvdx_samples ** 2, axis=1))
            dot_v_samples = np.sum(dvdx_samples * (-X_samples), axis=1) / (X_sample_norms * dvdx_sample_norms)

            v_1 = np.sum(v_samples > 0)  # The numbers of samples satisfying condition ii
            v_2 = np.sum(dot_v_samples < 0)  # The numbers of samples satisfying condition iv
            min_v_index = np.argmin(v_samples)
            max_dot_v_index = np.argmax(dot_v_samples)

            # Print info
            obj_l2 = self.L2 * np.sum(parameters ** 2)
            obj_cost = res.fun - obj_l2
            print('----- iteration', i, ' results -----')
            print('p1: ', v_1 * 1.0 / N_c)
            print('p2: ', v_2 * 1.0 / N_c)
            print('obj_cost: ', obj_cost, 'L2_cost: ', obj_l2)
            print('maximum constr_violation: ', res.constr_violation)
            print('U_1 size ', len(U_1))
            print('U_2 size ', len(U_2))

            i = i + 1
            # Plot
            # self.show_learning_result(parameters)

        if save_paths is not None:
            np.savetxt(save_paths[0], parameters)
            np.savetxt(save_paths[1], np.hstack((self.A.reshape(-1), self.b)))
        return parameters

    def show_learning_result(self, parameters, num_levels=10, plot_handle=None, scatter_flag=True):
        if self.d_x == 2:
            x_1_min = np.min(self.x_set[:, 0]) - 5
            x_1_max = np.max(self.x_set[:, 0]) + 5
            x_2_min = np.min(self.x_set[:, 1]) - 5
            x_2_max = np.max(self.x_set[:, 1]) + 5
            step = 0.5
            area_Cartesian = {'x_1_min': x_1_min, 'x_1_max': x_1_max, 'x_2_min': x_2_min, 'x_2_max': x_2_max, 'step': step}
            area = area_Cartesian
            step = area['step']
            x1 = np.arange(area['x_1_min'], area['x_1_max'], step)
            x2 = np.arange(area['x_2_min'], area['x_2_max'], step)
            length_x1 = np.shape(x1)[0]
            length_x2 = np.shape(x2)[0]
            X1, X2 = np.meshgrid(x1, x2)
            V = np.zeros((length_x2, length_x1))
            V_list = np.zeros(length_x2 * length_x1)

            for i in range(length_x2):
                for j in range(length_x1):
                    x = np.array([x1[j], x2[i]])
                    V[i, j] = self.v(parameters, x) - self.v(parameters, np.zeros(2))
                    V_list[i * length_x1 + j] = V[i, j]
            levels = np.sort(V_list, axis=0)[0::int(length_x2 * length_x1 / num_levels)]
            levels_ = []
            for i in range(np.shape(levels)[0]):
                if i is 0:
                    levels_.append(levels[i])
                else:
                    if levels[i] != levels[i - 1]:
                        levels_.append(levels[i])
            levels_ = np.array(levels_)
            # print('levels are ', levels_)
            # If we use external plot handle, the function will not
            # call show() function
            show_flag = False
            if plot_handle is None:
                import matplotlib.pyplot as plt
                plot_handle = plt
                show_flag = True
            contour = plot_handle.contour(X1, X2, V, levels=levels_, alpha=0.8, linewidths=1.0)
            # Show lf values on contours
            # plot_handle.clabel(contour, fontsize=8)
            mark_size = 5
            if scatter_flag is True:
                plot_handle.scatter(0, 0, c='black', alpha=1.0, s=50, marker='X')
                for tra_k in self.used_tras:
                    length_tra_k = np.shape(self.demos[tra_k].pos)[1]
                    for i in np.arange(self.start, length_tra_k + self.end - self.gap, self.gap):
                        if i == self.start:
                            plot_handle.scatter(self.demos[tra_k].pos[0, i], self.demos[tra_k].pos[1, i], c='black', alpha=1.0, s=15, marker='o')
                        elif self.v(parameters, self.demos[tra_k].pos[:, i + self.gap]) >= self.v(parameters, self.demos[tra_k].pos[:, i]):
                            plot_handle.scatter(self.demos[tra_k].pos[0, i], self.demos[tra_k].pos[1, i], c='blue', alpha=1.0, s=mark_size, marker='x')
                        else:
                            plot_handle.scatter(self.demos[tra_k].pos[0, i], self.demos[tra_k].pos[1, i], c='red', alpha=1.0, s=mark_size, marker='o')
            if show_flag is True:
                plot_handle.show()

    def compute_evaluations(self, parameters, beta=10.0):
        '''
        compute E_1, std(E_1), E_2, std(E_2)
        '''
        E_1_list = []
        E_2_list = []
        for i in self.used_tras:
            x_set = (self.demos[i].pos[:, self.start:self.end:self.gap]).T
            dot_x_set = (self.demos[i].vel[:, self.start:self.end:self.gap]).T
            lenghth_tra = np.shape(x_set)[0]
            E_1 = 0.0
            E_2 = 0.0
            for k in range(lenghth_tra):
                x, dot_x = x_set[k, :], dot_x_set[k, :]
                dvdx = self.dvdx(parameters, x)
                overline_j = np.dot(dvdx, dot_x) / np.sqrt(np.dot(dvdx, dvdx) * np.dot(dot_x, dot_x))
                j = np.tanh(beta * overline_j)
                E_1 = E_1 + j
                E_2 = E_2 + overline_j
            E_1_list.append(E_1 / lenghth_tra)
            E_2_list.append(E_2 / lenghth_tra)
        E_1_array, E_2_array = np.array(E_1_list), np.array(E_2_list)
        E_1 = 1 + np.average(E_1_array)
        std_E1 = np.std(E_1_array)
        E_2 = 1 + np.average(E_2_array)
        std_E2 = np.std(E_2_array)
        return E_1, std_E1, E_2, std_E2




