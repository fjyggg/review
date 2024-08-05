'''
Please find the paper "Learning Control Lyapunov Function to Ensure S
tability of Dynamical System-based Robot Reaching Motions"
for more details
'''

import numpy as np
from autograd import grad, jacobian
from scipy.optimize import Bounds
from scipy.optimize import minimize
import matplotlib.pyplot as plt
np.random.seed(5)


class Learn_WSAQF:
    def __init__(self, demos, K, L2=1e-6):
        '''
        :param demos: Lasa set
        :param K: L in the paper
        '''
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

        self.data_size = np.shape(self.x_set)[0]
        self.overline_x = np.max(np.sqrt(np.sum(self.x_set ** 2, axis=1)))
        self.d_x = np.shape(self.x_set)[1]
        self.K = K
        self.parameter_length = int((1 + self.d_x) * self.d_x / 2 + ((1 + self.d_x) * self.d_x / 2 + self.d_x) * self.K)
        self.L2 = L2  # L2 regularization parameter
        self.alpha = 0.0  # weight for the quadratic term

    def construct_demonstration_set(self):
        '''
        Obtain position, velocity and acceleration sets from the Lasa set
        :return: position, velocity and acceleration sets
        '''
        x_set = self.demos[self.used_tras[0]].pos[:, self.start:self.end:self.gap]
        dot_x_set = self.demos[self.used_tras[0]].vel[:, self.start:self.end:self.gap]
        # ddot_x_set = self.demos[self.used_tras[0]].acc[:, self.start:self.end:self.gap]
        for i in self.used_tras[1:]:
            x_set = np.hstack((x_set, self.demos[i].pos[:, self.start:self.end:self.gap]))
            dot_x_set = np.hstack((dot_x_set, self.demos[i].vel[:, self.start:self.end:self.gap]))
            # ddot_x_set = np.hstack((ddot_x_set, self.demos[i].acc[:, self.start:self.end:self.gap]))
        return x_set.T, dot_x_set.T  # , ddot_x_set.T

    def construct_parameter_components(self, parameters):
        '''
        Return L_set (K + 1, d_x, d_x), mu_set (K, d_x)
        '''
        d_x = self.d_x
        K = self.K
        L_set_parameters = parameters[0: int((K + 1) * (1 + d_x) * d_x / 2)].reshape(K + 1, -1)
        mu_set = parameters[int((K + 1) * (1 + d_x) * d_x / 2):].reshape(K, d_x)
        L_set = np.zeros((K + 1, d_x, d_x))
        tri_indices = np.tril_indices(self.d_x, k=0)
        for k in range(K + 1):
            L_k = np.zeros((d_x, d_x))
            L_k[tri_indices] = L_set_parameters[k, :]
            L_set[k, :, :] = L_k

        return np.array(L_set), mu_set

    def v(self, parameters, x):
        K = self.K
        L_set, mu_set = self.construct_parameter_components(parameters)
        L_0 = L_set[0, :, :]
        P_0 = np.dot(L_0, L_0.T)
        v = np.dot(np.dot(x, P_0), x)
        for k in (np.arange(K) + 1):
            L_k = L_set[k, :, :]
            P_k = np.dot(L_k, L_k.T)
            mu_k = mu_set[k - 1, :]
            kappa_k = np.dot(np.dot(x, P_k), x - mu_k)
            beta_k = (kappa_k >= 0)
            v = v + beta_k * (kappa_k**2)
        return v + self.alpha * np.dot(x, x)

    def dvdx(self, parameters, x):
        K = self.K
        L_set, mu_set = self.construct_parameter_components(parameters)
        L_0 = L_set[0, :, :]
        P_0 = np.dot(L_0, L_0.T)
        dvdx = 2 * np.dot(P_0, x)
        for k in (np.arange(K) + 1):
            L_k = L_set[k, :, :]
            P_k = np.dot(L_k, L_k.T)
            mu_k = mu_set[k - 1, :]
            kappa_k = np.dot(np.dot(x, P_k), x - mu_k)
            beta_k = (kappa_k >= 0)
            dvdx = dvdx + 2 * beta_k * kappa_k * (2 * np.dot(P_k, x) - np.dot(P_k, mu_k))

        '''
        # dvdx2 = grad(self.v, 1)(parameters, x)
        # max_diff = np.max(np.sqrt((dvdx - dvdx2) ** 2))
        # print('max dvdx diff: ', max_diff)
        '''
        return dvdx + 2 * self.alpha * x

    def ddvdxdp(self, parameters, x):
        K = self.K
        d_x = self.d_x

        def dPxdl(L, x):
            d_x = self.d_x
            tril_indices = np.tril_indices(d_x, k=0)
            if d_x == 2:
                l1, l2, l3 = L[tril_indices]
                x1, x2 = x
                return np.array([[2 * l1 * x1 + l2 * x2, l1 * x2, 0.0],
                                 [l2 * x1, l1 * x1 + 2 * l2 * x2, 2 * l3 * x2]])
            elif d_x == 3:
                l1, l2, l3, l4, l5, l6 = L[tril_indices]
                x1, x2, x3 = x
                return np.array([[2 * l1 * x1 + l2 * x2 + l4 * x3, l1 * x2, 0.0, l1 * x3, 0.0, 0.0],
                                 [l2 * x1, l1 * x1 + 2 * l2 * x2 + l4 * x3, 2 * l3 * x2 + l5 * x3, l2 * x3, l3 * x3, 0.0],
                                 [l4 * x1, l4 * x2, l5 * x2, l1 * x1 + l2 * x2 + 2 * l4 * x3, l3 * x2 + 2 * l5 * x3, 2 * l6 * x3]])

        L_set, mu_set = self.construct_parameter_components(parameters)
        L_0 = L_set[0, :, :]
        d_l = int((1 + d_x) * d_x / 2)
        ddvdxdls = np.zeros((d_x, (K + 1) * d_l))
        dP0xdl0 = dPxdl(L_0, x)
        ddvdxdls[:, 0: d_l] = 2 * dP0xdl0
        ddvdxdmus = np.zeros((d_x, K * d_x))

        for k in (np.arange(K) + 1):
            L_k = L_set[k, :, :]
            P_k = np.dot(L_k, L_k.T)
            mu_k = mu_set[k - 1, :]
            kappa_k = np.dot(np.dot(P_k, x), x - mu_k)
            beta_k = (kappa_k >= 0)
            dPkxdlk = dPxdl(L_k, x)  # ((d_x, d_l))
            dPkmukdlk = dPxdl(L_k, mu_k)
            temp1 = 2 * np.dot((2 * np.dot(P_k, x) - np.dot(P_k, mu_k)).reshape(-1, 1), np.dot(dPkxdlk.T, x - mu_k).reshape(1, -1))
            temp2 = 2 * np.dot(np.dot(P_k, x - mu_k), x) * (2 * dPkxdlk - dPkmukdlk)
            ddvdxdlk = beta_k * (temp1 + temp2)
            ddvdxdls[:, (k * d_l): ((k + 1) * d_l)] = ddvdxdlk

            temp3 = 2 * np.dot((2 * np.dot(P_k, x) - np.dot(P_k, mu_k)).reshape(-1, 1), -np.dot(P_k, x).reshape(1, -1))
            temp4 = -2 * np.dot(np.dot(P_k, x - mu_k), x) * P_k
            ddvdxdmu_k = beta_k * (temp3 + temp4)
            ddvdxdmus[:, ((k - 1) * d_x): (k * d_x)] = ddvdxdmu_k

        ddvdxdp = np.hstack((ddvdxdls, ddvdxdmus))
        '''
        ddvdxdp2 = jacobian(self.dvdx, 0)(parameters, x)
        max_diff = np.max(np.sqrt((ddvdxdp - ddvdxdp2) ** 2))
        print('max ddvdxdp diff: ', max_diff)
        '''
        return ddvdxdp

    def obj_and_gradient_for_single_pair_points(self, parameters, x, dot_x, beta):
        '''
        obj function and its gradient for one-pair of data (x_k, dot_x_k)
        :param parameters: parameters of the Lyapunov function
        :param x: position x
        :param dot_x: velocity dot_x
        :param beta: determine the slope of the tanh activation function in objective function
        :return: obj function jk and its gradient djkdp
        '''
        dvdx = self.dvdx(parameters, x)
        dvdx_norm = np.sqrt(np.dot(dvdx, dvdx))
        dot_x_norm = np.sqrt(np.dot(dot_x, dot_x))
        if dvdx_norm == 0 or dot_x_norm == 0:
            # in this case, the pair of data points is omitted
            return None, None
        else:
            overline_j = np.dot(dvdx, dot_x) / np.sqrt(np.dot(dvdx, dvdx) * np.dot(dot_x, dot_x))
            djd_dvdx = (dot_x / dot_x_norm).reshape(1, -1).dot(
                dvdx_norm * np.eye(self.d_x) - dvdx.reshape(-1, 1).dot(dvdx.reshape(1, -1)) / dvdx_norm) / (
                                   dvdx_norm ** 2)
            ddvdxdp = self.ddvdxdp(parameters, x)

            overline_djdp = np.dot(djd_dvdx, ddvdxdp).reshape(-1)
            jk = np.tanh(beta * overline_j)
            djkdp = (1 - (np.tanh(beta * overline_j) ** 2)) * beta * overline_djdp

            return jk, djkdp

    def obj_and_gradient_for_data_set(self, parameters, beta):
        '''
        Compute the objective function on the training set,
        J = Sigma_k{j_k}, and its gradient with respect to parameters dJdp
        :param parameters: parameters of the Lyapunov function
        :param beta: determine the slope of the tanh activation function in objective function
        :return: (J, dJdp)
        '''
        J = 0
        dJdp = np.zeros(self.parameter_length)
        number_unsingular_points = 0
        for k in range(self.data_size):
            x, dot_x = self.x_set[k, :], self.dot_x_set[k, :]
            jk, djkdp = self.obj_and_gradient_for_single_pair_points(parameters, x, dot_x, beta)
            # print('k: ', k, 'jk: ', jk)
            if jk is not None:
                number_unsingular_points = number_unsingular_points + 1
                J = J + jk
                dJdp = dJdp + djkdp

        J = J / number_unsingular_points + self.L2 * np.sum(parameters**2)
        dJdp = dJdp / number_unsingular_points + 2 * self.L2 * parameters
        return J, dJdp

    def train(self, save_path=None, beta=10.0, maxiter=500):
        K = self.K
        d_x = self.d_x
        paras_lb = np.ones(self.parameter_length) * (-np.inf)
        paras_ub = np.ones(self.parameter_length) * np.inf
        d_l = int((1 + d_x) * d_x / 2)
        for k in range(K + 1):
            for i in (np.arange(d_x) + 1):
                paras_lb[k * d_l + int((1 + i) * i / 2) - 1] = 1e-8  # 1e-8
        bounds = Bounds(paras_lb, paras_ub)
        parameters = np.random.uniform(-0.5, 0.5, self.parameter_length)
        # parameters = np.random.normal(0.0, 5.0, self.parameter_length)
        res = minimize(self.obj_and_gradient_for_data_set, parameters, method='trust-constr', jac=True, args=(beta),
                       options={'disp': True, 'maxiter': maxiter, 'xtol': 1e-50, 'gtol': 1e-20}, bounds=bounds,
                       callback=self.callback)
        parameters = res.x
        if save_path is not None:
            np.savetxt(save_path, parameters)
        return parameters

    def callback(self, parameters, state):
        '''
        The call back function of the trust-constr method
        :param parameters: parameters of the Lyapunov function
        :param state: Optimization status,
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        :return: void
        '''
        if state.nit % 50 == 0 or state.nit == 1:
            obj_l2 = self.L2 * np.sum(parameters ** 2)
            obj_cost = state.fun - obj_l2

            print('---------------------------------- iter ', state.nit, '----------------------------------')
            print('running time: ', state.execution_time)
            print('obj_cost: ', obj_cost, 'L2_cost: ', obj_l2)
            print('maximum constr_violation: ', state.constr_violation)

    def show_learning_result(self, parameters, num_levels=10, plot_handle=None, scatter_flag=True):
        '''
        Plotting the Lyapunov function learning results in Cartesian or Polar space (to be completed)
        :param parameters: parameters of the Lyapunov function (stacked as a vector)
        :param num_levels: number of the contour levels to be plotted
        :param plot_handle: the external plot handle, if None, use the matplotlib.pyplot
        :param scatter_flag: whether to plot scatters
        :param gradient_flag: whether to plot stream of the negative gradient -dvdx
        :return: void
        '''
        if self.d_x == 2:
            x_1_min = np.min(self.x_set[:, 0]) - 5
            x_1_max = np.max(self.x_set[:, 0]) + 5
            x_2_min = np.min(self.x_set[:, 1]) - 5
            x_2_max = np.max(self.x_set[:, 1]) + 5
            step = 0.1
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
                    V[i, j] = self.v(parameters, x)
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
            mark_size = 2
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
        elif self.d_x == 3:
            x_1_min = np.min(self.x_set[:, 0])
            x_1_max = np.max(self.x_set[:, 0])
            x_2_min = np.min(self.x_set[:, 1])
            x_2_max = np.max(self.x_set[:, 1])
            x_3_min = np.min(self.x_set[:, 2])
            x_3_max = np.max(self.x_set[:, 2])
            # enlarge the area
            delta_x1 = x_1_max - x_1_min
            x_1_min = x_1_min - 0.1 * delta_x1
            x_1_max = x_1_max + 0.1 * delta_x1
            delta_x2 = x_2_max - x_2_min
            x_2_min = x_2_min - 0.1 * delta_x2
            x_2_max = x_2_max + 0.1 * delta_x2
            delta_x3 = x_3_max - x_3_min
            x_3_min = x_3_min - 0.1 * delta_x3
            x_3_max = x_3_max + 0.1 * delta_x3

            step = 0.01
            area_Cartesian = {'x_1_min': x_1_min, 'x_1_max': x_1_max, 'x_2_min': x_2_min, 'x_2_max': x_2_max, 'x_3_min': x_3_min, 'x_3_max': x_3_max, 'step': step}
            area = area_Cartesian
            step = area['step']
            x1 = np.arange(area['x_1_min'], area['x_1_max'], step)
            x2 = np.arange(area['x_2_min'], area['x_2_max'], step)
            x3 = np.arange(area['x_3_min'], area['x_3_max'], step)
            length_x1 = np.shape(x1)[0]
            length_x2 = np.shape(x2)[0]
            length_x3 = np.shape(x3)[0]
            V = np.zeros((length_x1, length_x2, length_x3))
            X = []
            for i in range(length_x1):
                for j in range(length_x2):
                    for k in range(length_x3):
                        x = np.array([x1[i], x2[j], x3[k]])
                        X.append(x)
                        V[i, j, k] = self.v(parameters, x)
            X = np.array(X)
            # plotting isosurfaces
            show_flag = False
            if plot_handle is None:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                plot_handle = fig.add_subplot(111, projection='3d')
                show_flag = True
            from skimage.measure import marching_cubes_lewiner
            number_isosurfaces = 5
            interested_tra_index = 4
            length_tra = np.shape(self.demos[self.used_tras[interested_tra_index]].pos)[1]
            gap = int(length_tra / number_isosurfaces)
            for i in range(number_isosurfaces):
                x_iso_expected = self.demos[self.used_tras[interested_tra_index]].pos[:, i * gap]
                diff = np.sum((X - x_iso_expected.reshape(1, -1))**2, axis=1)
                x_iso = X[np.argmin(diff), :]
                plot_handle.scatter3D(x_iso[0], x_iso[1], x_iso[2], c='blue', alpha=1.0, s=30, marker='D')
                v_iso = self.v(parameters, x_iso)
                verts, faces, _, _ = marching_cubes_lewiner(V, v_iso, spacing=(step, step, step))
                # compensate the offset
                verts = verts + np.array([x_1_min, x_2_min, x_3_min]).reshape(1, -1)
                plot_handle.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1, alpha=0.5)
            if scatter_flag is True:
                mark_size = 5
                plot_handle.scatter3D(0, 0, 0, c='black', alpha=1.0, s=50, marker='X')
                for tra_k in self.used_tras:
                    length_tra_k = np.shape(self.demos[tra_k].pos)[1]
                    for i in np.arange(self.start, length_tra_k + self.end, self.gap):
                        if i == self.start:
                            plot_handle.scatter3D(self.demos[tra_k].pos[0, i], self.demos[tra_k].pos[1, i], self.demos[tra_k].pos[2, i], c='black', alpha=1.0, s=15, marker='o')
                        elif self.v(parameters, self.demos[tra_k].pos[:, i + self.gap]) >= self.v(parameters, self.demos[tra_k].pos[:, i]):
                            plot_handle.scatter3D(self.demos[tra_k].pos[0, i], self.demos[tra_k].pos[1, i], self.demos[tra_k].pos[2, i], c='blue', alpha=1.0, s=mark_size, marker='x')
                        else:
                            plot_handle.scatter3D(self.demos[tra_k].pos[0, i], self.demos[tra_k].pos[1, i], self.demos[tra_k].pos[2, i], c='red', alpha=1.0, s=mark_size, marker='o')
            if show_flag is True:
                plt.show()
        return

    def plot_loss_trajectory(self, plot_handle=None, loss_tra=None):
        '''
        Plotting the loss trajectory
        :param plot_handle: the external plot handle, if None, use the matplotlib.pyplot
        :param loss_tra: the loss trajectory, if None, use the self.loss_tra
        :return: void
        '''
        show_flag = False
        if plot_handle is None:
            import matplotlib.pyplot as plt
            plot_handle = plt
            show_flag = True
        if loss_tra is None:
            loss_tra = np.array(self.loss_tra)
        x_axis = np.arange(np.shape(loss_tra)[0]) * 50
        # gap = int(np.shape(x_axis)[0] / 10)
        # print(np.shape(x_axis)[0])

        plot_handle.subplot(211)
        plot_handle.plot(x_axis, loss_tra[:, 0], c='blue')
        plot_handle.scatter(x_axis, loss_tra[:, 0], c='black', alpha=1.0, s=5, marker='o')
        plot_handle.subplot(212)
        plot_handle.plot(x_axis, loss_tra[:, 1], c='blue')
        plot_handle.scatter(x_axis, loss_tra[:, 1], c='black', alpha=1.0, s=5, marker='o')
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












