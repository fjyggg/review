'''
Formulating the stable ADS.
Details can be found in the paper:
"Learning a flexible neural energy function with a unique minimum
for stable and accurate demonstration learning"
'''

import numpy as np
from cvxopt import solvers, matrix
np.random.seed(5)


class LearnSds:
    def __init__(self, lf_learner, ods_learner):
        '''
        Initializing the Stable ADS
        '''
        self.lf_learner = lf_learner
        self.ods_learner = ods_learner
        self.d_x = lf_learner.d_x

    def predict(self, energy_function_parameter, x, func_rho, P=None, r_thres=0.0, eta=0.1):
        '''
        Prediction of the Stable ADS.
        Solving the Optimization problem described by (46) and (47)
        in the paper
        '''
        d_x = np.shape(x)[0]
        if P is None:
            P = np.eye(d_x)
        x_P_norm = np.sqrt(np.dot(np.dot(P, x), x))
        if x_P_norm == 0.0:
            dot_x, u = np.zeros(self.lf_learner.d_x), np.zeros(self.lf_learner.d_x)
            return dot_x, u
        elif ((x_P_norm**2) > (r_thres**2) * ((1 - eta)**2)) and ((x_P_norm**2) < (r_thres**2) * ((1 + eta)**2)):
            # print('in QP case')
            P_ = matrix(np.eye(d_x))
            q = matrix(np.zeros(d_x))
            dvdx = self.lf_learner.dvdx(energy_function_parameter, x)
            G = matrix(np.array([dvdx, np.dot(P, x)]))
            ods_dot_x = self.ods_learner.predict(x.reshape(1, 2)).reshape(-1)
            rho = func_rho(x)
            temp = np.dot(ods_dot_x, dvdx) + rho
            h0 = -temp
            h1 = -np.dot(np.dot(P, ods_dot_x), x)
            h = matrix([h0, h1])
            solvers.options['show_progress'] = False
            solution = solvers.qp(P_, q, G, h)
            u = np.array(solution['x']).reshape(-1)
            dot_x = ods_dot_x + u
            return dot_x, u
        else:
            dvdx = self.lf_learner.dvdx(energy_function_parameter, x)
            dvdx_norm_2 = np.dot(dvdx, dvdx)
            ods_dot_x = self.ods_learner.predict(x.reshape(1, 2)).reshape(-1)
            rho = func_rho(x)
            temp = np.dot(ods_dot_x, dvdx) + rho
            if temp > 0:
                u = -temp / dvdx_norm_2 * dvdx
                dot_x = ods_dot_x + u
            else:
                u = np.zeros(self.lf_learner.d_x)
                dot_x = ods_dot_x
            return dot_x, u

    def func_rho(self, x):
        '''
        A default function rho(x), see Eq. (57)
        in the paper
        '''
        gamma = 10.0
        x_norm_2 = np.dot(x, x)
        beta = (self.lf_learner.overline_x ** 2)
        max_v_in_set = np.max(np.sqrt(np.sum(self.lf_learner.dot_x_set ** 2, axis=1)))
        scale = max_v_in_set / gamma
        return scale * (1 - np.exp(-0.5 * x_norm_2 / beta))

    def show_learning_result(self, lf_parameter, func_rho=None, plot_handle=None, x0s=None, P=None, r_thres=0.0, eta=0.1, stream_flag=True, energy_flag=False, area_Cartesian=None):
        if func_rho is None:
            func_rho = self.func_rho

        if area_Cartesian is None:
            x_1_min = np.min(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 0])
            x_1_max = np.max(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 0])
            x_2_min = np.min(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 1])
            x_2_max = np.max(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 1])

            delta_x1 = x_1_max - x_1_min
            x_1_min = x_1_min - 0.2 * delta_x1
            x_1_max = x_1_max + 0.2 * delta_x1
            delta_x2 = x_2_max - x_2_min
            x_2_min = x_2_min - 0.2 * delta_x2
            x_2_max = x_2_max + 0.2 * delta_x2

            num = 80
            step = np.min(np.array([(x_1_max - x_1_min) / num, (x_2_max - x_2_min) / num]))
            area_Cartesian = {'x_1_min': x_1_min, 'x_1_max': x_1_max, 'x_2_min': x_2_min, 'x_2_max': x_2_max, 'step': step}

        area = area_Cartesian
        step = area['step']
        x1 = np.arange(area['x_1_min'], area['x_1_max'], step)
        x2 = np.arange(area['x_2_min'], area['x_2_max'], step)
        length_x1 = np.shape(x1)[0]
        length_x2 = np.shape(x2)[0]
        X1, X2 = np.meshgrid(x1, x2)
        Dot_x1 = np.zeros((length_x2, length_x1))
        Dot_x2 = np.zeros((length_x2, length_x1))
        # Color = np.zeros((length_y, length_x))
        if stream_flag is True:
            for i in range(length_x2):
                for j in range(length_x1):
                    x = np.array([x1[j], x2[i]])
                    desired_v, u = self.predict(lf_parameter, x, func_rho, P=P, r_thres=r_thres, eta=eta)
                    Dot_x1[i, j], Dot_x2[i, j] = desired_v
        show_flag = False
        if plot_handle is None:
            import matplotlib.pyplot as plt
            plot_handle = plt
            show_flag = True
        if stream_flag is True:
            plot_handle.streamplot(X1, X2, Dot_x1, Dot_x2, density=1.0, linewidth=0.3, maxlength=1.0, minlength=0.1,
                                   arrowstyle='simple', arrowsize=0.5)
        if energy_flag is True:
            self.lf_learner.show_learning_result(lf_parameter, num_levels=10, plot_handle=plot_handle, scatter_flag=False, area_Cartesian=area_Cartesian)

        mark_size = 2
        plot_handle.scatter(0, 0, c='black', alpha=1.0, s=50, marker='X')
        plot_handle.scatter(self.lf_learner.x_set.reshape(-1, self.d_x)[:, 0], self.lf_learner.x_set.reshape(-1, self.d_x)[:, 1], c='red', alpha=1.0, s=mark_size, marker='o')
        n_tra = np.shape(self.lf_learner.x_set)[0]
        for i in range(n_tra):
            plot_handle.scatter(self.lf_learner.x_set[i, 0, 0], self.lf_learner.x_set[i, 0, 1], c='black', alpha=1.0, s=15,
                                marker='o')

        x0s = self.lf_learner.x_set[:, 0, :]
        for i in range(len(x0s)):
            self.plot_repro(lf_parameter, x0s[i, :], func_rho=func_rho, plot_handle=plot_handle, P=P, r_thres=r_thres, eta=eta)
        '''
        if x0s is not None:
            for i in range(len(x0s)):
                self.plot_repro(lf_parameter, x0s[i, :], func_rho=func_rho, plot_handle=plot_handle, P=P, r_thres=r_thres, eta=eta)
        '''
        if show_flag is True:
            plot_handle.show()

    def plot_repro(self, lf_parameter, x0, func_rho, plot_handle=None, P=None, r_thres=0.0, eta=0.1):
        x = x0
        period = 1e-2
        steps = int(40 / period)
        x_tra = [x]
        for i in range(steps):
            desired_v, u = self.predict(lf_parameter, x, func_rho, P=P, r_thres=r_thres, eta=eta)
            x = x + desired_v * period
            x_tra.append(x)
        show_flag = False
        if plot_handle is None:
            import matplotlib.pyplot as plt
            plot_handle = plt
            show_flag = True
        x_tra = np.array(x_tra)
        plot_handle.plot(x_tra[:, 0], x_tra[:, 1], c='black', linewidth=2, alpha=1.0)
        if show_flag is True:
            plot_handle.show()

