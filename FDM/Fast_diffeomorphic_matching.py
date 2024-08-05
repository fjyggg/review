
'''
Please find the paper "Fast diffeomorphic matching to learn
globally asymptotically stable nonlinear dynamical systems"
for more details
'''
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from autograd import grad, jacobian
from scipy.optimize import fsolve
from scipy.optimize import NonlinearConstraint

np.random.seed(5)


class FastDiffMatching:
    def __init__(self, demos, K=150, beta=0.5, mu=0.9):
        '''
        :param demos: LASA set
        :param K: Max iterations:
        '''
        self.demos = demos
        self.used_tras = [1, 2, 3, 4, 5, 6]
        # Gap of two successive points
        self.gap = 20  # 20
        # Start and end points of a trajectory
        self.start = 1  # 20
        self.end = -1  # -25
        # Get average position, time-index sets from the Lasa set
        self.Y_set, self.t_set = self.construct_Y_set()   #得到数据的平均值
        self.d_x = np.shape(self.Y_set)[1]  #2
        self.X_set = self.constrcut_X_set(self.Y_set, self.t_set)  #这个是要微分同胚的方程
        # self.X_set1 = self.constrcut_X_set1(self.Y_set, self.t_set)  #这个是要微分同胚的方程

        self.K = K
        self.beta = beta
        self.mu = mu
        self.parameter_length = (self.d_x * 2 + 1) * (self.K + 1)  #755

    def construct_Y_set(self):
        '''
        Construct the target set from Lasa set, see paper "Fast diffeomorphic matching to learn globally asymptotically stable
        nonlinear dynamical systems"
        '''
        ave_Y_set = self.demos[self.used_tras[0]].pos[:, self.start:self.end:self.gap]  #2*1000
        ave_t_set = self.demos[self.used_tras[0]].t[:, self.start:self.end:self.gap]   #(1, 50)
        for i in self.used_tras[1:]:
            ave_Y_set = ave_Y_set + self.demos[i].pos[:, self.start:self.end:self.gap]
            ave_t_set = ave_t_set + self.demos[i].t[:, self.start:self.end:self.gap]
        return ave_Y_set.T / len(self.used_tras), ave_t_set.reshape(-1) / len(self.used_tras)

    def constrcut_X_set(self, ave_Y_set, ave_t_set):
        '''
        Construct the X set, see paper "Fast diffeomorphic matching to learn globally asymptotically stable
        nonlinear dynamical systems"
        '''
        t_0, t_N = ave_t_set[0], ave_t_set[-1]
        print(t_0)
        print(t_N)
        y_0 = ave_Y_set[0, :]
        print(ave_t_set)
        X_set = np.dot(((t_N - ave_t_set) / (t_N - t_0)).reshape(-1, 1), y_0.reshape(1, self.d_x))  #(50, 2)
        # print(X_set)
        return X_set  #创建直线的数据集
    # def constrcut_X_set1(self, ave_Y_set, ave_t_set):
    #     '''
    #     Construct the X set, see paper "Fast diffeomorphic matching to learn globally asymptotically stable
    #     nonlinear dynamical systems"
    #     '''
    #     t_0, t_N = ave_t_set[0], ave_t_set[-1]
    #     # print(ave_Y_set)
    #     y_1 = ave_Y_set[0, :]
    #     X_set1 = np.dot(((t_N - ave_t_set) / (t_N - t_0)).reshape(-1, 1), y_1.reshape(1, self.d_x))  #(50, 2)
    #     X_set1 = X_set1 + [5,5]
    #     # print(X_set)
    #     return X_set1  #创建直线的数据集

    def phi_trans(self, x, rho, c, v):
        '''
        the locally diffeomorphism mapping, see paper "Fast diffeomorphic matching to learn globally asymptotically stable nonlinear dynamical systems"
        '''
        k = np.exp(-(rho**2) * np.sum((x - c)**2))
        return x + k * v

    def dphidx(self, x, rho, c, v):
        '''
        Jacobian of phi
        '''
        k = np.exp(-(rho ** 2) * np.sum((x - c) ** 2))
        I = np.eye(self.d_x)
        dkdx = -2 * (rho ** 2) * k * np.dot(v.reshape(-1, 1), (x - c).reshape(1, -1))
        return I + dkdx

    def inv_phi_trans(self, y, rho, c, v):
        '''
        The inverse function of diffeomorphism transformation  #求逆变换的
        '''
        def fun(x, rho, c, v, y):
            return self.phi_trans(x, rho, c, v) - y

        x0 = np.ones(self.d_x) * 1.0
        x = fsolve(fun, x0, args=(rho, c, v, y))
        return x

    def phi_trans_forSet(self, X_set, rho, c, v):
        '''
        Set locally diffeomorphism transformations
        '''
        k = np.exp(-(rho**2) * np.sum((X_set.reshape(1,2) - c.reshape(1,2))**2,axis=1))  # (N,)
        # print('k is ', k)
        return X_set + np.dot(k.reshape(-1, 1), v.reshape(1, self.d_x))

    def train(self, save_path=None):
        '''
        Training parameters (rho_j, c_j, v_j), see pseudo-code in paper "
        Fast diffeomorphic matching to learn globally asymptotically stable
        nonlinear dynamical systems". "p_j" in the paper is replaced by "c_j"
        here for the clarity
        '''
        beta = self.beta
        mu = self.mu
        parameters = np.zeros((self.K + 1, 2 * self.d_x + 1))   # (151,5)

        def sub_train(c_j, v_j, Z_set):
            '''
            for rho-learning in one iteration
            '''
            def obj_grad(rho_j):
                '''
                obj and gradient
                '''

                Z_set_t = self.phi_trans_forSet(Z_set, rho_j, c_j, v_j)
                # the cost
                j = np.sum((Z_set_t - self.Y_set)**2) / np.shape(self.Y_set)[0]     #把损失函数拿出来了
                # compute the gradient
                temp1 = np.sum((Z_set - c_j) ** 2, axis=1)  # (N,)
                k = np.exp(-(rho_j ** 2) * temp1)  # (N,)
                temp2 = np.dot(k.reshape(-1, 1), v_j.reshape(1, -1))  # (N, d_x)
                dZtdrho = -2 * rho_j * temp1.reshape(-1, 1) * temp2  # (N, d_x)
                djdZt = 2 * (Z_set_t - self.Y_set)  # (N, d_x)
                djdrho = np.sum(np.diag(np.dot(djdZt, dZtdrho.T))) / np.shape(self.Y_set)[0]

                # print('djdrho - djdrho2: ', djdrho - djdrho2)
                return j, djdrho

            def cons_f(rho_j):  #这一步给出了RHO的约束
                rho_max = np.exp(0.5) / np.sqrt(2.0 * np.dot(v_j, v_j))
                cons_f1 = mu * rho_max - rho_j
                cons_f2 = rho_j
                cons_f = np.hstack([cons_f1, cons_f2])
                return cons_f

            def cons_jac(rho_j):
                return np.array([-1.0, 1.0]).reshape(2, 1)

            def cons_hvp(rho_j, v):
                return np.zeros((1, 1))

            nonlinear_constraint = NonlinearConstraint(cons_f, 1e-8, np.inf, jac=cons_jac, hess=cons_hvp)  #np.inf表示正无穷，jac和hess在这里没有任何的用处
            rho_j = 1.0
            res = minimize(obj_grad, rho_j, method='trust-constr', jac=True,
                           options={'disp': False, 'maxiter': 500, 'xtol': 1e-50, 'gtol': 1e-20},
                           constraints=[nonlinear_constraint], callback=self.callback)
            return res

        Z_set = self.X_set.copy()   #把x的数据集给z
        c_K = np.zeros(self.d_x)
        for j in range(self.K):
            print('in ', j, 'th iteration...')
            diff = np.sum(np.sqrt((Z_set - self.Y_set)**2), axis=1)
            index = np.argmax(diff)   #l论文中的m
            c_j, y_j = Z_set[index, :], self.Y_set[index, :]   #论文中的pj，q
            v_j = (y_j - c_j) * beta  #这是第十个式子
            res = sub_train(c_j, v_j, Z_set)
            print('maximum constr_violation is ', res.constr_violation)
            rho_j = res.x
            Z_set = self.phi_trans_forSet(Z_set, rho_j, c_j, v_j)
            parameters[j, :] = np.hstack((rho_j, c_j, v_j))
            # Pre-computation for c_K
            c_K = self.phi_trans(c_K, rho_j, c_j, v_j)  #就是x在0点是带入的值，可以看4.2
        v_K = np.zeros(self.d_x) - c_K
        res = sub_train(c_K, v_K, Z_set)
        rho_K = res.x
        parameters[self.K, :] = np.hstack((rho_K, c_K, v_K))
        if save_path is not None:
            np.savetxt(save_path, parameters)
        return parameters.reshape(-1)

    def Phi(self, x, parameters):
        '''
        The globally diffeomorphism transformer
        mapping x (abstract space) to y (Cartesian space)
        '''
        parameters = parameters.reshape(self.K + 1, -1)
        rho, c, v = parameters[0, 0:1], parameters[0, 1:(1 + self.d_x)], parameters[0, (1 + self.d_x):(1 + 2 * self.d_x)]
        phi = self.phi_trans(x, rho, c, v)
        for j in (np.arange(self.K) + 1):
            rho, c, v = parameters[j, 0:1], parameters[j, 1:(1 + self.d_x)], parameters[j, (1 + self.d_x):(1 + 2 * self.d_x)]
            phi = self.phi_trans(phi, rho, c, v)
        return phi

    def Phi_forSet(self, X_set, parameters):
        '''
        Set globally diffeomorphism transformer
        '''
        parameters = parameters.reshape(self.K + 1, -1)
        rho, c, v = parameters[0, 0:1], parameters[0, 1:(1 + self.d_x)], parameters[0, (1 + self.d_x):(1 + 2 * self.d_x)]
        Phi_X_set = self.phi_trans_forSet(X_set, rho, c, v)
        for j in (np.arange(self.K) + 1):
            rho, c, v = parameters[j, 0:1], parameters[j, 1:(1 + self.d_x)], parameters[j, (1 + self.d_x):(1 + 2 * self.d_x)]
            Phi_X_set = self.phi_trans_forSet(Phi_X_set, rho, c, v)
        return Phi_X_set

    def inv_Phi(self, y, parameters):
        '''
        Inverse function of Phi
        mapping y (Cartesian space) to x (abstract space)
        '''
        parameters = parameters.reshape(self.K + 1, -1)
        rho, c, v = parameters[self.K, 0:1], parameters[self.K, 1:(1 + self.d_x)], parameters[self.K, (1 + self.d_x):(1 + 2 * self.d_x)]
        inv_phi = self.inv_phi_trans(y, rho, c, v)
        for j in np.arange(self.K - 1, -1, -1):
            rho, c, v = parameters[j, 0:1], parameters[j, 1:(1 + self.d_x)], parameters[j, (1 + self.d_x):(1 + 2 * self.d_x)]
            inv_phi = self.inv_phi_trans(inv_phi, rho, c, v)
        return inv_phi

    def dPhidx(self, x, parameters):
        '''
        Jacobian of Phi
        '''
        dPhidx = np.eye(self.d_x)
        parameters = parameters.reshape(self.K + 1, -1)
        rho, c, v = parameters[0, 0:1], parameters[0, 1:(1 + self.d_x)], parameters[0, (1 + self.d_x):(1 + 2 * self.d_x)]
        phi = self.phi_trans(x, rho, c, v)
        dphidx = self.dphidx(x, rho, c, v)
        dPhidx = np.dot(dphidx, dPhidx)
        for j in (np.arange(self.K) + 1):
            rho, c, v = parameters[j, 0:1], parameters[j, 1:(1 + self.d_x)], parameters[j, (1 + self.d_x):(1 + 2 * self.d_x)]
            dphidx = self.dphidx(phi, rho, c, v)
            dPhidx = np.dot(dphidx, dPhidx) #是要把连成的关系式子拿出来
            phi = self.phi_trans(phi, rho, c, v)
        return dPhidx

    def callback(self, parameters, state):
        '''
        The call back function of the trust-constr method
        :param parameters: parameters of the Lyapunov function
        :param state: Optimization status,
        see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        :return: void
        '''
        pass
        '''
        if state.nit % 50 == 0 or state.nit == 1:
            print('---------------------------------- iter ', state.nit, '----------------------------------')
            print('running time: ', state.execution_time)
            print('obj_cost: ', state.fun)
            print('maximum constr_violation: ', state.constr_violation)
        '''
    def v(self, parameters, y):
        '''
        compute Lyapunov function in the Cartesian space
        '''
        x = self.inv_Phi(y, parameters)
        v = np.sqrt(np.dot(x, x))
        return v,x

    def dvdy(self, parameters, y):
        '''
        gradient of the Lyapunov function with respect to y  ，V是np.sqrt(np.dot(x, x))
        '''
        x = self.inv_Phi(y, parameters)
        dvdx = 1 / np.sqrt(np.dot(x, x)) * x
        dydx = self.dPhidx(x, parameters)
        dxdy = np.linalg.inv(dydx)
        dvdy = np.dot(dxdy.T, dvdx)
        # dydx_2 = jacobian(self.Phi, 0)(x, parameters)
        # print(np.max(np.abs(dydx - dydx_2)))
        return dvdy

    def plot_lf_learning_result(self, parameters, num_levels=10, plot_handle=None, scatter_flag=True):
        if self.d_x == 2:
            y_1_min = np.min(self.Y_set[:, 0]) - 5
            y_1_max = np.max(self.Y_set[:, 0]) + 5
            y_2_min = np.min(self.Y_set[:, 1]) - 5
            y_2_max = np.max(self.Y_set[:, 1]) + 5
            step = 0.5
            area_Cartesian = {'y_1_min': y_1_min, 'y_1_max': y_1_max, 'y_2_min': y_2_min, 'y_2_max': y_2_max, 'step': step}
            area = area_Cartesian
            step = area['step']
            y1 = np.arange(area['y_1_min'], area['y_1_max'], step)
            y2 = np.arange(area['y_2_min'], area['y_2_max'], step)
            length_y1 = np.shape(y1)[0]
            length_y2 = np.shape(y2)[0]
            Y1, Y2 = np.meshgrid(y1, y2)
            V = np.zeros((length_y2, length_y1))
            V_list = np.zeros(length_y2 * length_y1)
            xxx = np.zeros((1, 2))
            for i in range(length_y2):
                for j in range(length_y1):
                    y = np.array([y1[j], y2[i]])
                    V[i, j],_ = self.v(parameters, y)
                    V_list[i * length_y1 + j] = V[i, j]

            for i in range(self.Y_set.shape[0]):
                if i == 0:
                    _,xxx = self.v(parameters, self.Y_set[i,:])
                else:
                    _,xxx1 =self.v(parameters, self.Y_set[i,:])
                    xxx = np.vstack((xxx,xxx1))                   #他的逆变换是从我的数据曲线变到直线，但是有一个条件就是终点要归一化为0

            for i in range(self.X_set.shape[0]):
                if i == 0:
                    xxxX = self.Phi_forSet(self.X_set[i,:],parameters)
                else:
                    xxxX1 =self.Phi_forSet(self.X_set[i,:],parameters)
                    xxxX = np.vstack((xxxX,xxxX1))    #这里是直线变到我想要的曲线，对喽对喽

            levels = np.sort(V_list, axis=0)[0::int(length_y2 * length_y1 / num_levels)]  #num_levels是V的值的密度
            levels_ = []
            for i in range(np.shape(levels)[0]):
                if i is 0:
                    levels_.append(levels[i])
                else:
                    if levels[i] != levels[i - 1]:
                        levels_.append(levels[i])
            levels_ = np.array(levels_)  #把李雅普诺夫的值给区分开来
            # print('levels are ', levels_)
            # If we use external plot handle, the function will not
            # call show() function
            show_flag = False
            if plot_handle is None:
                import matplotlib.pyplot as plt
                plot_handle = plt
                show_flag = True
            contour = plot_handle.contour(Y1, Y2, V, levels=levels_, alpha=0.8, linewidths=1.0)
            # Show lf values on contours
            # plot_handle.clabel(contour, fontsize=8)
            mark_size = 5
            violated_numbers = 0

            if scatter_flag is True:
                plot_handle.scatter(0, 0, c='black', alpha=1.0, s=50, marker='X')

                # for tra_k in self.used_tras:
                #     length_tra_k = np.shape(self.demos[tra_k].pos)[1]
                #     for i in np.arange(self.start, length_tra_k + self.end - self.gap, self.gap):
                #         if i == self.start:
                #             plot_handle.scatter(self.demos[tra_k].pos[0, i], self.demos[tra_k].pos[1, i], c='black',
                #                                 alpha=1.0, s=15, marker='o')
                #         elif self.v(parameters, self.demos[tra_k].pos[:, i + self.gap]) >= self.v(parameters,
                #                                                                                   self.demos[tra_k].pos[
                #                                                                                   :, i]):
                #             plot_handle.scatter(self.demos[tra_k].pos[0, i], self.demos[tra_k].pos[1, i], c='blue',
                #                                 alpha=1.0, s=mark_size, marker='x')
                #             violated_numbers = violated_numbers + 1
                #         else:
                #             plot_handle.scatter(self.demos[tra_k].pos[0, i], self.demos[tra_k].pos[1, i], c='red',
                #                                 alpha=1.0, s=mark_size, marker='o')


                #测试正反的，对喽对喽
                plt.scatter(self.X_set[:, 0], self.X_set[:, 1],c='g',alpha=1.0,s=mark_size, marker='o')
                # plt.scatter(self.X_set1[:, 0], self.X_set1[:, 1],c='y',alpha=1.0,s=mark_size, marker='o')
                plt.scatter(xxx[:, 0], xxx[:, 1],c='red',alpha=1.0,s=mark_size, marker='o')
                plt.scatter(xxxX[:, 0], xxxX[:, 1],c='g',alpha=1.0,s=mark_size, marker='o')


                for i in range(np.shape(self.Y_set)[0] - 1):
                    if self.v(parameters, self.Y_set[i + 1, :]) >= self.v(parameters, self.Y_set[i, :]):
                        plot_handle.scatter(self.Y_set[i, 0], self.Y_set[i, 1], c='g', alpha=1.0, s=mark_size, marker='s')
                    else:
                        plot_handle.scatter(self.Y_set[i, 0], self.Y_set[i, 1], c='red', alpha=1.0, s=mark_size, marker='s')

                '''
                Z_set = self.Phi_forSet(self.X_set, parameters)
                for i in range(np.shape(Z_set)[0]):
                    plot_handle.scatter(Z_set[i, 0], Z_set[i, 1], c='black', alpha=1.0, s=mark_size, marker='^')

                for i in range(np.shape(self.X_set)[0]):
                    plot_handle.scatter(self.X_set[i, 0], self.X_set[i, 1], c='red', alpha=1.0, s=mark_size,
                                        marker='o')
                    x_hat = self.inv_Phi(Z_set[i, :], parameters)
                    plot_handle.scatter(x_hat[0], x_hat[1], c='blue', alpha=1.0, s=mark_size, marker='o')
                '''

            # print('violated_numbers is ', violated_numbers)
            if show_flag is True:
                plot_handle.show()
            return violated_numbers

    def compute_evaluations(self, parameters, beta=10.0):
        '''
        compute E_1, std(E_1), E_2, std(E_2)
        '''
        E_1_list = []
        E_2_list = []
        for i in self.used_tras:
            y_set = (self.demos[i].pos[:, self.start:self.end:self.gap]).T
            dot_y_set = (self.demos[i].vel[:, self.start:self.end:self.gap]).T
            lenghth_tra = np.shape(y_set)[0]
            E_1 = 0.0
            E_2 = 0.0
            for k in range(lenghth_tra):
                y, dot_y = y_set[k, :], dot_y_set[k, :]
                dvdy = self.dvdy(parameters, y)
                overline_j = np.dot(dvdy, dot_y) / np.sqrt(np.dot(dvdy, dvdy) * np.dot(dot_y, dot_y))
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