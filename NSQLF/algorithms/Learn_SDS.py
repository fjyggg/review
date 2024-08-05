import time
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix, spdiag, sqrt, div, exp, spmatrix, log
from matplotlib.colors import LinearSegmentedColormap


# The initial point is in the RoA
#  Thus fun. is used when the activation condition is satisfied, i.e., overline_x - \epsilon <= ||x||_2 <= overline_x
def controller(x, dv_dx, o_x, rho_x):
    '''
    :param x: position, np.array form, shape:(d_x,)
    :param dv_dx: gradient of V with respect to x, np.array form, shape:(d_x,)
    :param o_x: output of the ODS, np.array form, shape:(d_x,)
    :param rho_x: output of the positive function rho(x), a scalar
    :return: additional input, np.array form, shape:(d_x,)
    '''
    d_x = np.shape(x)[0]
    P = spmatrix(1.0, range(d_x), range(d_x))
    q = matrix(np.zeros(d_x))
    G = matrix(np.array([dv_dx, x]))
    h = matrix([-rho_x - dv_dx.dot(o_x), -x.dot(o_x)])
    sol = solvers.qp(P, q, G, h)
    return np.array(sol['x'])


class LearnSds:
    def __init__(self, lf, ods):
        '''
        :param lf: lf_learner object
        :param ods: ods_learner object
        '''
        self.lf = lf
        self.ods = ods

    def controller(self, x, lf_para, epsilon=1e-2, alpha=1e-5):
        '''
        :param x: position, np.array form, shape:(d_x,)
        :param lf_para: parameters of the lf_learner object, an array with the form ((d_x+1)*d_H + d_H * d_x)
        :param epsilon: parameter to relax the activation condition, a scalar
        :param alpha: parameter to control the use of PDF rho(x), a scalar
        :return: dot_x or False, the latter implies that the x is beyond the RoA, the robot should be stopped for safety
         concern, or the operator should do some thing solve this issue
        '''
        o_x, _ = self.ods.predict(x)
        dv_dx = self.lf.dv_dx(lf_para, x)
        rho_x = alpha * x.dot(x)
        if np.sqrt(x.dot(x)) > self.lf.overline_x:
            print('beyond the RoA, the robot should be stopped for the safety concern')
            return False
        elif np.sqrt(x.dot(x)) >= (self.lf.overline_x - epsilon):
            return o_x + controller(x, dv_dx, o_x, rho_x).reshape(-1)
        else:
            if not np.any(x):
                return np.zeros(2)
            elif dv_dx.dot(o_x) + rho_x <= 0:
                return o_x
            else:
                return o_x - (dv_dx.dot(o_x) + rho_x) / (dv_dx.dot(dv_dx)) * dv_dx


# the initial position is out of the RoA, the projected position is in dangerous area
def controller_1(x, dv_dx, o_x, rho_x, v_):
    '''
    :param x: position, np.array form, shape:(d_x,)
    :param dv_dx: gradient of V with respect to x, np.array form, shape:(d_x,)
    :param o_x: output of the ODS, np.array form, shape:(d_x,)
    :param rho_x: output of the positive function rho(x), a scalar
    :param v_: output of the positive function \overline v(x), a scalar
    :return: additional input, np.array form, shape:(d_x,)
    '''
    G = matrix(np.array([dv_dx, x]))
    max_v_norm2 = v_
    x = matrix(x)
    d_x = x.size[0]
    o_x = matrix(o_x)

    def obj(dot_x=None, z=None):
        if dot_x is None:
            o_x_norm2 = o_x.T * o_x
            if o_x_norm2[0, 0] < max_v_norm2:
                return 1, matrix(o_x, (d_x, 1))
            else:
                return 1, o_x / sqrt(o_x_norm2) * sqrt(max_v_norm2)
        obj = matrix(0.0, (2, 1))
        obj_pd = matrix(0.0, (2, d_x))
        obj[0, 0] = sum((dot_x - o_x).T * (dot_x - o_x))
        obj[1, 0] = sum(dot_x.T * dot_x - max_v_norm2)
        obj_pd[0, :] = 2 * (dot_x - o_x).T
        obj_pd[1, :] = 2 * dot_x.T
        if z is None:
            return obj, obj_pd
        I = spmatrix(1.0, range(d_x), range(d_x))
        return obj, obj_pd, 2 * (z[0] + z[1]) * I

    h = matrix([-rho_x, 0.0])
    solvers.options['maxiters'] = 100
    solvers.options['show_progress'] = False
    dot_x = solvers.cp(obj, G=G, h=h)['x']
    return np.array(dot_x).reshape(-1)


# the initial position is out of the RoA, the projected position is not in dangerous area
def controller_2(x, dv_dx, o_x, rho_x, v_):
    '''
    :param x: position, np.array form, shape:(d_x,)
    :param dv_dx: gradient of V with respect to x, np.array form, shape:(d_x,)
    :param o_x: output of the ODS, np.array form, shape:(d_x,)
    :param rho_x: output of the positive function rho(x), a scalar
    :param v_: output of the positive function \overline v(x), a scalar
    :return: additional input, np.array form, shape:(d_x,)
    '''
    max_v_norm2 = v_
    x = matrix(x)
    d_x = x.size[0]
    o_x = matrix(o_x)

    def obj(dot_x=None, z=None):
        if dot_x is None:
            o_x_norm2 = o_x.T * o_x
            if o_x_norm2[0, 0] < max_v_norm2:
                return 1, matrix(o_x, (d_x, 1))
            else:
                return 1, o_x / sqrt(o_x_norm2) * sqrt(max_v_norm2)
        obj = matrix(0.0, (2, 1))
        obj_pd = matrix(0.0, (2, d_x))
        obj[0, 0] = sum((dot_x - o_x).T * (dot_x - o_x))
        obj[1, 0] = sum(dot_x.T * dot_x - max_v_norm2)
        obj_pd[0, :] = 2 * (dot_x - o_x).T
        obj_pd[1, :] = 2 * dot_x.T
        if z is None:
            return obj, obj_pd
        I = spmatrix(1.0, range(d_x), range(d_x))
        return obj, obj_pd, 2 * (z[0] + z[1]) * I

    G = matrix(dv_dx, (1, d_x))
    h = matrix(-rho_x, (1, 1))
    solvers.options['maxiters'] = 100
    solvers.options['show_progress'] = False
    dot_x = solvers.cp(obj, G=G, h=h)['x']
    return np.array(dot_x).reshape(-1)


class LearnSds2:
    def __init__(self, lf, ods, x0):
        '''
        :param lf: lf_learner object
        :param ods: ods_learner object
        :param x0: initial position
        '''
        self.lf = lf
        self.ods = ods
        self.x0 = x0
        self.eta = np.sqrt(x0.dot(x0)) / (lf.overline_x - 1e-2)

    def controller(self, x, lf_para, v_, epsilon=1e0, alpha=1e-5):
        '''
        :param x: position, np.array form, shape:(d_x,)
        :param lf_para: parameters of the lf_learner object, an array with the form ((d_x+1)*d_H + d_H * d_x)
        :param v_: parameter to constrained velocity^2 out of the RoA
        :param epsilon: parameter to relax the activation condition, a scalar
        :param alpha: parameter to control the use of PDF rho(x), a scalar
        :return: dot_x or False, the latter implies that the x is beyond the RoA, the robot should be stopped for safety
         concern, or the operator should do some thing solve this issue
        '''
        x = x / self.eta
        o_x, _ = self.ods.predict(x)
        dv_dx = self.lf.dv_dx(lf_para, x)

        v_ = v_ / self.eta / self.eta
        c = alpha * v_ / self.lf.overline_x / self.lf.overline_x
        G = np.vstack((dv_dx.reshape(1, -1), x.reshape(1, -1)))
        rho_x = np.sqrt(c * np.linalg.det(G.dot(G.T)))

        if np.sqrt(x.dot(x)) > self.lf.overline_x:
            print('projected x is ', x)
            print('projected x norm is ', np.sqrt(x.dot(x)))
            print('beyond the RoA, the robot should be stopped for the safety concern')
            return False
        elif np.sqrt(x.dot(x)) >= (self.lf.overline_x - epsilon):
            return self.eta * controller_1(x, dv_dx, o_x, rho_x, v_).reshape(-1)
        else:
            if not np.any(x):
                return np.zeros(2)
            else:
                return self.eta * controller_2(x, dv_dx, o_x, rho_x, v_).reshape(-1)
