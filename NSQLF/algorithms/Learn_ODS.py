import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad
from scipy.optimize import minimize
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve
import time
from scipy.io import loadmat
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib import gridspec


np.random.seed(5)


cdict1 = {'red':   ((0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.1),
                    (1.0, 1.0, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue':  ((0.0, 0.0, 1.0),
                    (0.5, 0.1, 0.0),
                    (1.0, 0.0, 0.0))
          }
blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
font_size = 14
font1 = {'family' : 'Times new Roman',
'weight' : 'normal',
'size'   : font_size,
}


def plot_examples(colormaps, data, plt_handle=None):
    """
    Helper function to plot data with associated colormap.
    """
    if plt_handle is None:
        plt_handle = plt
    n = len(colormaps)
    fig, axs = plt_handle.subplots(1, n, figsize=(n * 2 + 2, 3),
                                   constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-1, vmax=1)
        fig.colorbar(psm, ax=ax)
    plt_handle.show()


# 0均值，高斯核
class sgpr:
    def __init__(self, X, y, likelihood_noise=0.1, restart=1):
        self.X = X
        self.y = y
        self.init_param = []
        self.param = []
        self.input_dim = np.shape(self.X)[1]
        self.input_num = np.shape(self.X)[0]
        self.likelihood_noise = likelihood_noise
        self.restart = restart
        self.cov_y_y = None
        self.beta = None

    def init_random_param(self):
        kern_length_scale = 0.01 * np.random.normal(size=self.input_dim) + 0.1
        kern_noise = 0.1 * np.random.normal(size=1)
        self.init_param = np.hstack((kern_noise, kern_length_scale))
        self.param = self.init_param.copy()
        # print("self.init_param is", self.init_param)

    def set_param(self, param):
        self.param = param.copy()
        variance_matrix = self.likelihood_noise ** 2 * np.eye(self.input_num)
        variance_matrix[-1, -1] = 1e-5
        self.cov_y_y = self.rbf(self.X, self.X, self.param) + variance_matrix
        self.beta = solve(self.cov_y_y, self.y)
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

    def set_XY(self, X, y):
        self.X = X
        self.y = y
        self.input_dim = np.shape(self.X)[1]
        self.input_num = np.shape(self.X)[0]

    def build_objective(self, param):
        cov_y_y = self.rbf(self.X, self.X, param)
        variance_matrix = self.likelihood_noise ** 2 * np.eye(self.input_num)
        variance_matrix[-1, -1] = 1e-5
        cov_y_y = cov_y_y + variance_matrix
        # out = 0.5 * (np.dot(self.y, solve(cov_y_y, self.y)) + np.log(np.linalg.det(cov_y_y)) + self.input_dim / 2 * np.log(2 * np.pi))
        # print(np.shape(self.mean_function(self.X)))
        out = - mvn.logpdf(self.y, np.zeros(self.input_num), cov_y_y)
        return out

    def train(self):
        max_logpdf = -1e20
        # cons = con((0.001, 10))
        for i in range(self.restart):
            self.init_random_param()
            result = minimize(value_and_grad(self.build_objective), self.init_param, jac=True, method='L-BFGS-B', tol=0.01)
            logpdf = -result.fun
            param = result.x
            if logpdf > max_logpdf:
                self.param = param
                max_logpdf = logpdf
        variance_matrix = self.likelihood_noise ** 2 * np.eye(self.input_num)
        variance_matrix[-1, -1] = 1e-5
        self.cov_y_y = self.rbf(self.X, self.X, self.param) + variance_matrix
        self.beta = solve(self.cov_y_y, self.y)
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

    def rbf(self, x, x_, param):  # 输入的是矩阵，输出的是标量
        kern_noise = param[0]
        sqrt_kern_length_scale = param[1:]
        diffs = np.expand_dims(x / sqrt_kern_length_scale, 1) - np.expand_dims(x_ / sqrt_kern_length_scale, 0)
        return kern_noise**2 * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def predict_determined_input(self, inputs):
        # inputs 是矩阵
        num_inputs = np.shape(inputs)[0]
        cov_y_f = self.rbf(self.X, inputs, self.param)
        mean_outputs = np.dot(cov_y_f.T, self.beta.reshape((-1, 1)))
        var = (self.param[0]**2 - np.diag(np.dot(np.dot(cov_y_f.T, self.inv_cov_y_y), cov_y_f))).reshape(-1, 1)
        return mean_outputs, var

    def print_params(self):
        print('final param is', self.param)

    def callback(self, param):
        # ToDo: add something you want to know about the training process
        pass


class mgpr:
    def __init__(self, X, Y, likelihood_noise=1.0, restart=1):
        self.X = X
        self.Y = Y
        self.param = []
        self.input_dim = np.shape(X)[1]
        self.input_num = np.shape(X)[0]
        self.output_dim = np.shape(Y)[1]
        self.likelihood_noise = np.zeros(self.output_dim) + likelihood_noise
        self.restart = restart

    def set_XY(self, X, Y):
        self.X = X
        self.Y = Y
        self.input_num = np.shape(X)[0]
        for i in range(self.output_dim):
            self.models[i].set_XY(self.X, self.Y[i])

    def set_param(self, param):
        self.create_models()
        self.param = param.copy()
        for i in range(self.output_dim):
            self.models[i].set_param(self.param[i])

    def create_models(self):
        self.models = []
        for i in range(self.output_dim):
            self.models.append(sgpr(self.X, self.Y[:, i], likelihood_noise=self.likelihood_noise[i], restart=self.restart))

    def init_random_param(self):
        for i in range(self.output_dim):
            self.models[i].init_random_param()

    def train(self, save_options=None):
        self.create_models()
        self.init_random_param()
        for i in range(self.output_dim):
            print('training model ', i, '...')
            self.models[i].train()
            if i == 0:
                self.param = self.models[i].param.copy()
            else:
                self.param = np.vstack((self.param, self.models[i].param.copy()))
        if save_options is not None:
            if save_options['flag'] is True:
                np.savetxt(save_options['path'], self.param)

    def print_params(self):
        print('final param is', self.param)

    def predict_determined_input(self, inputs):
        mean_outputs0, var0 = self.models[0].predict_determined_input(inputs)
        mean_outputs1, var1 = self.models[1].predict_determined_input(inputs)
        mean_outputs2, var2 = self.models[2].predict_determined_input(inputs)
        mean_outputs = np.hstack((mean_outputs0, mean_outputs1, mean_outputs2))
        vars = np.hstack((var0, var1, var2))
        input_dim = np.shape(inputs)[0]
        if input_dim == 1:
            mean_outputs = mean_outputs.reshape(-1)
            vars = vars.reshape(-1)
        return mean_outputs, vars


class mgpr_ods:
    def __init__(self, training_set, likelihood_noise=1):
        self.input_set = training_set['input_set']
        self.output_set = training_set['output_set']
        self.mgpr = mgpr(self.input_set, self.output_set, likelihood_noise)
        self.data_size = np.shape(self.input_set)[0]

    def set_param(self, param):
        self.mgpr.set_param(param)

    def train(self, save_options=None):
        self.mgpr.train(save_options)

    def predict(self, input):
        input = input.reshape(1, -1)
        outputs, vars = self.mgpr.predict_determined_input(input)
        return outputs, vars

    def show_learning_result(self, area, save_options, scale=1):
        '''
        :param area: a dictionary with keys 'x_max', 'x_min', 'y_max', 'y_min' and 'step'
        :param save_options: a dictionary with keys 'save_flag', 'save_path'
        :param scale: to scale the influence of the ods uncertainty
        :return: void
        '''
        step = area['step']
        x = np.arange(area['x_min'], area['x_max'], step)
        y = np.arange(area['y_min'], area['y_max'], step)
        X, Y = np.meshgrid(x, y)
        length_x = np.shape(x)[0]
        length_y = np.shape(y)[0]
        Dot_x = np.zeros((length_y, length_x))
        Dot_y = np.zeros((length_y, length_x))
        Color = np.zeros((length_y, length_x))
        for i in range(length_y):
            for j in range(length_x):
                pose = np.array([x[j], y[i]])
                velocitys, vars = self.predict(pose)
                Dot_x[i, j], Dot_y[i, j] = velocitys
                Color[i, j] = np.exp(-0.5 * 1 / scale * np.sum(vars)**2)
        fig, ax = plt.subplots()
        strm = ax.streamplot(X, Y, Dot_x, Dot_y, density=2.0, color=Color, cmap=blue_red1, linewidth=0.3, maxlength=0.1,
                             minlength=0.01, arrowstyle='simple', arrowsize=0.5)
        # fig.colorbar(strm.lines)
        mark_size = np.ones(self.data_size) * 10
        ax.scatter(self.input_set[:, 0], self.input_set[:, 1], c='red', alpha=1.0, s=mark_size)
        ax.scatter(0, 0, c='black', alpha=1.0, s=300, marker='*')
        if save_options is not None:
            save_flag = save_options['save_flag']
            save_path = save_options['save_path']
            if save_flag is True:
                plt.savefig(save_path, dpi=300)
        plt.show()

    def show_properties(self, area):
        step = area['step']
        x = np.arange(area['x_min'], area['x_max'], step)
        y = np.arange(area['y_min'], area['y_max'], step)
        length_x = np.shape(x)[0]
        length_y = np.shape(y)[0]
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((length_y, length_x))
        V = np.zeros((length_y, length_x))
        fig = plt.figure(figsize=(8, 4), dpi=100)  # figsize=(6, 10), dpi=100
        plt.subplots_adjust(left=0.05, right=0.99, wspace=0.2, hspace=0.05, bottom=0.15, top=0.99)
        gs = gridspec.GridSpec(4, 8)
        ax0 = fig.add_subplot(gs[0:4, 0:4])
        ax1 = fig.add_subplot(gs[0:4, 4:8])

        for i in range(length_y):
            for j in range(length_x):
                pose = np.array([x[j], y[i]])
                velocitys, vars = self.predict(pose)
                Z[i, j] = np.sqrt(velocitys.dot(velocitys))
                V[i, j] = np.sqrt(vars.dot(vars))
        pcm1 = ax0.pcolor(X, Y, Z, alpha=1.0)
        pcm2 = ax1.pcolor(X, Y, V, alpha=1.0)
        fig.colorbar(pcm1, ax=ax0, extend='max')
        fig.colorbar(pcm2, ax=ax1, extend='max')

        mark_size = np.ones(self.data_size) * 10
        ax0.scatter(self.input_set[:, 0], self.input_set[:, 1], c='red', alpha=1.0, s=mark_size)
        ax1.scatter(self.input_set[:, 0], self.input_set[:, 1], c='red', alpha=1.0, s=mark_size)
        # ax.scatter(0, 0, c='black', alpha=1.0, s=300, marker='*')

        ax0.set_title('(a): Velocity norm', y=-0.15, fontname='Times New Roman', fontsize=font_size)
        ax1.set_title('(b): Variance norm', y=-0.15, fontname='Times New Roman', fontsize=font_size)
        plt.show()



