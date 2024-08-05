from algorithms.Learn_QLF import LearningQLF
from algorithms.Learn_ODS import mgpr_ods

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec


np.random.seed(5)
# -------------------------------------loading demonstration set--------------------------------------------------------
type = '/insert_test_tube/'
gap = 40
path = 'dataSet' + type + 'error_set/'
x_set1 = np.loadtxt(path + 'tra1.txt')[0:-1:gap, :]
successive_x_set1 = np.loadtxt(path + 'tra1.txt')[1::gap, :]
x_set2 = np.loadtxt(path + 'tra2.txt')[0:-1:gap, :]
successive_x_set2 = np.loadtxt(path + 'tra2.txt')[1::gap, :]
x_set3 = np.loadtxt(path + 'tra3.txt')[0:-1:gap, :]
successive_x_set3 = np.loadtxt(path + 'tra3.txt')[1::gap, :]
x_set4 = np.loadtxt(path + 'tra4.txt')[0:-1:gap, :]
successive_x_set4 = np.loadtxt(path + 'tra4.txt')[1::gap, :]
x_set5 = np.loadtxt(path + 'tra5.txt')[0:-1:gap, :]
successive_x_set5 = np.loadtxt(path + 'tra5.txt')[1::gap, :]
x_set6 = np.loadtxt(path + 'tra6.txt')[0:-1:gap, :]
successive_x_set6 = np.loadtxt(path + 'tra6.txt')[1::gap, :]

dot_x_set1 = np.loadtxt(path + 'velocity_tra1.txt')[0:-1:gap, :]
dot_x_set2 = np.loadtxt(path + 'velocity_tra2.txt')[0:-1:gap, :]
dot_x_set3 = np.loadtxt(path + 'velocity_tra3.txt')[0:-1:gap, :]
dot_x_set4 = np.loadtxt(path + 'velocity_tra4.txt')[0:-1:gap, :]
dot_x_set5 = np.loadtxt(path + 'velocity_tra5.txt')[0:-1:gap, :]
dot_x_set6 = np.loadtxt(path + 'velocity_tra6.txt')[0:-1:gap, :]

# --------------------------------------------------QLF learning--------------------------------------------------------
x_set = np.vstack((x_set1, x_set2, x_set3, x_set4, x_set5, x_set6))
dot_x_set = np.vstack((dot_x_set1, dot_x_set2, dot_x_set3, dot_x_set4, dot_x_set5, dot_x_set6))
successive_x_set = np.vstack((successive_x_set1, successive_x_set2, successive_x_set3, successive_x_set4,
                              successive_x_set5, successive_x_set6))
demonstration_set = {'x_set': x_set, 'dot_x_set': dot_x_set, 'successive_x_set': successive_x_set}
# Initialize the algorithm
qlf = LearningQLF(demonstration_set=demonstration_set)
# Start learning
print('start qlf_learning:')
learning_options = {'max_iter': 20000, 'disp': True, 'ftol': 1e-6}
save_options = {'save_flag': True, 'save_path': 'LF_paras' + type + 'QLF_para/para.txt'}
d_x = np.shape(x_set)[1]


def ineq_cons(l):
    L = l.reshape(d_x, d_x)
    P = L.dot(L.T)
    eigenvalue, _ = np.linalg.eig(P)
    return np.hstack((eigenvalue - 0.5, 1e0 - eigenvalue))


ineq_cons_ = {'type': 'ineq', 'fun': lambda l: ineq_cons(l)}
l0 = np.eye(d_x).reshape(-1) * 1e-2
print(l0)
# qlf_para = qlf.learning(l0=l0, learning_options=learning_options, cons=ineq_cons_, save_options=save_options)
qlf_para = np.loadtxt('LF_paras' + type + 'QLF_para/para.txt')
# qlf.show_learning_result(w=qlf_para)

# ---------------------------------------------------ODS learning-------------------------------------------------------
dataset2ODS = {'input_set': np.vstack((x_set1, x_set2, x_set3, x_set4, x_set5, x_set6)),
               'output_set': np.vstack((dot_x_set1, dot_x_set2, dot_x_set3, dot_x_set4, dot_x_set5, dot_x_set6))}
ods = mgpr_ods(training_set=dataset2ODS, likelihood_noise=0.01)
print('start ODS learning:')
ods.train()


# ------------------------------------------------online conducting SDS-------------------------------------------------
def controller(x, lf_para, alpha=1e-5):
    '''
    :param x: position, np.array form, shape:(d_x,)
    :param lf_para: parameters of the lf_learner object
    :param alpha: parameter to control the use of PDF rho(x), a scalar
    :return:
    '''
    o_x, _ = ods.predict(x)
    dv_dx = qlf.dv_dx(lf_para, x)
    rho_x = alpha * x.dot(x)
    if not np.any(x):
        return 0
    elif dv_dx.dot(o_x) + rho_x <= 0:
        return o_x
    else:
        return o_x - (dv_dx.dot(o_x) + rho_x) / (dv_dx.dot(dv_dx)) * dv_dx

# qlf.show_learning_result(qlf_para.reshape(d_x, d_x))

fig = plt.figure(figsize=(8, 8), dpi=100)
gs = gridspec.GridSpec(4, 4)
ax = fig.add_subplot(gs[0:4, 0:4], projection='3d')
x = x_set1[0, :]
ax.scatter(x[0], x[1], x[2], c='black', alpha=1.0, s=50, marker='o')
ax.scatter(0, 0, 0, c='black', alpha=1.0, s=100, marker='*')
max_steps = 2000
T = 1e-2
x_list = []
for i in range(max_steps):
    x_list.append(x)
    print('-----------at step ', i, '------------')
    x_dot = controller(x, qlf_para, alpha=1e-5)
    print('x: ', x)
    print('x_norm is : ', np.sqrt(x.dot(x)))
    x = x + x_dot * T

x_list = np.array(x_list)
demonstration_set_size = np.shape(x_set)[0]
mark_size = np.ones(demonstration_set_size) * 10
count = 0
for i in range(demonstration_set_size):
    ax.scatter(successive_x_set[i, 0], successive_x_set[i, 1], successive_x_set[i, 2], c='blue', alpha=1.0, s=0.5, marker='o')

ax.plot(x_list[:, 0], x_list[:, 1], x_list[:, 2])
plt.show()

