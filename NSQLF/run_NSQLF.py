from algorithms.Learn_NSQLF import Learn_LF
from algorithms.Learn_ODS import mgpr_ods
from algorithms.Learn_SDS import LearnSds
from algorithms.Learn_SDS import LearnSds2  # for learning stable ADS out of the RoA

from scipy.io import loadmat
import numpy as np
from cvxopt import solvers, matrix, spdiag, sqrt, div, exp, spmatrix, log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import time

np.random.seed(5)

# -------------------------------------loading demonstration set--------------------------------------------------------
type = '/insert_test_tube/'
scale = 1
gap = 40
path = 'dataSet' + type + 'error_set/'
x_set1 = np.loadtxt(path + 'tra1.txt')[0:-1:gap, :] / scale
successive_x_set1 = np.loadtxt(path + 'tra1.txt')[1::gap, :] / scale
x_set2 = np.loadtxt(path + 'tra2.txt')[0:-1:gap, :] / scale
successive_x_set2 = np.loadtxt(path + 'tra2.txt')[1::gap, :] / scale
x_set3 = np.loadtxt(path + 'tra3.txt')[0:-1:gap, :] / scale
successive_x_set3 = np.loadtxt(path + 'tra3.txt')[1::gap, :] / scale
x_set4 = np.loadtxt(path + 'tra4.txt')[0:-1:gap, :] / scale
successive_x_set4 = np.loadtxt(path + 'tra4.txt')[1::gap, :] / scale
x_set5 = np.loadtxt(path + 'tra5.txt')[0:-1:gap, :] / scale
successive_x_set5 = np.loadtxt(path + 'tra5.txt')[1::gap, :] / scale
x_set6 = np.loadtxt(path + 'tra6.txt')[0:-1:gap, :] / scale
successive_x_set6 = np.loadtxt(path + 'tra6.txt')[1::gap, :] / scale

dot_x_set1 = np.loadtxt(path + 'velocity_tra1.txt')[0:-1:gap, :] / scale
dot_x_set2 = np.loadtxt(path + 'velocity_tra2.txt')[0:-1:gap, :] / scale
dot_x_set3 = np.loadtxt(path + 'velocity_tra3.txt')[0:-1:gap, :] / scale
dot_x_set4 = np.loadtxt(path + 'velocity_tra4.txt')[0:-1:gap, :] / scale
dot_x_set5 = np.loadtxt(path + 'velocity_tra5.txt')[0:-1:gap, :] / scale
dot_x_set6 = np.loadtxt(path + 'velocity_tra6.txt')[0:-1:gap, :] / scale


# -------------------------------------------rbfnn-based neural-shaped LF learning--------------------------------------
x_set = np.vstack((x_set1, x_set2, x_set3, x_set4, x_set5, x_set6))
dot_x_set = np.vstack((dot_x_set1, dot_x_set2, dot_x_set3, dot_x_set4, dot_x_set5, dot_x_set6))
successive_x_set = np.vstack((successive_x_set1, successive_x_set2, successive_x_set3, successive_x_set4,
                              successive_x_set5, successive_x_set6))
demonstration_set = {'x_set': x_set, 'dot_x_set': dot_x_set, 'successive_x_set': successive_x_set}
print(np.max(np.sqrt(np.sum(x_set * x_set, 1))))
mu_set = []
sigma_set = []
# 0.3 for insert test tube task and 0.6 for stacking task
if type == '/insert_test_tube/':
    overline_x = 0.25
else:
    overline_x = 0.58
dx = 3
for i in np.arange(-0.2, 0.2, 0.1):
    for j in np.arange(-0.5, 0.5, 0.2):
        for k in np.arange(-0.1, 0.3, 0.15):
            mu_set.append(np.array([i, j, k]) * 1.0)
            # using the constraint
            temp = np.sqrt(dx) / 2 * (np.sqrt(mu_set[-1].dot(mu_set[-1])) + overline_x) * overline_x
            sigma_set.append(np.array([np.sqrt(temp), np.sqrt(temp), np.sqrt(temp)]))
mu_set = np.array(mu_set)
sigma_set = np.array(sigma_set)
feature_parameters = {'mu_set': mu_set, 'sigma_set': sigma_set}
P0 = matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * 0.0
beta = 10.0
# Initialize the algorithm
ns_qlf = Learn_LF(demonstration_set=demonstration_set, feature_parameters=feature_parameters, P0=P0, alpha=0.0,
                  beta=beta, overline_x=overline_x)
# Start learning
print('start nnqlf_learning:')
training_options = {'feastol': 1e-6, 'abstol': 1e-6, 'reltol': 1e-6, 'maxiters': 150, 'show_progress': True}
save_options = {'save_flag': True, 'save_path': 'LF_paras' + type + 'NSQLF_para/para.txt'}
# nsqlf_para = ns_qlf.learning(training_options=training_options, save_options=save_options)
nsqlf_para = np.loadtxt('LF_paras' + type + 'NSQLF_para/para.txt')
# ns_qlf.show_learning_result(w=nsqlf_para, save_options=None)
# ---------------------------------------------------ODS learning-------------------------------------------------------
dataset2ODS = {'input_set': np.vstack((x_set1, x_set2, x_set3, x_set4, x_set5, x_set6)),
               'output_set': np.vstack((dot_x_set1, dot_x_set2, dot_x_set3, dot_x_set4, dot_x_set5, dot_x_set6))}
ods = mgpr_ods(training_set=dataset2ODS, likelihood_noise=1e-2)
print('start ODS learning:')
ods.train()

# ------------------------------------------------online conducting SDS-------------------------------------------------
fig = plt.figure(figsize=(8, 8), dpi=100)
gs = gridspec.GridSpec(4, 4)
ax = fig.add_subplot(gs[0:4, 0:4], projection='3d')
x = x_set1[0, :] + np.array([0, 0.3, 0])
# ax.scatter(x[0], x[1], x[2], c='black', alpha=1.0, s=50, marker='o')
ax.scatter(0, 0, 0, c='black', alpha=1.0, s=100, marker='*')
max_steps = 2000
T = 1e-2
x_list = []

# judge the initial position out of the RoA
OutofRoA_flag = False
if type == '/insert_test_tube/':
    ns_qlf.overline_x = 0.46
if np.sqrt(x.dot(x)) >= ns_qlf.overline_x:
    OutofRoA_flag = True
if OutofRoA_flag is True:
    print('the initial position is out of the RoA')
    ns_qlf.overline_x = 0.45
    sds = LearnSds2(ns_qlf, ods, x)
else:
    print('the initial position is in the RoA')
    sds = LearnSds(ns_qlf, ods)

for i in range(max_steps):
    x_list.append(x)
    print('-----------at step ', i, '------------')
    if OutofRoA_flag is True:
        x_dot = sds.controller(x, nsqlf_para, v_=0.1 * 0.1, epsilon=1e-2, alpha=1e-5)
    else:
        x_dot = sds.controller(x, nsqlf_para, epsilon=1e-2, alpha=1e-5)
    print('x: ', x)
    print('x_dot is : ', x_dot)
    x = x + x_dot * T

x_list = np.array(x_list)
demonstration_set_size = np.shape(x_set)[0]
print(demonstration_set_size)
mark_size = 10
count = 0
for i in range(demonstration_set_size):
    ax.scatter(successive_x_set[i, 0], successive_x_set[i, 1], successive_x_set[i, 2], c='blue', alpha=1.0, s=mark_size, marker='o')


# ax.plot(x_list[:, 0], x_list[:, 1], x_list[:, 2])
plt.show()
# np.savetxt('reproduced trajectories/NSQLF/tra1.txt', x_list)
