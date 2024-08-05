import numpy as np
import matplotlib.pyplot as plt
from algorithms.Learn_QLF import LearningQLF
from algorithms.Learn_SOSLF import LearningSoSLf
from algorithms.Learn_NNLF import LearnLyapunovFunction

font_size = 14
x_set_all1 = np.loadtxt('../x_set.txt')[0:1000:20, :]
dot_x_set_all1 = np.loadtxt('../dot_x_set.txt')[0:1000:20, :]
x_set1 = x_set_all1[0:-1:1, :]
dot_set1 = dot_x_set_all1[0:-1:1, :]
successive_x_set1 = x_set_all1[1::1, :]

x_set_all3 = np.loadtxt('../x_set.txt')[2000:3000:20, :] / 1.1
dot_x_set_all3 = np.loadtxt('../dot_x_set.txt')[2000:3000:20, :] / 1.1
x_set3 = x_set_all3[0:-1:1, :]
dot_set3 = dot_x_set_all3[0:-1:1, :]
successive_x_set3 = x_set_all3[1::1, :]

QLF_tra1 = np.loadtxt('QLF/tra1.txt')
QLF_tra2 = np.loadtxt('QLF/tra2.txt')

SOSLF_tra1 = np.loadtxt('SOSLF/tra1.txt')
SOSLF_tra2 = np.loadtxt('SOSLF/tra2.txt')

NNLF_tra1 = np.loadtxt('NNLF/tra1.txt')
NNLF_tra2 = np.loadtxt('NNLF/tra2.txt')

NSQLF_tra1 = np.loadtxt('NSQLF/tra1.txt')
NSQLF_tra2 = np.loadtxt('NSQLF/tra2.txt')

x_set = np.vstack((x_set1, x_set3))
dot_x_set = np.vstack((dot_set1, dot_set3))
successive_x_set = np.vstack((successive_x_set1, successive_x_set3))

demonstration_set = {'x_set': x_set, 'successive_x_set': successive_x_set}
# Initialize the algorithm

plt.figure(figsize=(16, 3), dpi=100)
plt.subplots_adjust(left=0.05, right=0.99, wspace=2.5, hspace=0.7, bottom=0.23, top=0.99)
plt1 = plt.subplot2grid((4, 16), (0, 0), rowspan=4, colspan=4)
plt2 = plt.subplot2grid((4, 16), (0, 4), rowspan=4, colspan=4)
plt3 = plt.subplot2grid((4, 16), (0, 8), rowspan=4, colspan=4)
plt4 = plt.subplot2grid((4, 16), (0, 12), rowspan=4, colspan=4)
area = {'x_min': -2, 'x_max': 27, 'y_min': -20, 'y_max': 25, 'step': 0.1}

mark_size = 5
plt1.scatter(successive_x_set1[:, 0], successive_x_set1[:, 1], c='red', alpha=1.0, s=mark_size, marker='o')
plt1.scatter(successive_x_set3[:, 0], successive_x_set3[:, 1], c='red', alpha=1.0, s=mark_size, marker='o')
plt1.scatter(0, 0, c='black', alpha=1.0, s=100, marker='*')
plt1.scatter(QLF_tra1[0, 0], QLF_tra1[0, 1], c='black', s=10, marker='o')
plt1.scatter(QLF_tra2[0, 0], QLF_tra2[0, 1], c='black', s=10, marker='o')
plt1.plot(QLF_tra1[:, 0], QLF_tra1[:, 1], c='blue', ls='-', linewidth=2)
plt1.plot(QLF_tra2[:, 0], QLF_tra2[:, 1], c='blue', ls='-', linewidth=2)
plt1.set_xlabel('$x$/mm', fontname='Times New Roman', fontsize=font_size)
plt1.set_ylabel('$y$/mm', fontname='Times New Roman', fontsize=font_size)
plt1.set_title('(a): Reproduced trajectory by using QLF', y=-0.30, fontname='Times New Roman', fontsize=font_size)
qlf = LearningQLF(demonstration_set=demonstration_set)
qlf_para = np.loadtxt('../LF_paras/QLF_para/para.txt')
step = area['step']
x = np.arange(area['x_min'], area['x_max'], step)
y = np.arange(area['y_min'], area['y_max'], step)
length_x = np.shape(x)[0]
length_y = np.shape(y)[0]
X, Y = np.meshgrid(x, y)
V = np.zeros((length_y, length_x))
for i in range(length_y):
    for j in range(length_x):
        pose = np.array([x[j], y[i]])
        V[i, j] = qlf.V(qlf_para, pose)
# plt.contourf(X, Y, Z)
levels = np.linspace(np.min(V), np.max(V), 20)
contour = plt1.contour(X, Y, V, levels=levels, alpha=0.5, linewidths=1.0, c='grey')
# plt1.clabel(contour, fontsize=8)


plt2.scatter(successive_x_set1[:, 0], successive_x_set1[:, 1], c='red', alpha=1.0, s=mark_size, marker='o')
plt2.scatter(successive_x_set3[:, 0], successive_x_set3[:, 1], c='red', alpha=1.0, s=mark_size, marker='o')
plt2.scatter(0, 0, c='black', alpha=1.0, s=100, marker='*')
plt2.scatter(SOSLF_tra1[0, 0], SOSLF_tra1[0, 1], c='black', s=10, marker='o')
plt2.scatter(SOSLF_tra2[0, 0], SOSLF_tra2[0, 1], c='black', s=10, marker='o')
plt2.plot(SOSLF_tra1[:, 0], SOSLF_tra1[:, 1], c='blue', ls='-', linewidth=2)
plt2.plot(SOSLF_tra2[:, 0], SOSLF_tra2[:, 1], c='blue', ls='-', linewidth=2)
plt2.set_xlabel('$x$/mm', fontname='Times New Roman', fontsize=font_size)
plt2.set_ylabel('$y$/mm', fontname='Times New Roman', fontsize=font_size)
plt2.set_title('(b): Reproduced trajectory by using SOS-LF', y=-0.30, fontname='Times New Roman', fontsize=font_size)
soslf = LearningSoSLf(demonstration_set=demonstration_set, M=2)
soslf_para = np.loadtxt('../LF_paras/SOSLF_para/para.txt')
step = area['step']
x = np.arange(area['x_min'], area['x_max'], step)
y = np.arange(area['y_min'], area['y_max'], step)
length_x = np.shape(x)[0]
length_y = np.shape(y)[0]
X, Y = np.meshgrid(x, y)
V = np.zeros((length_y, length_x))
for i in range(length_y):
    for j in range(length_x):
        pose = np.array([x[j], y[i]])
        V[i, j] = soslf.V(soslf_para, pose)
# plt.contourf(X, Y, Z)
levels = np.linspace(np.min(V), np.max(V), 45)
contour = plt2.contour(X, Y, V, levels=levels, alpha=0.5, linewidths=1.0, c='grey')

plt3.scatter(successive_x_set1[:, 0], successive_x_set1[:, 1], c='red', alpha=1.0, s=mark_size, marker='o')
plt3.scatter(successive_x_set3[:, 0], successive_x_set3[:, 1], c='red', alpha=1.0, s=mark_size, marker='o')
plt3.scatter(0, 0, c='black', alpha=1.0, s=100, marker='*')
plt3.scatter(NNLF_tra1[0, 0], NNLF_tra1[0, 1], c='black', s=10, marker='o')
plt3.scatter(NNLF_tra2[0, 0], NNLF_tra2[0, 1], c='black', s=10, marker='o')
plt3.plot(NNLF_tra1[:, 0], NNLF_tra1[:, 1], c='blue', ls='-', linewidth=2)
plt3.plot(NNLF_tra2[:, 0], NNLF_tra2[:, 1], c='blue', ls='-', linewidth=2)
plt3.set_xlabel('$x$/mm', fontname='Times New Roman', fontsize=font_size)
plt3.set_ylabel('$y$/mm', fontname='Times New Roman', fontsize=font_size)
plt3.set_title('(c): Reproduced trajectory by using NN-LF', y=-0.30, fontname='Times New Roman', fontsize=font_size)
nn_structure = [2, 3, 4, 5, 5]  # d_x, d_hidden, d_phi, q1, q2
# Initialize the algorithm
nn_lf = LearnLyapunovFunction(demonstration_set=demonstration_set, nn_structure=nn_structure)
nnlf_para = np.loadtxt('../LF_paras/NNLF_para/para.txt')
step = area['step']
x = np.arange(area['x_min'], area['x_max'], step)
y = np.arange(area['y_min'], area['y_max'], step)
length_x = np.shape(x)[0]
length_y = np.shape(y)[0]
X, Y = np.meshgrid(x, y)
V = np.zeros((length_y, length_x))
for i in range(length_y):
    for j in range(length_x):
        pose = np.array([x[j], y[i]])
        V[i, j] = nn_lf.lyapunov_function(nnlf_para, pose)
# plt.contourf(X, Y, Z)
levels = np.linspace(np.min(V), np.max(V), 45)
contour = plt3.contour(X, Y, V, levels=levels, alpha=0.5, linewidths=1.0, c='grey')

from algorithms.Learn_NSQLF_2 import LearnLyapunovFunction


plt4.scatter(successive_x_set1[:, 0], successive_x_set1[:, 1], c='red', alpha=1.0, s=mark_size, marker='o')
plt4.scatter(successive_x_set3[:, 0], successive_x_set3[:, 1], c='red', alpha=1.0, s=mark_size, marker='o')
plt4.scatter(0, 0, c='black', alpha=1.0, s=100, marker='*')
plt4.scatter(NSQLF_tra1[0, 0], NSQLF_tra1[0, 1], c='black', s=10, marker='o')
plt4.scatter(NSQLF_tra2[0, 0], NSQLF_tra2[0, 1], c='black', s=10, marker='o')
plt4.plot(NSQLF_tra1[:, 0], NSQLF_tra1[:, 1], c='blue', ls='-', linewidth=2)
plt4.plot(NSQLF_tra2[:, 0], NSQLF_tra2[:, 1], c='blue', ls='-', linewidth=2)
plt4.set_xlabel('$x$/mm', fontname='Times New Roman', fontsize=font_size)
plt4.set_ylabel('$y$/mm', fontname='Times New Roman', fontsize=font_size)
plt4.set_title('(d): Reproduced trajectory by using NS-QLF', y=-0.30, fontname='Times New Roman', fontsize=font_size)
ns_qlf = LearnLyapunovFunction(demonstration_set=demonstration_set, overline_x=30, d_H=30)
nsqlf_para = np.loadtxt('../LF_paras/NSQLF_para/para.txt')
step = area['step']
x = np.arange(area['x_min'], area['x_max'], step)
y = np.arange(area['y_min'], area['y_max'], step)
length_x = np.shape(x)[0]
length_y = np.shape(y)[0]
X, Y = np.meshgrid(x, y)
V = np.zeros((length_y, length_x))
for i in range(length_y):
    for j in range(length_x):
        pose = np.array([x[j], y[i]])
        V[i, j] = ns_qlf.lyapunov_function(nsqlf_para, pose)
# plt.contourf(X, Y, Z)
levels = np.linspace(np.min(V), np.max(V), 45)
contour = plt4.contour(X, Y, V, levels=levels, alpha=0.5, linewidths=1.0, c='grey')

plt.show()
