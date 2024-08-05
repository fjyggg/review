from algorithms.Learn_NSQLF_2 import LearnLyapunovFunction
import numpy as np
import matplotlib.pyplot as plt

font_size = 12
font1 = {'family' : 'Times new Roman',
'weight' : 'normal',
'size'   : font_size,
}

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

x_set = np.vstack((x_set1, x_set3 / 1.1))
dot_x_set = np.vstack((dot_set1, dot_set3 / 1.1))
successive_x_set = np.vstack((successive_x_set1, successive_x_set3 / 1.1))
demonstration_set = {'x_set': x_set, 'successive_x_set': successive_x_set}
ns_qlf = LearnLyapunovFunction(demonstration_set=demonstration_set, overline_x=30, d_H=30)
nsqlf_para = np.loadtxt('../LF_paras/NSQLF_para/para.txt')
plt.figure(figsize=(7, 3), dpi=100)
plt.subplots_adjust(left=0.1, right=0.99, wspace=3.0, hspace=0.7, bottom=0.28, top=0.99)
plt1 = plt.subplot2grid((4, 8), (0, 0), rowspan=4, colspan=4)
plt2 = plt.subplot2grid((4, 8), (0, 4), rowspan=4, colspan=4)

area = {'x_min': -2, 'x_max': 30, 'y_min': -20, 'y_max': 25, 'step': 0.1}
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
contour = plt1.contour(X, Y, V, levels=levels, alpha=0.5, linewidths=1.0, c='grey')
# plot a circle:
theta = np.arange(-np.pi, np.pi, 2 * np.pi / 100)
x_ = 29 * np.cos(theta)
y_ = 29 * np.sin(theta)
x__ = []
y__ = []
for i in range(100):
    if (x_[i] > x[0]) and (x_[i] < x[-1]) and (y_[i] > y[0]) and (y_[i] < y[-1]):
        x__.append(x_[i])
        y__.append(y_[i])
x__ = np.array(x__)
y__ = np.array(y__)
plt1.plot(x__, y__, ls='--', linewidth=1, c='black', alpha=1)
plt1.scatter(successive_x_set1[:, 0], successive_x_set1[:, 1], c='red', alpha=1.0, s=5, marker='o')
plt1.scatter(successive_x_set3[:, 0], successive_x_set3[:, 1], c='red', alpha=1.0, s=5, marker='o')
NSQLF_tra1 = np.loadtxt('NSQLF/OutOfRoA/tra.txt')
plt1.scatter(0, 0, c='black', alpha=1.0, s=100, marker='*')
plt1.scatter(NSQLF_tra1[0, 0], NSQLF_tra1[0, 1], c='black', s=20, marker='o')
plt1.scatter(NSQLF_tra1[1000, 0], NSQLF_tra1[1000, 1], c='black', s=20, marker='o')
plt1.scatter(NSQLF_tra1[2000, 0], NSQLF_tra1[2000, 1], c='black', s=20, marker='o')
plt1.scatter(NSQLF_tra1[2500, 0], NSQLF_tra1[2500, 1], c='black', s=20, marker='o')
plt1.plot(NSQLF_tra1[:, 0], NSQLF_tra1[:, 1], c='black', ls='-', linewidth=2, label='real trajectory')
x0 = np.array([30.0, 35.0])
eta = np.sqrt(x0.dot(x0)) / (ns_qlf.overline_x - 1)
plt1.scatter(NSQLF_tra1[0, 0] / eta, NSQLF_tra1[0, 1] / eta, c='blue', s=20, marker='o')
plt1.scatter(NSQLF_tra1[1000, 0] / eta, NSQLF_tra1[1000, 1] / eta, c='blue', s=20, marker='o')
plt1.scatter(NSQLF_tra1[2000, 0] / eta, NSQLF_tra1[2000, 1] / eta, c='blue', s=20, marker='o')
plt1.scatter(NSQLF_tra1[2500, 0] / eta, NSQLF_tra1[2500, 1] / eta, c='blue', s=20, marker='o')
plt1.plot(NSQLF_tra1[:, 0] / eta, NSQLF_tra1[:, 1] / eta, c='blue', ls='-', linewidth=2, label='projected trajectory')
plt1.legend(loc='upper left', prop=font1, frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt1.arrow(NSQLF_tra1[0, 0], NSQLF_tra1[0, 1], NSQLF_tra1[0, 0] / eta - NSQLF_tra1[0, 0], NSQLF_tra1[0, 1] / eta - NSQLF_tra1[0, 1])
plt1.arrow(NSQLF_tra1[1000, 0], NSQLF_tra1[1000, 1], NSQLF_tra1[1000, 0] / eta - NSQLF_tra1[1000, 0], NSQLF_tra1[1000, 1] / eta - NSQLF_tra1[1000, 1])
plt1.arrow(NSQLF_tra1[2000, 0], NSQLF_tra1[2000, 1], NSQLF_tra1[2000, 0] / eta - NSQLF_tra1[2000, 0], NSQLF_tra1[2000, 1] / eta - NSQLF_tra1[2000, 1])
plt1.arrow(NSQLF_tra1[2500, 0], NSQLF_tra1[2500, 1], NSQLF_tra1[2500, 0] / eta - NSQLF_tra1[2500, 0], NSQLF_tra1[2500, 1] / eta - NSQLF_tra1[2500, 1])
plt1.set_xlabel('$x$/mm', fontname='Times New Roman', fontsize=font_size)
plt1.set_ylabel('$y$/mm', fontname='Times New Roman', fontsize=font_size)
plt1.set_title('(a): Reproduced trajectory\n out of the RoA', y=-0.4, fontname='Times New Roman', fontsize=font_size)

NSQLF_dot_x_tra1 = np.loadtxt('NSQLF/OutOfRoA/dot_x_tra.txt')
plt2.plot(NSQLF_dot_x_tra1[:, 0], NSQLF_dot_x_tra1[:, 1], c='black', ls='-',  linewidth=2, label='projected trajectory')
plt2.plot(NSQLF_dot_x_tra1[:, 0] / eta, NSQLF_dot_x_tra1[:, 1] / eta, c='blue', ls='-',  linewidth=2, label='real trajectory')
# plot a circle:
theta = np.arange(-np.pi, np.pi, 2 * np.pi / 100)
x_ = 40 * np.cos(theta)
y_ = 40 * np.sin(theta)
plt2.plot(x_, y_, ls='--', linewidth=1, c='black', alpha=1)
plt2.plot(x_ / eta, y_ / eta, ls='--', linewidth=1, c='blue', alpha=1)
plt2.scatter(NSQLF_dot_x_tra1[0, 0], NSQLF_dot_x_tra1[0, 1], c='black', s=20, marker='o')
plt2.scatter(NSQLF_dot_x_tra1[0, 0] / eta, NSQLF_dot_x_tra1[0, 1] / eta, c='blue', s=20, marker='o')
plt2.scatter(0, 0, c='black', alpha=1.0, s=100, marker='*')
# plt2.legend(loc='upper left', prop=font1, frameon=False, handlelength=1, ncol=2, columnspacing=1)
plt2.set_xlabel('$\dot x$/ $(mm\cdot s^{-1})$', fontname='Times New Roman', fontsize=font_size)
plt2.set_ylabel('$\dot y$/ $(mm\cdot s^{-1})$', fontname='Times New Roman', fontsize=font_size)
plt2.set_title('(b): Reproduced velocity trajectory\n out of the RoA', y=-0.4, fontname='Times New Roman', fontsize=font_size)
plt.show()


