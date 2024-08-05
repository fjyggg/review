import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

font1 = {'family': 'Times new Roman', 'weight': 'normal', 'size': 15}

P_tra1 = np.loadtxt('tra1.txt')
P_tra2 = np.loadtxt('tra2.txt')
P_tra3 = np.loadtxt('tra3.txt')
P_tra4 = np.loadtxt('tra4.txt')
P_tra5 = np.loadtxt('tra5.txt')
P_tra6 = np.loadtxt('tra6.txt')

V_tra1 = np.loadtxt('velocity_tra1.txt')
V_tra2 = np.loadtxt('velocity_tra2.txt')
V_tra3 = np.loadtxt('velocity_tra3.txt')
V_tra4 = np.loadtxt('velocity_tra4.txt')
V_tra5 = np.loadtxt('velocity_tra5.txt')
V_tra6 = np.loadtxt('velocity_tra6.txt')

print(np.shape(P_tra1))
fig = plt.figure(figsize=(8, 4), dpi=100)  # figsize=(6, 10), dpi=100
plt.subplots_adjust(left=0.01, right=0.95, wspace=0.1, hspace=0.2, bottom=0.05, top=0.99)
gs = gridspec.GridSpec(4, 8)
ax1 = fig.add_subplot(gs[0:4, 0:4], projection='3d')

ax1.plot(P_tra1[:, 0], P_tra1[:, 1], P_tra1[:, 2], c='red', ls='--', label='tra1')
ax1.plot(P_tra2[:, 0], P_tra2[:, 1], P_tra2[:, 2], c='blue', ls='--', label='tra2')
ax1.plot(P_tra3[:, 0], P_tra3[:, 1], P_tra3[:, 2], c='green', ls='--', label='tra3')
ax1.plot(P_tra4[:, 0], P_tra4[:, 1], P_tra4[:, 2], c='grey', ls='--', label='tra4')
ax1.plot(P_tra5[:, 0], P_tra5[:, 1], P_tra5[:, 2], c='black', ls='--', label='tra5')
ax1.plot(P_tra6[:, 0], P_tra6[:, 1], P_tra6[:, 2], c='yellow', ls='--', label='tra6')

ax1.set_xlabel('$e_x$', fontdict=font1)
ax1.set_ylabel('$e_y$', fontdict=font1)
ax1.set_zlabel('$e_z$', fontdict=font1)
ax1.legend()

ax2 = fig.add_subplot(gs[0:4, 4:8], projection='3d')

# ax2.plot(V_tra1[:, 0], V_tra1[:, 1], V_tra1[:, 2], c='red', ls='--')
ax2.plot(V_tra2[:, 0], V_tra2[:, 1], V_tra2[:, 2], c='blue', ls='--')
ax2.plot(V_tra3[:, 0], V_tra3[:, 1], V_tra3[:, 2], c='green', ls='--')
ax2.plot(V_tra4[:, 0], V_tra4[:, 1], V_tra4[:, 2], c='grey', ls='--')
ax2.plot(V_tra5[:, 0], V_tra5[:, 1], V_tra5[:, 2], c='black', ls='--')
ax2.plot(V_tra6[:, 0], V_tra6[:, 1], V_tra6[:, 2], c='yellow', ls='--')

ax2.set_xlabel('$e_x$', fontdict=font1)
ax2.set_ylabel('$e_y$', fontdict=font1)
ax2.set_zlabel('$e_z$', fontdict=font1)

plt.show()