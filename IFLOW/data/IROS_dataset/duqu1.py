import numpy as np
import matplotlib.pyplot as plt
from gpr import sgpr
import torch

input_data = np.load(r"SShape.npy")  #这个文件可以直接当成txt使用的
# print(input_data.shape)
data = input_data.reshape(201,2)
n_dim = data.shape[0]
print(n_dim)
data = data[:,0:2]
# print(data.shape)
# print(data)

# np.savetxt(r"SShape.txt",data,delimiter=',')
plt.figure()
plt.scatter(data[:,0],data[:,1])
# plt.scatter(data[:,0],data[:,1])


plt.figure()
# ax1 = plt.gca()   # 保存当前的axes
# plt.plot(data[:,0])
# plt.imshow()
# plt.figure()
# ax1 = plt.gca()   # 保存当前的axes
# plt.plot(data[:,1])
# plt.imshow()
#

plt.subplot(2, 1, 1)  # 2行 2列 第二个子图 = 右上角
plt.plot(data[:,0])
plt.subplot(2, 1, 2)  # 2行 1列 第二个子图 = 底部
plt.plot(data[:,1])
plt.show()


# x_train = data[:,0]
# x_train = np.hstack((x_train.reshape(-1,1),x_train.reshape(-1,1)))
# y_train = data[:,1]
# Sgpr = sgpr(x_train, y_train, likelihood_noise=0.1, restart=1)
# Sgpr.train()
# Sgpr.save_param('para/' + 'GP_SShape.txt')
# # 测试集
# # x_test1 = np.linspace(-2.5,2.5,200)
# # x_test = np.hstack((x_test1.reshape(-1,1),x_test1.reshape(-1,1)))
# x_text1 = data[:,0]
# x_text = np.hstack((x_text1.reshape(-1,1),x_text1.reshape(-1,1)))
# y_text,_= Sgpr.predict_determined_input(x_text)
#
# print(y_text)
#
# plt.scatter(x_text1,y_text)
# plt.show()