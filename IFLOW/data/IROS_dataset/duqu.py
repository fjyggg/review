import numpy as np
import matplotlib.pyplot as plt
from gpr import sgpr
import torch

# input_data = np.load(r"dataSet_2D_rectangle.npy")  #这个文件可以直接当成txt使用的
# print(input_data.shape)
# data = input_data.reshape(999,2)
# data = data[::4,0:2]
# print(data.shape)
# print(data)
#
# np.savetxt(r"dataSet_2D_rectangle.txt",data,delimiter=',')


input_data = np.load(r"rShape.npy")  #这个文件可以直接当成txt使用的
print(input_data.shape)
data = input_data[0].reshape(838,2)  #2, 838, 2
data = data[:,0:2]
print(data.shape)
print(data)
np.savetxt(r"RShape.txt",data,delimiter=' ')


'''
# txt文件转为npy文件
embedding_vec = np.zeros((999,6))  #定义size
data = open('dataSet_2D_rectangle.txt')  #打开txt文件
next(data) #跳过首行
lines = data.readlines()
index = 0
for line in lines:
    list = line.strip('\n').split(" ")
    list = list[0:]   #舍去节点名字
    embedding_vec[index] = list
    index =index+1
embedding_vec = embedding_vec[300:-66,0:2]
print(embedding_vec)
embedding_vec = torch.tensor(embedding_vec)
embedding_vec = torch.unsqueeze(embedding_vec,0)
np.save('dataSet_2D_rectangle.npy', embedding_vec)
'''
