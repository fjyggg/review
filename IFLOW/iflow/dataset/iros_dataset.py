import numpy as np
import os
import torch
from iflow.dataset.generic_dataset import CycleDataset

from sklearn.decomposition import PCA
import scipy.fftpack

directory = os.path.abspath(__file__+'/../../..')+'/data/IROS_dataset/'

class IROS():
    def __init__(self, filename, device = torch.device('cpu')):   #把形状轨迹给过来了，主要是在cpu里面训练
        ## Define Variables and Load trajectories ##
        self.filename = filename
        data = np.load(directory+filename+'.npy')  #载入数据
        self.dim = 2
        self.dt = 0.05  # 0.05对标的是franka的50hz频率，机械臂的，原来是0.01
        self.trajs_real=[]
        for i in range(data.shape[0]):
            self.trajs_real.append(data[i, :, :])  #SShape只有一条轨迹，多体哦啊轨迹的话，会把这个赋值给他
        trajs_np = np.asarray(self.trajs_real)
        self.n_trajs = trajs_np.shape[0]    #轨迹的数量
        self.n_steps = trajs_np.shape[1]    #当前轨迹的行数
        self.n_dims = trajs_np.shape[2]     #当前轨迹的列数

        ## Normalize Trajectories
        trajs_np = np.reshape(trajs_np, (self.n_trajs * self.n_steps, self.n_dims))  #全部叠加在一起(201, 2)
        self.mean = np.mean(trajs_np, axis=0)
        self.std = np.std(trajs_np, axis=0)
        self.trajs_normalized = self.normalize(self.trajs_real)  #归一化  均值方差归一化，(x-mean）/std,改变量纲(1, 201, 2),下面的函数有的


        ## Build Train Dataset
        self.train_data = []
        for i in range(self.trajs_normalized.shape[0]):  #(1, 201, 2)
            self.train_data.append(self.trajs_normalized[i, ...])
        ### Mean Angular velocity 平均角速度###
        self.w = self.get_mean_ang_vel()  #这只是一个数啊1.269330365086785
        self.train_phase_data = []
        for i in range(len(self.train_data)):#1
            trj = self.train_data[0]
            N = trj.shape[0]  #201
            t = np.linspace(0,N*self.dt,N)  #总时间分段
            phase_trj = np.arctan2(np.sin(t),np.cos(t))  #这个是角度吗，半径默认为1吗
            self.train_phase_data.append(phase_trj)  #得到相位的数据集

        self.dataset = CycleDataset(trajs=self.train_data, device=device, trajs_phase=self.train_phase_data)

    def get_mean_ang_vel(self):
        ########## PCA trajectories and Fourier Transform #############
        self.pca = PCA(n_components=2)      #pca降维到两个维度
        self.pca.fit(self.train_data[0])    #对数据self.train_data[0]用pca训练,(201, 2)
        pca_trj = self.pca.transform(self.train_data[0])   #用X来训练PCA模型，同时返回降维后的数据为pca_trj
        ### Fourier Analysis
        N = pca_trj.shape[0]  #(201, 2)
        yf = scipy.fftpack.fft(pca_trj[:, 0])
        xf = np.linspace(0.0, 1. / (2 * self.dt), N // 2)  #0-10取100个数，原来是0-50取100个数，看dt的大小

        max_i = np.argmax(np.abs(yf[:N // 2]))  #返回最大索引值

        self.freq = xf[max_i]
        w = 2 * np.pi * self.freq  #1.269330365086785这个有什么意义吗，用傅里叶搞这么一出
        return w
    
    def normalize(self, X):
        Xn = (X - self.mean)/self.std
        return Xn

    def unormalize(self, Xn):
        X = Xn*self.std + self.mean
        return X


if __name__ == "__main__":
    filename = 'IShape'
    device = torch.device('cpu')
    dataset = IROS(filename, device)
    print(dataset)

    # import matplotlib.pyplot as plt
    #
    # plt.plot(dataset.train_data[0][:,0])
    # plt.plot(dataset.train_data[1][:,0])
    # N = dataset.train_data[0].shape[0]
    # t = np.linspace(0, N*dataset.dt, N)
    # x = np.sin(dataset.w*t)
    # plt.plot(x)
    #
    # plt.show()
