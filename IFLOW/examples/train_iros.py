import os, sys

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from iflow.dataset import iros_dataset   #数据集有的
from torch.utils.data import DataLoader    #处理数据的，比较简单
from iflow.utils import makedirs   #在generic中
from iflow import model
from iflow.trainers import cycle_dynamics_train
from iflow.utils.generic import to_torch
import numpy as np
from iflow.visualization import visualize_vector_field, visualize_2d_generated_trj, save_vector_field,visualize_trajectories,visualize_latent_distribution
from iflow.test_measures.log_likelihood import cycle_log_likelihood


percentage = .99
batch_size = 100   #批量处理数据的
depth = 15
## optimization ##
lr = 0.001
weight_decay = 0.#学习率衰减因子
## training variables ##
nr_epochs = 500            #训练的次数吧
filename = 'dataSet_2D_rectangle'          #轨迹
save_folder = 'experiments'  #保存路径

dir_save = os.path.join(os.path.dirname(__file__),save_folder)   #路径拼接，os.path.dirname(__file__)获取当前运行脚本的绝对路径
makedirs(dir_save)  #就是创建目录的意思，可以自己改，如果目录已经存在，那么就不创建

######### GPU/ CPU #############
#device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')   #分配设备

#### Invertible Flow model #####可逆流模型
def main_layer(dim):
    return  model.ResNetCouplingLayer(dim)   #残差神经网络，模型定义好了，1*64 64*64 64*2的神经网络，激活函数为ReLu


def create_flow_seq(dim, depth):
    chain = []
    for i in range(depth):   #15次
        chain.append(main_layer(dim))
        chain.append(model.RandomPermutation(dim))   #把随机打乱的序列加进去
        chain.append(model.LULinear(dim))
    chain.append(main_layer(dim))  #那总共是十六层啊
    return model.SequentialFlow(chain)


if __name__ == '__main__':
    ########## Data Loading #########
    data = iros_dataset.IROS(filename=filename)
    dim = data.dim  #2
    T_period = (2*np.pi)/data.w  #6.28/6.41=0.9899对的，data.w平均角速度，估计是做微分同胚圆的，转一圈大概要4.95秒

    params = {'batch_size': batch_size, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)  #这个就是generic中的CycleDataset文件
    ######### Model #########
    lsd = model.LinearLimitCycle(dim, device, dt=data.dt, T_period=T_period)   #这应该可以看成是一个极限环，用来微分同胚的
    flow = create_flow_seq(dim, depth)    #2*16
    iflow = model.ContinuousDynamicFlow(dynamics=lsd, model=flow, dim=dim).to(device)   #这应该是微分同胚了
    ########## Optimization ################
    params = list(flow.parameters()) + list(lsd.parameters())
    optimizer = optim.Adamax(params, lr = lr, weight_decay= weight_decay)
    #######################################
    # print(dataloader)
    for i in range(nr_epochs):
        # Training
        #这里会把所有的点都跑一遍
        for local_x, local_y in dataloader:   #local_x和local_y[0]一样,都是从原轨迹中随机采样batch_size个点
            # print(local_y,'bbb')   #torch.Size([100, 2])torch.Size([100])torch.Size([100])[100, 2]
            # print(local_y[0].shape,'bbb')   #torch.Size([80, 2])torch.Size([100])torch.Size([100])
            dataloader.dataset.set_step()
            optimizer.zero_grad()   #梯度初始化为零
            loss = cycle_dynamics_train(iflow, local_x, local_y)   #求loss，就是2式中下面的
            loss.backward(retain_graph=True)   #反向求梯度
            optimizer.step()   #更新所有参数
        # print('参数开始')
        # print(params)
        # print('参数结束')
        ## Validation 审定##

        with torch.no_grad():
            iflow.eval()   #        return self.module_rref.rpc_sync().eval()  # type: ignore[operator, union-attr]

            # visualize_2d_generated_trj(data.train_data, iflow, device, fig_number=2)  #data.train_data这个轨迹就是归一化后的轨迹
            # visualize_vector_field(data.train_data, iflow, device, fig_number=3)
            # visualize_trajectories(data.train_data, iflow, device, fig_number=2)
            # visualize_latent_distribution(data.train_data, iflow, device, fig_number=3)
            # plt.show()

            step = 20
            trj = data.train_data[0]   #(1200, 2)
            trj_x0 = to_torch(trj[:-step,:], device)
            trj_x1 = to_torch(trj[step:,:], device)
            # print(data.train_phase_data[0])
            # print(np.array(data.train_phase_data[0]).shape)  #(1200,)
            phase = to_torch(data.train_phase_data[0][:-step], device)    #这里是算条件概率和终点概率的，好像是（9）（10）两个式子
            print('第{}张图片'.format(i))
            cycle_log_likelihood(trj_x0, trj_x1, phase, step, iflow, device)   #这里又算了一边最后的参数

            fig_name = filename + str(i) + '.png'
            save_filename = os.path.join(dir_save, fig_name)
            save_vector_field(data.train_data, iflow, device, save_fig=save_filename)
            # visualize_latent_distribution(data.train_data, iflow, device)
            # plt.show()












