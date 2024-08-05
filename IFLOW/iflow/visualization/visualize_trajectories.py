import torch
import matplotlib.pyplot as plt
import numpy as np

'''
以下是把轨迹复现给刻画出来，第一个是画单个维度的对比图，共两个
第二个是画二维的对比图，更加直观
'''
def visualize_trajectories(val_trajs, iflow, device, fig_number=1):
    dim = val_trajs[0].shape[1]
    plt.figure(fig_number, figsize=(20, int(10 * dim))).clf()
    fig, axs = plt.subplots(dim, 1, num=fig_number)   #在画板中创建子图

    for trj in val_trajs:
        n_trj = trj.shape[0]  #201行
        y0 = trj[0, :] #起始点
        y0 = torch.from_numpy(y0[None, :]).float().to(device)  #张成张量
        traj_pred = iflow.generate_trj( y0, T=n_trj)
        traj_pred = traj_pred.detach().cpu().numpy()

        for j in range(dim):
            axs[j].plot(trj[:,j],'b')
            axs[j].plot(traj_pred[:,j],'r')
    plt.draw()
    plt.pause(0.001)


def visualize_2d_generated_trj(val_trj, iflow, device, fig_number=1):  #这个已经搞懂了
    n_trj = len(val_trj)
    dim = val_trj[0].shape[-1]
    # print(np.array(val_trj).shape)#归一化后的轨迹(1, 201, 2)
    plt.figure(fig_number).clf()  #清除当前 figure 的所有axes，但是不关闭这个 window，所以能继续复用于其他的 plot。
    fig = plt.figure(figsize=(15, 15), num=fig_number)

    for i in range(len(val_trj)):  #1
        y_0 = torch.from_numpy(val_trj[i][:1, :]).float().to(device)  #[[-1.21447306 -1.22066082]]找到轨迹的起点
        trj_y = iflow.generate_trj(y_0, T=val_trj[i].shape[0])   #201，这个执行的是ciflow中的generate_trj
        trj_y = trj_y.detach().cpu().numpy()    #生成极限圆的轨迹

        plt.plot(trj_y[:,0], trj_y[:,1], 'g')
        plt.plot(val_trj[i][:,0], val_trj[i][:,1], 'b')   #画原来的轨迹
    plt.draw()
    plt.pause(0.001)


