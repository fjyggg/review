import torch
import matplotlib.pyplot as plt
import time

def goto_dynamics_train(iflow, x, y):
    ## Separate Data ##
    y0 = x
    y1 = y[0]
    step = y[1][0]
    yN = y[2]
    t = y[3]
    ## Evolve dynamics backwards ##
    x_0, log_det_J_x0 = iflow(y0)
    x_1, log_det_J_x1 = iflow(y1)
    p_x0_x1 = iflow.dynamics.conditional_distribution(x_1, T=step, reverse=True)
    log_p_z0 = p_x0_x1.log_prob(x_0)
    loss_trj = log_p_z0 + log_det_J_x0.squeeze()

    ########## Last step #############
    yN = yN[:1,:]
    x_n, log_det_J_xn = iflow(yN)
    log_p_xn = iflow.dynamics.compute_stable_log_px(x_n)
    loss_end = log_p_xn + log_det_J_xn.squeeze()

    #### Complete Loss is composed between the stable loss and the trajectory loss
    loss_total = torch.mean(loss_trj) + torch.mean(loss_end)
    return -loss_total


def cycle_dynamics_train(iflow, x, y):   #x和y都是原轨迹中随机的一百个点
    ## Separate Data ##
    y0 = x
    y1 = y[0]

    # y0 = y0.detach().cpu().numpy()
    # y1 = y1.detach().cpu().numpy()
    # plt.figure(figsize=(8,8))
    # plt.scatter(y0[:,0],y0[:,1])
    # plt.scatter(y1[:,0],y1[:,1])
    # # plt.show()
    # y0 = torch.Tensor(y0).float()
    # y1 = torch.Tensor(y1).float()

    step = y[1][0]  #时间步是19
    phase = y[2]
    ## Evolve dynamics forward向前发展的动力 ##
    x_0, log_det_J_x0 = iflow(y0)  #这里在算伪代码中的tau_z和雅可比矩阵
    x_1, log_det_J_x1 = iflow(y1)

    # x_01 = x_0.detach().cpu().numpy()
    # x_11 = x_1.detach().cpu().numpy()
    # plt.figure(figsize=(8,8))
    # plt.scatter(x_01[:,0],x_01[:,1])
    # plt.scatter(x_11[:,0],x_11[:,1])
    # plt.show()
    # time.sleep(0.5)

    ### Forward Conditioning ###
    log_p_z1 = iflow.dynamics.cartesian_cond_log_prob(x_0, x_1, T=step)
    log_trj = log_p_z1 + log_det_J_x1.squeeze()   #这个算的是十式，不过都加了log方便计算

    ### Stable Point ###
    log_p_z0 = iflow.dynamics.stable_log_prob(x_0, ref_phase=phase)
    log_stable = log_p_z0 + log_det_J_x0.squeeze()   #这个算的是九式，不过都加了log方便计算，直接看伪代码的最后划红线的地方公式

    log_total = torch.mean(log_stable) + torch.mean(log_trj)
    return -log_total



