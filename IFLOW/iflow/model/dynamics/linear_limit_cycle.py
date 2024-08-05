import numpy as np
import torch
import torch.nn as nn

from iflow.model.dynamics.generic_dynamic import LimitCycleDynamicModel

from iflow.utils.math import block_diag


class LinearLimitCycle(LimitCycleDynamicModel):
    ''' The Dynamic model is composed of two parts. A limit 动态模型由两部分组成。一个是极限'''
    def __init__(self, dim, device=None, dt = 0.01, requires_grad=True, T_to_stable=1., T_period= 1.):
        super().__init__(dim, device, dt, requires_grad)

        #### Deterministic Dynamics 确定性动力学####
        self.N_to_stable = int(T_to_stable/dt)   #100,时间是看外面给进来的
        self.N_to_period = int(T_period/dt)      #98,119.8
        # print(T_to_stable)
        # print(dt)
        # print(self.N_to_period)
        # print(self.N_to_stable)
        # print(T_period)

        self.T_period = T_period
        self.T_to_stable = T_to_stable

        ## Set the dynamics: N STEPS ##
        _x0 = 1.
        _xn = 0.01
        _adt_1 = _xn**(1/self.N_to_stable)   #0.9549
        _a = (_adt_1 - 1)/dt      #-4.500
        A = torch.eye(dim-2)*_a   #tensor([], size=(0, 0))

        ## Limit Cycle Velocities 极限周期速度##
        w = -(2*np.pi)/(self.N_to_period*dt)   #-6.41141357875468对的-0.5244728970934546

        self.r_des = nn.Parameter(torch.ones(1),requires_grad=False)  #参数加到模型中去，便于训练，1的张量tensor([1.])
        self.v_r = nn.Parameter(torch.ones(1)*10,requires_grad=False)  #10的张量tensor([10.])
        self.w = nn.Parameter(torch.ones(1)*w,requires_grad=False)
        self.A = nn.Parameter(A*10,requires_grad=False)

        ## Variance in Linear Dynamics线性动力学中的方差
        _std = 0.1
        self.log_var = nn.Parameter(torch.ones(dim)*np.log(_std ** 2)).to(device).requires_grad_(requires_grad)

    @property
    def var(self):
        return torch.diag(torch.exp(self.log_var))

    def forward(self, x, logpx=None, reverse=False):
        if not reverse:
            y = self.transform(x, reverse=reverse)
            log_abs_det_J = torch.log(y[:,0])
        else:
            y = self.transform(x, reverse=reverse)
            log_abs_det_J = -torch.log(x[:,0])

        if logpx is None:
            return y, logpx
        else:
            return y, logpx + log_abs_det_J.unsqueeze(1)

    def transform(self, x, reverse=False):
        y = x.clone()
        ## reverse to cartesian // forward to polar反向到笛卡尔，//向前到极地 ##
        if reverse:
            y[..., 0] = x[..., 0] * torch.cos(x[..., 1])
            y[..., 1] = x[..., 0] * torch.sin(x[..., 1])
        else:
            r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)  #这是极坐标的长lou
            theta = torch.atan2(x[..., 1], x[..., 0])
            y[..., 0] = r
            y[..., 1] = theta#返回的是极坐标的值
        return y

    def velocity(self, x):
        vel_r = -self.v_r*(x[:,0] - self.r_des)             #-10*(x[:,0]-1)
        vel_theta = self.w*torch.ones(x.shape[0]).to(x)     #
        if self.A.shape[0]!=0:
            vel_z = torch.matmul(self.A, x[:,2:].T).T
            return torch.cat((vel_r[:,None], vel_theta[:,None], vel_z),1)
        else:
            return torch.cat((vel_r[:,None], vel_theta[:,None]),1)   #这里返回的是线速度和角速度

    def first_Taylor_dyn(self, x):
        vel_r =  torch.cat(x.shape[0]*[-self.v_r.unsqueeze(0)[None,...]],0)
        vel_theta = torch.cat(x.shape[0]*[torch.zeros(1,1)[None,...]],0)
        vel_z = torch.cat(x.shape[0]*[self.A[None,...]],0)
        l_vel = [vel_r,vel_theta,vel_z]
        vel_mat = block_diag(l_vel)
        return vel_mat





