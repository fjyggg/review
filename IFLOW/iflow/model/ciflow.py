import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class ContinuousDynamicFlow(nn.Module):
    def __init__(self, model, dynamics, dim=2, context_dim=2, device=None, dt=0.01):
        super().__init__()
        self.device = device
        self.flow = model
        self.flow_backward, self.flow_forward = self.get_transforms(model)  #正逆微分同胚变换
        self.dynamics = dynamics

    def get_transforms(self, model):   #获取变换

        def sample_fn(z, logpz=None, context=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z,  reverse=True)

        def density_fn(x, logpx=None, context=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

        return sample_fn, density_fn

    def forward(self, yt, context=None):
        zero = torch.zeros(yt.shape[0], 1).to(yt)  #yt是我传进来的，先定义了一个和yt维度一样的
        xt, log_detjacobians = self.flow_forward(yt, zero, context=context)
        #zt, log_p = self.dynamics(xt,log_detjacobians)
        return xt, log_detjacobians

    def generate_trj(self, y0, T=100, noise=False, reverse=False):
        # print(T)201,y0是目标归一化轨迹的起点,y0[[-1.21447306 -1.22066082]]
        z0 = self.flow_forward(y0)   #这个值会变

        trj_z = self.dynamics.generate_trj(z0, T=T, reverse = reverse, noise = noise)  #自己调用自己，然后再反变换回去变成相对应的y,这里执行的是LimitCycleDynamicModel(nn.Module)中的generate_trj
        trj_y = self.flow_backward(trj_z[:, 0, :])#变过去的极限圆的轨迹
        return trj_y

    def evolve(self, y0, T=100, noise=False, reverse=False):
        print(reverse)
        print(noise)
        print(T)
        z0 = self.flow_forward(y0)#y0是两列堆叠的
        # z0 = z0.detach().cpu().numpy()
        # # plt.figure(figsize=(8, 8))
        # # plt.scatter(z0[:, 0], z0[:, 1])
        # # plt.show()
        # z0 = torch.Tensor(z0).float()
        z1 = self.dynamics.evolve(z0, T=T, reverse=reverse, noise=noise)
        z1 = z1.detach().cpu().numpy()
        print(z1)
        z1 = torch.Tensor(z1).float()
        y1 = self.flow_backward(z1)
        return y1
