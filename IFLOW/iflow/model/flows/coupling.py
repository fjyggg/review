import torch
import torch.nn as nn

__all__ = ['CouplingLayer','ResNetCouplingLayer','MaskedCouplingLayer']


class CouplingLayer(nn.Module):
    """Used in 2D experiments."""

    def __init__(self, d, intermediate_dim=64, swap=False, nonlinearity='ReLu'):
        nn.Module.__init__(self)
        self.d = d - (d // 2)
        self.swap = swap
        if nonlinearity=='ReLu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif nonlinearity=='Tanh':
            self.nonlinearity = nn.Tanh()

        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim),  #（1，64)执行一个线性变换，y=wx+b,默认包含偏置层
            self.nonlinearity,    #这里表示用什么激活函数
            nn.Linear(intermediate_dim, intermediate_dim),   #（64，64）
            self.nonlinearity,    #这里表示用什么激活函数
            nn.Linear(intermediate_dim, (d - self.d) * 2),   #（64，2）
        )    #构建模型

    def forward(self, x, logpx=None, reverse=False):

        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_dim])
        scale = torch.sigmoid(s_t[:, :out_dim]) +0.01
        shift = s_t[:, out_dim:]

        logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = -logdetjac

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


class ResNetCouplingLayer(nn.Module):
    """Used in 2D experiments."""

    def __init__(self, d, intermediate_dim=64, swap=False, nonlinearity='ReLu'):  #非线性激活函数ReLu
        nn.Module.__init__(self)
        self.d = d - (d // 2)  #1
        self.swap = swap
        if nonlinearity == 'ReLu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif nonlinearity == 'Tanh':
            self.nonlinearity = nn.Tanh()

        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, intermediate_dim),
            self.nonlinearity,
            nn.Linear(intermediate_dim, (d - self.d) * 2),
        )

        nonlinearity_name = 'relu' if nonlinearity == 'ReLu' else 'tanh'
        nn.init.xavier_uniform_(self.net_s_t[0].weight,
                                gain=nn.init.calculate_gain(nonlinearity_name)/10)
        nn.init.xavier_uniform_(self.net_s_t[2].weight,
                                gain=nn.init.calculate_gain(nonlinearity_name)/10)
        nn.init.xavier_uniform_(self.net_s_t[4].weight,
                                gain=nn.init.calculate_gain('linear')/10)

    def forward(self, x, logpx=None, reverse=False):
        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        in_dim = self.d
        out_dim = x.shape[1] - self.d

        s_t = self.net_s_t(x[:, :in_dim])   #这里就是把第一列给拿出来了
        scale = torch.tanh(s_t[:, :out_dim]) + 1.0
        shift = s_t[:, out_dim:]

        logdetjac = torch.sum(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)
        #对的，以e为底的对数
        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = -logdetjac

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


class MaskedCouplingLayer(nn.Module):
    """Used in the tabular experiments."""

    def __init__(self, d, hidden_dims, mask_type='channel', swap=False):
        nn.Module.__init__(self)
        self.d = d
        self.register_buffer('mask', sample_mask(d, mask_type, swap).view(1, d))
        self.net_scale = build_net(d, hidden_dims, activation="tanh")
        self.net_shift = build_net(d, hidden_dims, activation="relu")

    def forward(self, x, logpx=None, reverse=False):

        scale = torch.exp(self.net_scale(x * self.mask))
        shift = self.net_shift(x * self.mask)

        masked_scale = scale * (1 - self.mask) + torch.ones_like(scale) * self.mask
        masked_shift = shift * (1 - self.mask)

        logdetjac = -torch.sum(torch.log(masked_scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y = x * masked_scale + masked_shift
            delta_logp = -logdetjac
        else:
            y = (x - masked_shift) / masked_scale
            delta_logp = logdetjac

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp


def sample_mask(dim, mask_type, swap):
    if mask_type == 'alternate':
        # Index-based masking in MAF paper.
        mask = torch.zeros(dim)
        mask[::2] = 1
        if swap:
            mask = 1 - mask
        return mask
    elif mask_type == 'channel':
        # Masking type used in Real NVP paper.
        mask = torch.zeros(dim)
        mask[:dim // 2] = 1
        if swap:
            mask = 1 - mask
        return mask
    else:
        raise ValueError('Unknown mask_type {}'.format(mask_type))


def build_net(input_dim, hidden_dims, activation="relu"):
    dims = (input_dim,) + tuple(hidden_dims) + (input_dim,)
    activation_modules = {"relu": nn.ReLU(inplace=True), "tanh": nn.Tanh()}

    chain = []
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        chain.append(nn.Linear(in_dim, out_dim))
        if i < len(hidden_dims):
            chain.append(activation_modules[activation])
    return nn.Sequential(*chain)
