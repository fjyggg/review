import numpy as np
import torch

from torch import nn
from torch.nn import functional as F, init

from iflow.model.flows.linear import Linear


class LULinear(Linear):
    """A linear transform where we parameterize the LU decomposition of the weights.一个线性变换，我们对权重的LU分解进行参数化处理"""

    def __init__(self, features, using_cache=False, identity_init=True, eps=1e-3):
        super().__init__(features, using_cache)

        self.eps = eps

        self.lower_indices = np.tril_indices(features, k=-1)  #下三角往右上覆盖一斜列
        self.upper_indices = np.triu_indices(features, k=1)   #上三角往坐下覆盖一斜列
        self.diag_indices = np.diag_indices(features)   #2,(array([0, 1]), array([0, 1]))

        n_triangular_entries = ((features - 1) * features) // 2  #1

        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))  # nn.Parameter可以看作是一个类型转换函数，将一个不可训练的类型 Tensor 转换成可以训练的类型 parameter ，并将这个 parameter 绑定到这个module 里面(net.parameter() 中就有这个绑定的 parameter，所以在参数优化的时候可以进行优化)，所以经过类型转换这个变量就变成了模型的一部分，成为了模型中根据训练可以改动的参数。使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.unconstrained_upper_diag = nn.Parameter(torch.zeros(features))
        self._initialize(identity_init)

    def _initialize(self, identity_init):
        init.zeros_(self.bias)   #Linear中给了偏置层，两个

        if identity_init:
            init.zeros_(self.lower_entries)
            init.zeros_(self.upper_entries)  #初始化
            constant = np.log(np.exp(1 - self.eps) - 1)
            init.constant_(self.unconstrained_upper_diag, constant)
        else:
            stdv = 1.0 / np.sqrt(self.features)
            init.uniform_(self.lower_entries, -stdv, stdv)
            init.uniform_(self.upper_entries, -stdv, stdv)
            init.uniform_(self.unconstrained_upper_diag, -stdv, stdv)

    def _create_lower_upper(self):
        lower = self.lower_entries.new_zeros(self.features, self.features)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        # The diagonal of L is taken to be all-ones without loss of generality.
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.

        upper = self.upper_entries.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper[self.diag_indices[0], self.diag_indices[1]] = self.upper_diag

        return lower, upper

    def forward_no_cache(self, x, logpx=None, reverse=False, context=None):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        if not reverse:
            y = F.linear(x, upper)
            y = F.linear(y, lower, self.bias)
            delta_logp = self.logabsdet() * x.new_ones(y.shape[0])
        else:
            y = x - self.bias
            y, _ = torch.triangular_solve(y.t(), lower, upper=False, unitriangular=True)
            y, _ = torch.triangular_solve(y, upper, upper=True, unitriangular=False)
            y = y.t()

            delta_logp = -self.logabsdet()
            delta_logp = delta_logp * x.new_ones(y.shape[0])

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp[:,None]

    def inverse_no_cache(self, inputs):
        """Cost:
            output = O(D^2N)
            logabsdet = O(D)
        where:
            D = num of features
            N = num of inputs
        """
        lower, upper = self._create_lower_upper()
        outputs = inputs - self.bias
        outputs, _ = torch.triangular_solve(outputs.t(), lower, upper=False, unitriangular=True)
        outputs, _ = torch.triangular_solve(outputs, upper, upper=True, unitriangular=False)
        outputs = outputs.t()

        logabsdet = -self.logabsdet()
        logabsdet = logabsdet * inputs.new_ones(outputs.shape[0])

        return outputs, logabsdet

    def weight(self):
        """Cost:
            weight = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        return lower @ upper

    def weight_inverse(self):
        """Cost:
            inverse = O(D^3)
        where:
            D = num of features
        """
        lower, upper = self._create_lower_upper()
        identity = torch.eye(self.features, self.features)
        lower_inverse, _ = torch.trtrs(identity, lower, upper=False, unitriangular=True)
        weight_inverse, _ = torch.trtrs(lower_inverse, upper, upper=True, unitriangular=False)
        return weight_inverse

    @property
    def upper_diag(self):
        return F.softplus(self.unconstrained_upper_diag) + self.eps

    def logabsdet(self):
        """Cost:
            logabsdet = O(D)
        where:
            D = num of features
        """
        return torch.sum(torch.log(self.upper_diag))