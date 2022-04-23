import torch as th
from torch import nn

#Input : (B, 35)
#Output: (B, 28)
class ForwardResNet(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(35, 33)
        self.L2 = nn.Linear(33, 31)
        self.L3 = nn.Linear(31, 29)
        self.L4 = nn.Linear(29, 28)
        self.act = nn.ReLU()

    def forward(self, xin):
        x = self.L1(xin)
        x = self.act(x)
        x = self.L2(x)
        x = self.act(x)
        x = self.L3(x)
        x = self.act(x)
        x = self.L4(x)
        dim_1 = xin.shape[0]

        lower = th.zeros(dim_1, 7, 7).to('cuda')
        for i, item in enumerate(th.tril_indices(7,7).T):
            lower[:, item[0], item[1]] = x[:, i]
        Mr = th.bmm(lower ,th.transpose(lower, 1, 2))
        return Mr