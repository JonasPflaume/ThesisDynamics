import torch
from torch import nn
import torch.nn.functional as F

class NetBase(nn.Module):
    def __init__(self, input_dim, output, dor, hidden_u):
        super(NetBase, self).__init__()
        hidden_l = hidden_u

        ingredients = []
        ingredients.append(nn.Linear(input_dim, hidden_l))
        ingredients.append(nn.Dropout(p=dor))
        ingredients.append(nn.LeakyReLU())
        ingredients.append(nn.Linear(hidden_l, hidden_l))
        ingredients.append(nn.Dropout(p=dor))
        ingredients.append(nn.LeakyReLU())
        ingredients.append(nn.Linear(hidden_l, output))
        
        self.net = nn.Sequential( *ingredients )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.net(x)
