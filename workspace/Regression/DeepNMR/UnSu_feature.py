import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_dataset_RR(D, horizon, feature=1):
    ''' build a simple block to one dataset
    '''
    datanum = len(D)
    dof = 7
    horizon = D.horizon
    Ytrain = np.zeros([datanum, dof])
    Xtrain = np.zeros([datanum, 14*horizon])
    for i, (state, action) in enumerate(D):
        Ytrain[i, :] = action
        Xtrain[i, :] = state.reshape(-1)

    return Ytrain, Xtrain

class Encoder(nn.Module):
    def __init__(self, horizon, dof_num, batch_size):
        super().__init__()
        dof = 7
        input_feature = horizon * dof_num * dof
        self.L1 = nn.Linear(input_feature, input_feature//2)
        self.L2 = nn.Linear(input_feature//2, input_feature//4)
        self.L3 = nn.Linear(input_feature//4, input_feature//2)
        self.L4 = nn.Linear(input_feature//2, input_feature)
        self.embed = input_feature//4
        self.batch_size = batch_size
        print("Embedding dimension: ", self.embed)

    def forward(self, x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        x = F.relu(x)
        x = self.L3(x)
        x = F.relu(x)
        x = self.L4(x)
        x = F.relu(x)
        return x


    @staticmethod
    def get_w_bias(layer, batch_size):
        w = layer.weight.t()
        w = w.repeat(batch_size, 1, 1)
        bias = layer.bias
        return w, bias

def batch_generator(X, batch_size):
    b_n = len(X) // batch_size
    input_feature = X.shape[1]
    x = np.zeros([batch_size, input_feature])
    counter = 0
    for item in X:
        x[counter, :] = item
        counter += 1
        if counter == batch_size:
            x = torch.from_numpy(x).float().to(device)
            yield x
            counter = 0
            x = np.zeros([batch_size, input_feature])
