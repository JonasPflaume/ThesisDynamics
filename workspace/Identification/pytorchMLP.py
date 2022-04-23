import sys
import glob
import random
import tqdm
sys.path.append('..')

from torch import nn
import torch
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
from Common.utils import read_data, ButterWorthFilter, numerical_grad_nd

from torchdiffeq import odeint, odeint_adjoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LossFunc = nn.MSELoss()
statespace_dim = 14
dt = 1e-3

class ResidualNormal(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features, 28),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(28, 14),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(14, out_features),
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print('Residual structure:', self)

    def forward(self, x):
        return self.net(x)

def batch_gen_normal(root, batch_size, shuffle=True, sampling_gap=50):
    RootList = glob.glob(root)
    for r in RootList:
        dataset = read_data(r)
        time = dataset[:,0]
        q_real = dataset[:,1:1+7]
        q_ref = dataset[:,8:8+7]
        qDot_real = dataset[:,15:15+7]
        u_cmd = dataset[:, 22:22+7]
        tau_real = dataset[:,29:29+7]
        G_dat = dataset[:,36:36+7]
        C_dat = dataset[:, 43:43+7]
        M_dat = dataset[:, 50:50+49]
        Xreal = np.concatenate([q_real, qDot_real], axis=1)

        qDot_real_filtered, tau_real_filtered = ButterWorthFilter(qDot_real, tau_real, time)
        qDDot_inf = numerical_grad_nd(qDot_real_filtered) # numerical derivative to get joints acceleration.

        ## have to calculate MCG torques by hand
        tau_MCG = []
        for i in range(len(tau_real)):
            tau_i = M_dat[i,:].reshape(7,7) @ qDDot_inf[i, :].reshape(7,1)
            tau_i += C_dat[i,:].reshape(7,1)
            tau_i += G_dat[i,:].reshape(7,1)
            tau_MCG.append(tau_i)
        tau_MCG = np.array(tau_MCG).reshape(len(tau_real), 7)

        Y = tau_real - tau_MCG

        if shuffle:
            dataset_idx = list(range(0, len(dataset)))
            random.shuffle(dataset_idx)
        else:
            dataset_idx = list(range(0, len(dataset)))

        while len(dataset_idx) > batch_size:
            X_b = np.zeros([batch_size, statespace_dim, 1])
            Y_b = np.zeros([batch_size, statespace_dim // 2])
            for i, p in enumerate([dataset_idx.pop() for _ in range(batch_size)]):
                X_b[i, :, :] = Xreal[p, :].reshape(statespace_dim, 1)
                Y_b[i, :] = Y[p, :].reshape(statespace_dim//2, )
            yield torch.from_numpy(X_b).float().to(device), torch.from_numpy(Y_b).float().to(device)


class ResTrainer:
    def __init__(self, batch_size, lr):
        self.batch_size = batch_size
        self.lr = lr
        self.res_net = ResidualNormal(statespace_dim, statespace_dim //2).to(device)

    def train(self, epoch):
        optimizer = Adam(self.res_net.parameters(), lr=self.lr)
        pbar = tqdm.tqdm(range(epoch))
        vali_loss = float('Inf')
        train_loss = float('Inf')
        for epoch in pbar:
            pbar.set_description('|| Vali loss %.4f || Train loss %.4f ||' % (vali_loss, train_loss))

            count = 0
            train_loss = 0.
            for batch in batch_gen_normal('../data/trajectories/*', self.batch_size):
                X_b, Y_b = batch
                X_b = torch.squeeze(X_b, dim=2)
                pred = self.res_net(X_b)
                L = self.loss(pred, Y_b)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
                count += 1
                train_loss += L.item()
            
            vali_loss = self.validate(epoch)
            train_loss /= count

    def validate(self, epoch):
        with torch.no_grad():
            self.res_net.eval()
            path = '../data/test_trajectory/traj17_z.panda.dat'
            dataset = read_data(path)

            time = dataset[:,0]

            q_real = dataset[:,1:1+7]
            q_ref = dataset[:,8:8+7]
            qDot_real = dataset[:,15:15+7]
            u_cmd = dataset[:, 22:22+7]
            tau_real = dataset[:,29:29+7]
            G_dat = dataset[:,36:36+7]
            C_dat = dataset[:, 43:43+7]
            M_dat = dataset[:, 50:50+49]

            u_G = u_cmd + G_dat

            qDot_real_filtered, tau_real_filtered = ButterWorthFilter(qDot_real, tau_real, time)
            qDot_real_filtered, u_G_filtered = ButterWorthFilter(qDot_real, u_G, time)

            qDDot_inf = numerical_grad_nd(qDot_real_filtered) # numerical derivative to get joints acceleration.
            ## have to calculate MCG torques by hand
            tau_MCG = []
            for i in range(len(tau_real)):
                tau_i = M_dat[i,:].reshape(7,7) @ qDDot_inf[i, :].reshape(7,1)
                tau_i += C_dat[i,:].reshape(7,1)
                tau_i += G_dat[i,:].reshape(7,1)
                tau_MCG.append(tau_i)
                
            tau_MCG = np.array(tau_MCG).reshape(len(tau_real), 7)
            Xreal = np.concatenate([q_real, qDot_real], axis=1)

            Xreal = torch.from_numpy(Xreal).float().to(device)
            T_res = self.res_net(Xreal)

            T_res = T_res.detach().cpu().numpy()
            
            MLP = tau_MCG  + T_res
            vali_loss = np.linalg.norm(tau_real - MLP)
            self.res_net.train()
            self.save_model(vali_loss, epoch)

        return vali_loss

    def loss(self, Xres, Xr):
        return LossFunc(Xres, Xr)

    def save_model(self, valiLoss, epoch):
        Name = 'Ep{0}_valiLoss_{1:.3f}.pth'.format(epoch, valiLoss)
        Path = './ResMCG/Model/' + Name

        Name_sd = 'Ep{0}_valiLoss_{1:.3f}'.format(epoch, valiLoss)
        Path_sd = './ResMCG/Model/' + Name_sd
        torch.save(self.res_net, Path)
        torch.save(self.res_net.state_dict(), Path_sd)

if __name__ == '__main__':
    T = ResTrainer(128, 1e-3)
    T.train(400)