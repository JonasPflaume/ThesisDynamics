import sys
import glob
import random
import tqdm
sys.path.append('..')

from torch import nn
import torch
from torch.optim import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt

from Common.utils import read_data, ButterWorthFilter, numerical_grad_nd
from pytorchDiff_infrastructure import batch_gen, MCGDiffSimulator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LossFunc = nn.MSELoss()
statespace_dim = 14
dt = 1e-3

class Residual(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.Bnet = nn.Sequential(
            nn.Linear(in_features, 17),
            nn.ReLU(),
            nn.Linear(17, 15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, out_features)
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
        print('Residual structure:', self)

    def forward(self, x):
        t = self.Bnet(x)
        return t
            
class ResMCGTrainer:
    def __init__(self, horizon, batch_size, lr):
        self.horizon = horizon
        self.batch_size = batch_size
        self.lr = lr
        self.res_net = Residual(statespace_dim+7, statespace_dim //2).to(device)

    def train(self, epoch):
        optimizer = Adam(self.res_net.parameters(), lr=self.lr)
        pbar = tqdm.tqdm(range(epoch))
        vali_loss = float('Inf')
        train_loss = float('Inf')
        for epoch in pbar:
            pbar.set_description('|| Vali loss %.3f || Train loss %.3f ||' % (vali_loss, train_loss))
            print(' ')

            count = 0
            train_loss = 0.
            for batch in batch_gen('../data/trajectories/*', self.batch_size, self.horizon):
                M, C, G, u, x0, Y = batch
                simulator = MCGDiffSimulator(M, C, G, u, x0, dt, self.horizon, adoint=True).to(device)
                solution = simulator.residual_simulate(self.res_net).to(device)
                L = self.loss(x0, solution, Y, u, M, C, G)
                optimizer.zero_grad()
                L.backward()
                optimizer.step()
                count += 1
                train_loss += L.item()
            
            vali_loss = self.validate(epoch)
            self.plot(solution, Y, epoch)
            train_loss /= count

    def loss(self, x0, solution, Y, u, M, C, G):
        # tracking loss
        L1 = LossFunc(solution, Y)

        X = torch.cat([torch.transpose(x0, 1, 2), Y], dim=1)
        # torque loss
        tau_pred = self.res_net(X[:, :-1, :])
        target = torch.squeeze(u[:, :-1, :], dim=3)
        L2 = LossFunc(tau_pred, target)
        return L1

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
            Xreal = np.concatenate([q_real, qDot_real, qDDot_inf], axis=1)

            Xreal = torch.from_numpy(Xreal).float().to(device)
            T_c = self.res_net(Xreal).detach().cpu().numpy()
            
            MLP = tau_MCG + T_c
            vali_loss = np.abs((tau_real - MLP)).sum() / len(tau_real)
            self.res_net.train()
            self.save_model(vali_loss, epoch)

        return vali_loss

    def plot(self, solution, target, epoch):
        one_traj = solution[0, :, :].detach().cpu().numpy()
        one_traj_target = target[0, :, :].detach().cpu().numpy()
        plt.figure(figsize=[12,8])
        for channel in range(14, 21):
            if channel == 5 or channel == 6:
                plt.subplot(4,2,channel-14+1,xlabel="0.001s", ylabel="rad/s")
            else:
                plt.subplot(4,2,channel-14+1, ylabel="rad/s")
            plt.plot(one_traj[:, channel], label='prediction')
            plt.plot(one_traj_target[:, channel], label='target')
            plt.grid()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)  
        plt.legend()
        plt.savefig('./ResMCG/Plot/valiPlot_ep_{0}.jpg'.format(epoch), dpi=200)
        plt.close()

    def save_model(self, valiLoss, epoch):
        Name = 'Ep{0}_valiLoss_{1:.3f}.pth'.format(epoch, valiLoss)
        Path = './ResMCG/Model/' + Name

        Name_sd = 'Ep{0}_valiLoss_{1:.3f}'.format(epoch, valiLoss)
        Path_sd = './ResMCG/Model/' + Name_sd
        torch.save(self.res_net, Path)
        torch.save(self.res_net.state_dict(), Path_sd)

if __name__ == '__main__':
    T = ResMCGTrainer(100, 500, 1e-3)
    T.train(400)