import sys
sys.path.append('..')
import glob
import random
import tqdm

from torch import nn
import torch
import numpy as np
from Common.utils import read_data, ButterWorthFilter, numerical_grad_nd
from torchdiffeq import odeint, odeint_adjoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
statespace_dim = 14
dt = 1e-3

def batch_gen(root, batch_size, horizon, shuffle=True, sampling_gap=50):

    RootList = glob.glob(root)

    for r in RootList:
        dataset = read_data(r)
        time = dataset[:,0]
        q_real = dataset[:,1:1+7]
        q_ref = dataset[:,8:8+7]
        qDot_real = dataset[:,15:15+7]
        u_cmd = dataset[:, 22:22+7]
        tau_real = dataset[:,29:29+7]
        G = dataset[:,36:36+7]
        C = dataset[:, 43:43+7]
        M = dataset[:, 50:50+49]

        qDot_real_filtered, tau_real_filtered = ButterWorthFilter(qDot_real, tau_real, time)
        qDDot_inf = numerical_grad_nd(qDot_real_filtered) # numerical derivative to get joints acceleration.
        ## have to calculate MCG torques by hand
        tau_MCG = []
        for i in range(len(tau_real)):
            tau_i = M[i,:].reshape(7,7) @ qDDot_inf[i, :].reshape(7,1)
            tau_i += C[i,:].reshape(7,1)
            tau_i += G[i,:].reshape(7,1)
            tau_MCG.append(tau_i)

        tau_MCG = np.array(tau_MCG).reshape(len(tau_real), 7)

        Xreal = np.concatenate([q_real, qDot_real, qDDot_inf], axis=1)

        sampling_gap = horizon // 2 if horizon // 2 else 1

        if shuffle:
            sampling_start = random.randint(0, sampling_gap)
            dataset_idx = list(range(sampling_start, len(dataset) - horizon, sampling_gap))
        
            random.shuffle(dataset_idx)
        else:
            sampling_start = 0
            dataset_idx = list(range(sampling_start, len(dataset) - horizon, sampling_gap))

        while len(dataset_idx) > batch_size:
            M_b = np.zeros([batch_size, horizon+1, statespace_dim//2, statespace_dim//2])
            C_b = np.zeros([batch_size, horizon+1, statespace_dim//2, 1])
            G_b = np.zeros([batch_size, horizon+1, statespace_dim//2, 1])
            u_b = np.zeros([batch_size, horizon+1, statespace_dim//2, 1])
            x0 = np.zeros([batch_size, statespace_dim+7, 1])
            Y_b = np.zeros([batch_size, horizon, statespace_dim+7])

            for i, p in enumerate([dataset_idx.pop() for _ in range(batch_size)]):
                M_b[i, :, :, :] = M[p:p+horizon+1, :].reshape(horizon+1, statespace_dim//2, statespace_dim//2)
                C_b[i, :, :, :] = C[p:p+horizon+1, :].reshape(horizon+1, statespace_dim//2, 1)
                G_b[i, :, :, :] = G[p:p+horizon+1, :].reshape(horizon+1, statespace_dim//2, 1)
                u_b[i, :, :, :] = tau_real[p:p+horizon+1, :].reshape(horizon+1, statespace_dim//2, 1)
                x0[i, :, :] = Xreal[p, :].reshape(statespace_dim+7, 1)
                Y_b[i, :, :] = Xreal[p+1:p+1+horizon, :].reshape(horizon, statespace_dim+7)

            yield torch.from_numpy(M_b).float().to(device), torch.from_numpy(C_b).float().to(device), \
                torch.from_numpy(G_b).float().to(device), torch.from_numpy(u_b).float().to(device), \
                    torch.from_numpy(x0).float().to(device), torch.from_numpy(Y_b).float().to(device)

class MCGDiffSimulator(nn.Module):
    
    def __init__(self, M, C, G, u, x0, dt, horizon, adoint=False):
        super().__init__()

        self.odeint = odeint if adoint else odeint_adjoint

        self.M = M # (B,H+1,7,7)
        self.C = C
        self.G = G #(B,H+1,7,1)
        self.u = u #(B,H+1,7,1)
        self.x0 = x0 #(B,21,1)
        self.horizon = horizon # int
        assert self.M.shape[1] == horizon+1
        assert self.C.shape[1] == horizon+1
        assert self.G.shape[1] == horizon+1
        assert self.u.shape[1] == horizon+1
        self.batch_size = self.M.shape[0]
        self.dt = dt
        self.steps = torch.arange(0,horizon+1).to(device) # h 1 -> step2 2
        self.t = self.steps * self.dt
        self.curr_res_torque = torch.zeros(M.shape[0], statespace_dim//2, 1)
        self.curr_step = 0

    def forward(self, t, x):
        step = self.curr_step
        Minv = torch.inverse(self.M[:,step,:,:])
        C = self.C[:,step,:,:]
        G = self.G[:,step,:,:] #(B,7,1)
        u = self.u[:,step,:,:] + self.curr_res_torque

        Minv = torch.squeeze(Minv, dim=1)
        C = torch.squeeze(C, dim=1)
        G = torch.squeeze(G, dim=1)
        u = torch.squeeze(u, dim=1)

        xDot_0 = x[:,7:,:]
        temp = u - C - G
        xDot_1 = torch.bmm(Minv, temp)

        return torch.cat([xDot_0, xDot_1], dim=1)

    def simulate(self):
        solution = self.odeint(self, self.x0, self.t, atol=1e-8, rtol=1e-8, method='rk4')
        return solution

    def residual_simulate(self, res_net):
        x0 = self.x0#[:, :14, :]
        res = torch.zeros(self.batch_size, self.horizon, x0.shape[1]).to(device)
        for i in range(self.horizon):
            #e = torch.reshape(X[:, i, :], x0.shape) - x0   ###### Here to remove the error dependence
            #Input = torch.cat([e, x0], dim=1)
            Input = x0
            Input = torch.squeeze(Input, dim=2) # (B, 21)
            T_c = res_net(Input) #(B, 7)
            residual_torque = torch.unsqueeze(T_c, 2) # (B, 7, 1)
            self.curr_res_torque = residual_torque
            
            solution = self.odeint(self, x0[:, :14, :], self.t[i:i+2], atol=1e-10, rtol=1e-10, method='rk4') #(B,14,1)
            try:
                x0_1 = torch.squeeze(solution[1], dim=2)
            except:
                x0_1 = torch.squeeze(solution[0], dim=2)
            qDD = (x0_1[:, 7:] - Input[:, 7:14]) / dt
            x0 = torch.cat([x0_1, qDD], dim=1)
            res[:, i, :] = x0
            x0 = torch.unsqueeze(x0, dim=2)
            self.steps += 1
        return res