import sys
sys.path.append('..')
from Common.utils import read_data, ButterWorthFilter, numerical_grad_nd
import glob
import numpy as np
import torch
root = glob.glob('../data/trajectories/*')

def GetRoughData(path=root):
    ## Dataset preparation
    tau_list = []
    x_list = []
    for path in root:
        dataset = read_data(path)

        time = dataset[:,0]

        indx = max(np.where(time<38)[0])
        time = time[:indx]
        q_real = dataset[:indx,1:1+7]
        q_ref = dataset[:indx,8:8+7]
        qDot_real = dataset[:indx,15:15+7]
        u_cmd = dataset[:indx, 22:22+7]
        tau_real = dataset[:indx,29:29+7]
        Gra = dataset[:indx,36:36+7]
        u_G = u_cmd + Gra
        G_dat = dataset[:indx,36:36+7]
        C_dat = dataset[:indx, 43:43+7]
        M_dat = dataset[:indx, 50:50+49]

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
        x_i = np.concatenate([q_real, qDot_real, qDDot_inf], axis=1)
        x_list.append(x_i)
        tau_list.append(tau_real)
    x = np.concatenate(x_list)
    tau = np.concatenate(tau_list)
    return x, tau
