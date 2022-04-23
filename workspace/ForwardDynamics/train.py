import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss
import torch as th
import sys
import random
import glob
sys.path.append('..')
from Common.utils import read_data, ButterWorthFilter, numerical_grad_nd
from network import ForwardResNet

np.set_printoptions(precision=3, suppress=True)
Loss = MSELoss()

train_root_path = "/home/jiayun/git/workspace/data/trajectories/*"
vali_root_path = "/home/jiayun/git/workspace/data/8_figure/*"

device = th.device('cuda' if th.cuda.is_available() else 'cpu')

def data_generator(path, batch_size):
    root_files = glob.glob(path)
    for file_i in root_files:
        dataset = read_data(file_i)
        time = dataset[:,0]

        indx = max(np.where(time<100)[0])
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
        u_cmd_filtered,_ = ButterWorthFilter(u_cmd, tau_real, time)
        qDDot_inf = numerical_grad_nd(qDot_real_filtered) # numerical derivative to get joints acceleration.

        qDotRef_inf = numerical_grad_nd(q_ref)
        qDDotRef_inf = numerical_grad_nd(qDotRef_inf)

        from sklearn.preprocessing import PolynomialFeatures
        def feature_tri_25(X):
            aug = X
            for k in range(1,26):
                aug = np.concatenate([aug, np.sin(k * np.pi * X)], axis=1)
                aug = np.concatenate([aug, np.cos(k * np.pi * X)], axis=1)
            poly = PolynomialFeatures(1)
            X = poly.fit_transform(aug)
            return X

        decoupling_beta = np.loadtxt('../Regression/Regression_weight/Tri_feature_decoupling.txt')
        X = np.concatenate([q_real, qDot_real], axis=1)
        fric_pred = []
        for i in range(7):
            x = np.concatenate([X[:, i:i+1], X[:, 7+i:i+8]], axis=1)
            fric_pred_i = feature_tri_25(x)@decoupling_beta[:, i:i+1]
            fric_pred.append(fric_pred_i)
            
        fric_pred = np.concatenate(fric_pred, axis=1)

        ###########################################
        pick_index = np.arange(0, len(time))
        random.shuffle(pick_index)
        length = (len(time)-batch_size) // batch_size

        for i in range(length):
            M_b = M_dat[pick_index[i*batch_size:(i+1)*batch_size], :].reshape(batch_size, 7, 7)
            C_b = C_dat[pick_index[i*batch_size:(i+1)*batch_size], :]
            G_b = G_dat[pick_index[i*batch_size:(i+1)*batch_size], :]
            tau_real_b = tau_real[pick_index[i*batch_size:(i+1)*batch_size], :]
            fric_b = fric_pred[pick_index[i*batch_size:(i+1)*batch_size], :]

            q_real_b = q_real[pick_index[i*batch_size:(i+1)*batch_size], :]
            qDot_real_b = qDot_real[pick_index[i*batch_size:(i+1)*batch_size], :]
            qDDot_inf_b = qDDot_inf[pick_index[i*batch_size:(i+1)*batch_size], :]
            q_ref_b = q_ref[pick_index[i*batch_size:(i+1)*batch_size], :]
            qDotRef_inf_b = qDotRef_inf[pick_index[i*batch_size:(i+1)*batch_size], :]
            qDDotRef_inf_b = qDDotRef_inf[pick_index[i*batch_size:(i+1)*batch_size], :]

            x_i = np.concatenate([q_real_b, qDot_real_b, q_ref_b, qDotRef_inf_b, qDDotRef_inf_b], axis=1)
            tau_real_b, M_b, C_b, G_b, fric_b, qDDot_inf_b = th.from_numpy(tau_real_b).float().to(device), th.from_numpy(M_b).float().to(device), \
                th.from_numpy(C_b).float().to(device), th.from_numpy(G_b).float().to(device), th.from_numpy(fric_b).float().to(device),\
                    th.from_numpy(qDDot_inf_b).float().to(device)
            param = (tau_real_b, M_b, C_b, G_b, fric_b, qDDot_inf_b)

            yield th.from_numpy(x_i).float().to(device), param


def Loss_calc(M_pred, tau_real_b, M_b, C_b, G_b, fric_b, qDDot_inf_b):
    qDDot_inf_b = th.unsqueeze(qDDot_inf_b, dim=2)
    rhs = th.bmm((M_pred + M_b), qDDot_inf_b)
    lhs = tau_real_b - C_b - G_b - fric_b
    lhs = th.unsqueeze(lhs, dim=2)
    L = Loss(rhs, lhs)
    return L

def train(train_root, vali_root, batch_size, lr, epoch_num):

    net = ForwardResNet().to(device)
    optimizer = Adam(net.parameters(), lr=lr)
    for epoch in range(epoch_num):
        total_L = 0
        for x_i, param in data_generator(train_root, batch_size):
            tau_real_b, M_b, C_b, G_b, fric_b, qDDot_inf_b = param
            M_pred = net(x_i)
            optimizer.zero_grad()
            L = Loss_calc(M_pred, tau_real_b, M_b, C_b, G_b, fric_b, qDDot_inf_b)
            L.backward()
            optimizer.step()
            total_L += L.item()
        vali_L = validate(vali_root, net, batch_size)
        print("Training Loss: {}, Validate Loss: {}".format(total_L, vali_L))

def validate(root, net, batch_size):
    net.eval()
    vali_L = 0
    with th.no_grad():
        for x_i, param in data_generator(root, batch_size):
            tau_real_b, M_b, C_b, G_b, fric_b, qDDot_inf_b = param
            M_pred = net(x_i)
            L = Loss_calc(M_pred, tau_real_b, M_b, C_b, G_b, fric_b, qDDot_inf_b)
            vali_L += L.item()
    net.train()
    return vali_L

if __name__ == "__main__":
    train(train_root_path, vali_root_path, 64, 1e-3, 20)