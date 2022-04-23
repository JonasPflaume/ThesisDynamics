import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import glob
from random import shuffle

from .utils import read_data
from .utils import ButterWorthFilter
from .utils import numerical_grad_nd

dof = 7

class Datasets(object):
    def __init__(self, path, horizon, overlapping, shuffle):
        self.overlapping = overlapping # split the trajectory by overlapping way
        self.shuffle = shuffle
        self.file_list = glob.glob(path + '/*')
        self.traj_num = len(self.file_list)
        self.horizon = horizon
        # return a dataset[num_batch][0 for state 1 for torque]
        self.dataset, self.time_list = self._init_dataset(self.file_list)
        self.end = len(self.dataset)
        self.start = 0
        self.q_dim = 7
        self.state_dim = self[0][0].shape[0] # first dim of augmented state
        try:
            self.mean, self.var = self.get_mean_var()
        except:
            print(self, ": no mean-var initialization!")
        self.action_size = 1 # how many torques are concatenated, tau_real, tau_measured

    def __iter__(self):
        return self
        
    def __next__(self):
        if self.start < self.end:
            item = self.dataset[self.start]
            self.start += 1
            return item
        else:
            self.start = 0
            raise StopIteration('Exausted list')

    def __getitem__(self, idx):
        if idx >= 0 and idx < self.end:
            return self.dataset[idx]
        else:
            print("Index out of bound!")
            
    def __len__(self):
        return len(self.dataset)

    def _init_dataset(self, file_list):
        # if overlapping, the split gap will be 1msc
        splitted_samples = []
        time_list = []
        for file_addr in file_list:
            trajectory = read_data(file_addr)
            if self.overlapping:
                splitted_traj, time = self._split_traj_overlap(trajectory)

            else:
                splitted_traj, time = self._split_traj(trajectory)

            if type(splitted_traj) == list:
                splitted_samples += splitted_traj
                time_list.append(time)

        if self.shuffle:
            shuffle(splitted_samples)

        return splitted_samples, np.concatenate(time_list)

    def _split_traj(self, traj, filter=True):
        if type(traj) != np.ndarray:
            return
        split_number = len(traj) // self.horizon
        time = traj[:, 0]
        q_real = traj[:, 1:1+7]
        q_ref = traj[:, 8:8+7]
        qDot_real = traj[:, 15:15+7]
        tau_real = traj[:, 29:29+7]
        tau_u_G = traj[:, 22:22+7] + traj[:,36:36+7] # u_cmd + gravity
        
        # get numerical derived qDDot
        if self.horizon == 1:
            qDot_real_filtered, _ = ButterWorthFilter(qDot_real, tau_real, time)
            qDDot_inf = numerical_grad_nd(qDot_real_filtered)
        if filter:
            tau_u_G, tau_real = ButterWorthFilter(tau_u_G, tau_real, time)
            
        splitted_traj = []
        for i in range(split_number):
            start_q = q_real[self.horizon*i, :] # start q
            start_q_dot = qDot_real[self.horizon*i, :] # start q_dot
            
            ref_q = q_ref[self.horizon*i:self.horizon*(i+1), :] # reference next horizon
            tau_r = tau_real[self.horizon*i:self.horizon*(i+1), :] # joints torque sent by PD controller
            aug_state = np.concatenate((start_q.reshape(1,-1), start_q_dot.reshape(1,-1)), axis=0)
            if self.horizon == 1: # markovian setting
                start_q_ddot = qDDot_inf[self.horizon*i, :]
                aug_state = np.concatenate((aug_state, start_q_ddot.reshape(1,-1)), axis=0)
            else:
                aug_state = np.concatenate((aug_state, ref_q), axis=0)
            splitted_traj.append((aug_state, tau_r))

        return splitted_traj, time

    def _split_traj_overlap(self, traj, filter=True):
        if type(traj) != np.ndarray:
            return
        split_number = len(traj) - self.horizon
        time = traj[:, 0]
        q_real = traj[:, 1:1+7]
        q_ref = traj[:, 8:8+7]
        qDot_real = traj[:, 15:15+7]
        tau_real = traj[:, 29:29+7]
        tau_u_G = traj[:, 22:22+7] + traj[:,36:36+7] # u_cmd + gravity
        if filter:
            tau_u_G, tau_real = ButterWorthFilter(tau_u_G, tau_real, time)
            
        splitted_traj = []
        for i in range(split_number):
            start_q = q_real[i, :] # start q
            start_q_dot = qDot_real[i, :] # start q_dot
            ref_q = q_ref[i:i+self.horizon, :] # reference next horizon
            pd_tau = tau_u_G[i:i+self.horizon, :] # joints torque sent by PD controller
            aug_state = np.concatenate((start_q.reshape(1,-1), start_q_dot.reshape(1,-1)), axis=0)
            aug_state = np.concatenate((aug_state, ref_q), axis=0)
            splitted_traj.append((aug_state, pd_tau))

        return splitted_traj, time

    def render_sample(self):
        idx = np.random.randint(0, self.end)
        joint = np.random.randint(0, self.q_dim)
        (state, action) = self[idx]
        t = np.arange(0,len(action))
        plt.plot(t, action[:,joint], '.', label='Torques')
        plt.legend()
        plt.show()
        
        t = np.arange(0,len(state))
        plt.plot(t[1:-1], state[2:,joint], '.', label="States")
        plt.scatter(t[0], state[0, joint], marker='x', label='Real measured q')
        plt.legend()
        plt.show()

    def get_mean_var(self):
        mean = 0
        var = 0
        count = 0
        for state, action in self.dataset:
            mean += action
            count += 1
        mean = mean.sum(axis=0)
        mean /= count*self.horizon
        count = 0
        for state, action in self.dataset:
            var += np.power(action - mean[np.newaxis,:], 2)
            count += 1
        var = var.sum(axis=0)
        var /= count * self.horizon
        var = np.power(var, 0.5)
        return mean, var

class NNDatasets(Datasets):

    def __init__(self, path, horizon, overlapping, shuffle):
        super(NNDatasets, self).__init__(path, horizon, overlapping, shuffle)
        self.action_size = 2
        # parameters for networks
        self.input_size = self.q_dim * (self.state_dim + 1) # 1 for the output torques
        self.output_size = self.q_dim                       # output residual dimention

    def _split_traj(self, traj, filter=True):
        if type(traj) != np.ndarray:
            return
        split_number = len(traj) // self.horizon
        time = traj[:, 0]
        q_real = traj[:, 1:1+7]
        q_ref = traj[:, 8:8+7]
        qDot_real = traj[:, 15:15+7]
        tau_real = traj[:, 29:29+7]
        tau_u_G = traj[:, 22:22+7] + traj[:,36:36+7] # u_cmd + gravity
        
        # get numerical derived qDDot
        if self.horizon == 1:
            qDot_real_filtered, _ = ButterWorthFilter(qDot_real, tau_real, time)
            qDDot_inf = numerical_grad_nd(qDot_real_filtered)
        if filter:
            tau_u_G, tau_real = ButterWorthFilter(tau_u_G, tau_real, time)

        # for evaluate
        self.tau_real = tau_real
        self.tau_u_G = tau_u_G
            
        splitted_traj = []
        for i in range(split_number):
            start_q = q_real[self.horizon*i, :] # start q
            start_q_dot = qDot_real[self.horizon*i, :] # start q_dot
            
#             ref_q = q_ref[self.horizon*i:self.horizon*(i+1), :] # reference next horizon
            pd_tau = tau_u_G[self.horizon*i:self.horizon*(i+1), :] # joints torque sent by PD controller
            mea_tau = tau_real[self.horizon*i:self.horizon*(i+1), :] # measured tau
            aug_state = np.concatenate((start_q.reshape(1,-1), start_q_dot.reshape(1,-1)), axis=0)

            tau_pd_real = np.concatenate((pd_tau, mea_tau), axis=0)
            splitted_traj.append((aug_state, tau_pd_real))

        return splitted_traj, time

    def get_mean_var(self):
        pass


class DatasetsNMResidual(Datasets):
    def __init__(self, path, horizon, overlapping, shuffle):
        super().__init__(path, horizon, overlapping, shuffle)

    def _split_traj(self, traj, filter=True):
        if type(traj) != np.ndarray:
            return
        split_number = len(traj) // self.horizon
        time = traj[:, 0]
        q_real = traj[:, 1:1+7]
        q_ref = traj[:, 8:8+7]
        qDot_real = traj[:, 15:15+7]
        tau_real = traj[:, 29:29+7]
        tau_u_G = traj[:, 22:22+7] + traj[:,36:36+7] # u_cmd + gravity
        
        # get numerical derived qDDot
        qDot_real_filtered, _ = ButterWorthFilter(qDot_real, tau_real, time)
        qDDot_inf = numerical_grad_nd(qDot_real_filtered)

        qDotRef_inf = numerical_grad_nd(q_ref)
        qDDotRef_inf = numerical_grad_nd(qDotRef_inf)

        if filter:
            tau_u_G, tau_real = ButterWorthFilter(tau_u_G, tau_real, time)
            
        splitted_traj = []
        for i in range(split_number):
            start_q = q_real[self.horizon*i, :] # start q
            start_q_dot = qDot_real[self.horizon*i, :] # start q_dot
#             start_q_ddot = qDDot_inf[self.horizon*i, :] # start q_ddot
            
            ref_q = q_ref[self.horizon*i:self.horizon*(i+1), :] # reference next horizon
            ref_qd = qDotRef_inf[self.horizon*i:self.horizon*(i+1), :] # reference next horizon
            ref_qdd = qDDotRef_inf[self.horizon*i:self.horizon*(i+1), :] # reference next horizon
            aug_state = np.concatenate((start_q.reshape(1,-1), start_q_dot.reshape(1,-1)), axis=0)
#             aug_state = np.concatenate((aug_state, start_q_ddot.reshape(1,-1)), axis=0)
            aug_state = np.concatenate((aug_state, ref_q), axis=0)
            aug_state = np.concatenate((aug_state, ref_qd), axis=0)
            aug_state = np.concatenate((aug_state, ref_qdd), axis=0)

            pd_tau = tau_u_G[self.horizon*i:self.horizon*(i+1), :] # joints torque sent by PD controller
            Mea_tau = tau_real[self.horizon*i:self.horizon*(i+1), :]
            splitted_traj.append((aug_state, pd_tau - Mea_tau))

        return splitted_traj, time
    
    
class DatasetsNMResidualBlockToOne(Datasets):
    ''' Helper class to split the trajectories
        output format:
            list of tuple, X[0] state, X[1] action
            X[0] with (5*horizon, 7), X[1] with (7,)
            X[0]: q_before, qd_before, q_ref, qd_ref, qdd_ref
            X[1]: residual error
    '''
    def __init__(self, path, horizon, overlapping, shuffle, subsample=1, filter=False):
        self.subsample = subsample
        self.filter = filter
        super().__init__(path, horizon, overlapping, shuffle)
        
    def _split_traj(self, traj):
        if type(traj) != np.ndarray:
            return
        split_number = len(traj)-1
        time = traj[:, 0]
        q_real = traj[:, 1:1+7]
        q_ref = traj[:, 8:8+7]
        qDot_real = traj[:, 15:15+7]
        tau_real = traj[:, 29:29+7]
        tau_u_G = traj[:, 22:22+7] + traj[:,36:36+7] # u_cmd + gravity
        
        # get numerical derived qDDot
        if self.filter:
            qDot_real_filtered, _ = ButterWorthFilter(qDot_real, tau_real, time)
        else:
            qDot_real_filtered = qDot_real
        qDDot_inf = numerical_grad_nd(qDot_real_filtered)

        qDotRef_inf = numerical_grad_nd(q_ref)
        qDDotRef_inf = numerical_grad_nd(qDotRef_inf)

        if self.filter:
            tau_u_G, tau_real = ButterWorthFilter(tau_u_G, tau_real, time)
        residual = tau_u_G[:-1,:] - tau_real[1:,:]
        residual = np.concatenate([residual, np.zeros([1, 7])])
            
        splitted_traj = []
        horizon = self.horizon
        for i in range(0, split_number, self.subsample):
            if i < horizon:
                q_history = np.concatenate([np.zeros([horizon-i-1, dof]), q_real[:i+1, :]])
                q_dot_history = np.concatenate([np.zeros([horizon-i-1, dof]), qDot_real_filtered[:i+1, :]])
                q_ddot_history = np.concatenate([np.zeros([horizon-i-1, dof]), qDDot_inf[:i+1, :]])
                # residual_history = np.concatenate([np.zeros([horizon-i-1, dof]), residual[:i+1, :]])
                
            else:
                q_history = q_real[i-horizon+1:i+1, :]
                q_dot_history = qDot_real_filtered[i-horizon+1:i+1, :]
                q_ddot_history = qDDot_inf[i-horizon+1:i+1, :]
                # residual_history = residual[i-horizon+1:i+1, :]

#             if split_number - i -1 < horizon:
#                 ref_q = np.concatenate([q_ref[i+1:i+1+horizon, :], np.zeros([horizon - split_number + i + 1, dof])])
#                 ref_qd = np.concatenate([qDotRef_inf[i+1:i+1+horizon, :], np.zeros([horizon - split_number + i + 1, dof])])
#                 ref_qdd = np.concatenate([qDDotRef_inf[i+1:i+1+horizon, :], np.zeros([horizon - split_number + i + 1, dof])])
#             else:
#                 ref_q = q_ref[i+1:i+1+horizon, :]
#                 ref_qd = qDotRef_inf[i+1:i+1+horizon, :]
#                 ref_qdd = qDDotRef_inf[i+1:i+1+horizon, :]
                
#             assert ref_q.shape[0] == ref_qd.shape[0] == ref_qdd.shape[0] == residual_history.shape[0] == horizon
            # assert q_history.shape[0] == q_dot_history.shape[0] == horizon
            
            aug_state = np.concatenate((q_history.reshape(self.horizon,-1), q_dot_history.reshape(self.horizon,-1)), axis=1)
            aug_state = np.concatenate((aug_state, q_ddot_history), axis=1)
#             aug_state = np.concatenate((aug_state, residual_history), axis=1)
            
#             aug_state = np.concatenate((aug_state, ref_q))
#             aug_state = np.concatenate((aug_state, ref_qd))
#             aug_state = np.concatenate((aug_state, ref_qdd))

            splitted_traj.append((aug_state, residual[i+1,:]))

        return splitted_traj, time
