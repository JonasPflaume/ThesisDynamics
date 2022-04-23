####################################################################################
#### the helper function to generator batches from real panda trajectories files ###
####################################################################################
import numpy as np
import torch
import glob

from utils import read_data
import matplotlib.pyplot as plt
from typing import Tuple, Generator
import h5py
import os
import random

state_dim = 14
action_dim = 7

def OneTrajSplit(path:str) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    ''' get the seperated data from one *.dat file， timestamp x 99 text data
        return q_real, qDot_real, tau_cmd + Gravity
    '''
    data = read_data(path)

    time = data[:,0]
    q_m = data[:, 1:8]
    q_ref = data[:,8:15]
    qd_m = data[:, 15:22]
    aq = data[:, 22:29]
    res_u = data[:, 29:36]
    G = data[:, 36:43]
    C = data[:,43:50]
    M = data[:,50:99]
    qd_ref = data[:, 99:105]

    return q_m, qd_m, aq

def FileTrajGenerator(K:int, batch_size:int, TrainVali='train') -> Tuple[torch.Tensor, torch.Tensor]:
    ''' batch generator, split the trajectory into mini batch, traj length K!
        input: 
            TrainOrVali:'train' for training generator, 'vali' for validation generator
            K:          Steps number look ahead
            batch_size: Literal
        return:
            State:      concatenated torch.Tensor instance from X_{t} to X_{t+K}
            Action:     torques trajectory from U_{t} to U_{t+K}

        remark:
            This function will only read one traj. once at a time.
    '''
    
    if TrainVali == 'train':
        RootPath = glob.glob('./Data/PandaTraj/trainTraj/*')
    elif TrainVali == 'vali':
        RootPath = glob.glob('./Data/PandaTraj/valiTraj/*')

    dof = 7 # robot dof
    
    State = np.zeros([batch_size, K, 2*dof])
    Action = np.zeros([batch_size, K, dof])
    bCounter = 0
    for path in RootPath:
        q_real, qDot_real, u_G = OneTrajSplit(path)
        qt = q_real[:K,:]
        qdt = qDot_real[:K,:]
        state = np.concatenate([qt, qdt], axis=1)
        action = u_G[:K,:]
        State[bCounter,:,:] = state
        Action[bCounter,:,:] = action
        bCounter += 1
        print(bCounter)
        if bCounter == batch_size:
            yield torch.from_numpy(State).float(), torch.from_numpy(Action).float()
            State = np.zeros_like(State)
            Action = np.zeros_like(Action)
            bCounter = 0

def RecordTraj(Generator:Generator, datasetSize:int, TrainVali:str) -> None:
    ''' record the simulation results
    '''
    path = os.path.dirname(os.path.abspath(__file__))
    if TrainVali == 'train':
        hf = h5py.File(path + '/Data/PandaTraj/PandaTrajTrain.h5', 'w')
    elif TrainVali == 'vali':
        hf = h5py.File(path + '/Data/PandaTraj/PandaTrajVali.h5', 'w')
    else:
        print("give a purpose...")
    count = 0
    try:
        while True:
            bs=len(glob.glob('./Data/PandaTraj/trainTraj/*')) if TrainVali == 'train' else 1
            for state, action in Generator(11000, bs): #### Here needs rewrite ####
                state = state.cpu().detach().numpy()
                action = action.cpu().detach().numpy()
                size = action.shape[0]
                for i in range(size):
                    state_i = state[i, :, :]
                    action_i = action[i, :, :]
                    data_i = np.concatenate((state_i, action_i), axis=1)
                    hf.create_dataset('data_{0}'.format(count), data=data_i, dtype='float')
                    count += 1
                    if count >= datasetSize:
                        raise # break nested for loop
                    print("{0:.1f}% finished".format(100 * count/datasetSize))
    except:
        hf.close()
        print("{0:.1f}% finished".format(100 * count/datasetSize))
        print("{} data pairs was created.".format(count))

def RecordTrajGenerator(K:int, batch_size:int, TrainVali='train') -> Tuple[torch.Tensor, torch.Tensor]:
    ''' Generate batch from saved file from RecordTraj
    '''
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        if TrainVali == 'train':
            hf = h5py.File(path + '/Data/PandaTraj/PandaTrajTrain.h5', 'r')
        elif TrainVali == 'vali':
            hf = h5py.File(path + '/Data/PandaTraj/PandaTrajVali.h5', 'r')
    except:
        raise ValueError("No file under /Data/SliderTraj folder")
    
    State = np.zeros([batch_size, K, state_dim])
    Action = np.zeros([batch_size, K, action_dim])
    
    b_inx = 0
    KeyPool = list(hf.keys())
    random.shuffle(KeyPool)  # every epoch shuffle the dataset
    for key in KeyPool:
        data = hf.get(key)
        state = data[:K, :state_dim]
        action = data[:K, state_dim:]
        State[b_inx, :, :] = state
        Action[b_inx, :, :] = action
        b_inx += 1
        if b_inx == batch_size:
            yield torch.from_numpy(State).float(), torch.from_numpy(Action).float()
            b_inx = 0
            State = np.zeros([batch_size, K, state_dim])
            Action = np.zeros([batch_size, K, action_dim])

if __name__ == '__main__':
    RecordTraj(FileTrajGenerator, 95, 'train')
    RecordTraj(FileTrajGenerator, 5, 'vali')
    #import matplotlib.pyplot as plt
    #for i,j in RecordTrajGenerator(2100, 1, TrainVali='vali'):
    #    data = i.detach().cpu().numpy().squeeze()
    #    plt.plot(data[:,13])
    #    plt.show()