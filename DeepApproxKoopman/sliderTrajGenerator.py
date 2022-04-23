from typing import Tuple, Generator
import h5py
import os

import matplotlib.pyplot as plt
import casadi
import numpy as np
import torch
from odeSysBuilder import SpringSlider_ode_solver

# x = [x, xdot] position and velocity of slider

state_dim = 2
action_dim = 1

def reference_traj(simNum:int) -> np.ndarray:
    # traj hyperparameters
    alpha = np.random.uniform(0.001, 0.02)
    xref1 = np.sin(np.arange(0,simNum) * alpha)[np.newaxis,:]
    alpha = np.random.uniform(0.001, 0.02)
    xref2 = np.sin(np.arange(0,simNum) * alpha)[np.newaxis,:] * 5
    alpha = np.random.uniform(0.001, 0.02)
    xref3 = np.sin(np.arange(0,simNum) * alpha)[np.newaxis,:] * 1.5
    alpha = np.random.uniform(0.001, 0.07)
    xref4 = np.sin(np.arange(0,simNum) * alpha)[np.newaxis,:] * 3
    xref = xref1 + xref2 + xref3 + xref4
    xref = np.repeat(xref, 2, axis=0)
    xref[1:,:] = 0
    return xref

def controller(xref: np.ndarray, xcurr: np.ndarray) -> np.ndarray:
    # sloppy PD controller
    xref = xref.reshape(state_dim, 1)
    xcurr = xcurr.reshape(state_dim, 1)
    u = (xcurr[0] - xref[0]) * 10 + (xcurr[1] - xref[1]) * 10
    u = np.array(u).reshape(1,1)
    return -u

def SliderGenerator(K:int, batch_size:int, nonlinear=True, batchNum=15) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' Traj generator, data from online ode solver
        input:
            K:          length of trajectories
            batch_size: generateed traj will be merged in one batch
            nonlinear:  If true use nonlinear ode
            batchNum:   one interation will retuen batchNum batches
    '''
    # total length one epoch
    simNum = K
    # create ode solver
    odeSolver = SpringSlider_ode_solver()

    # create tensor
    State = np.zeros([batch_size, K, state_dim])
    Action = np.zeros([batch_size, K, action_dim])
    for _ in range(batchNum): # each epoch has batchNum batches
        for b_j in range(batch_size):
            # tracking traj
            xref = reference_traj(simNum)
            # simulation start point
            pRand = np.random.uniform(0,5)
            pDRand = np.random.uniform(-3,3)
            x_0 = np.array([[pRand, pDRand]]).reshape(2,1)

            res_x_sundials = [x_0]
            u_k = np.zeros([1,1])
            res_u = [u_k]
            try:
                for i in range(simNum-1):
                    u_k = controller(xref[:,i], x_0)
                    res_u.append(u_k)
                    res_integrator = odeSolver(x0=x_0, p=u_k)
                    x_next = res_integrator['xf']
                    res_x_sundials.append(x_next)
                    x_0 = x_next.full()
            except:
                print("Integrator return unstable results.")
                return

            res_x_sundials = np.concatenate(res_x_sundials, axis=0).reshape(-1, state_dim)
            res_u = np.concatenate(res_u, axis=0).reshape(-1, action_dim)

            if not __debug__:
                plt.figure(figsize=[10,5])
                plt.plot(xref.T, 'cx')
                plt.plot(res_x_sundials[:,0], label='p')
                plt.plot(res_x_sundials[:,1], label='pd')
                #plt.plot(res_u, '-r')
                plt.legend()
                plt.show()
        

            state = res_x_sundials[0:K, :]
            action = res_u[0:K, :]
            State[b_j, :, :] = state
            Action[b_j, :, :] = action

        yield torch.from_numpy(State).float(), torch.from_numpy(Action).float()

def RecordTraj(Generator:Generator, datasetSize:int, TrainVali:str) -> None:
    ''' record the simulation results
    '''
    path = os.path.dirname(os.path.abspath(__file__))
    if TrainVali == 'train':
        hf = h5py.File(path + '/Data/SliderTraj/SliderTrajTrain.h5', 'w')
    elif TrainVali == 'vali':
        hf = h5py.File(path + '/Data/SliderTraj/SliderTrajVali.h5', 'w')
    else:
        print("give a purpose...")
    count = 0
    try:
        while True:
            for state, action in Generator(1400, 8):
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
            hf = h5py.File(path + '/Data/SliderTraj/SliderTrajTrain.h5', 'r')
        elif TrainVali == 'vali':
            hf = h5py.File(path + '/Data/SliderTraj/SliderTrajVali.h5', 'r')
    except:
        raise ValueError("No file under /Data/SliderTraj folder")
    
    State = np.zeros([batch_size, K, state_dim])
    Action = np.zeros([batch_size, K, action_dim])
    
    b_inx = 0
    for key in hf.keys():
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
    # use code to save data as H5 file from simulation
    RecordTraj(SliderGenerator, 1600, 'train')
    RecordTraj(SliderGenerator, 64, 'vali')