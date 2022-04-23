from typing import Tuple, Generator
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

# x = [x, xdot] position and velocity of slider

env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
del env

def GymGenerator(K:int, batch_size:int, nonlinear=True, batchNum=15) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' Traj generator, data from online ode solver
        input:
            K:          length of trajectories, gym < 200
            batch_size: generateed traj will be merged in one batch
            nonlinear:  If true use nonlinear ode
            batchNum:   one interation will retuen batchNum batches
    '''
    # total length one epoch
    simNum = K
    # create ode solver
    env = gym.make('Pendulum-v0')

    # create tensor
    State = np.zeros([batch_size, K, state_dim])
    Action = np.zeros([batch_size, K, action_dim])
    for _ in range(batchNum): # each epoch has batchNum batches
        for b_j in range(batch_size):
            # simulation start point
            x_0 = env.reset()
            res_x_sundials = [x_0]
            u_k = env.action_space.sample()
            res_u = [u_k]
            for i in range(simNum-1):
                u_k = env.action_space.sample()
                res_u.append(u_k)
                x_next, reward, done, info = env.step(u_k)
                res_x_sundials.append(x_next)
                x_0 = x_next

            res_x_sundials = np.concatenate(res_x_sundials, axis=0).reshape(-1, state_dim)
            res_u = np.concatenate(res_u, axis=0).reshape(-1, action_dim)

            if not __debug__:
                plt.figure(figsize=[10,5])
                plt.plot(res_x_sundials[:,0], label='sin(theta)')
                plt.plot(res_x_sundials[:,1], label='cos(theta)')
                plt.plot(res_x_sundials[:,2], label='thetaDot')
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
        hf = h5py.File(path + '/Data/GymTraj/GymTrajTrain.h5', 'w')
    elif TrainVali == 'vali':
        hf = h5py.File(path + '/Data/GymTraj/GymTrajVali.h5', 'w')
    else:
        print("give a purpose...")
    count = 0
    try:
        while True:
            for state, action in Generator(200, 8): ###### here for data length
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
            hf = h5py.File(path + '/Data/GymTraj/GymTrajTrain.h5', 'r')
        elif TrainVali == 'vali':
            hf = h5py.File(path + '/Data/GymTraj/GymTrajVali.h5', 'r')
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
    RecordTraj(GymGenerator, 5000, 'train')
    RecordTraj(GymGenerator, 56, 'vali')