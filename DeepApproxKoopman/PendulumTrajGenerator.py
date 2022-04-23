from typing import Tuple, Generator
import h5py
import os

import matplotlib.pyplot as plt
import casadi
import numpy as np
import torch
from odeSysBuilder import pendulum_ode_solver

#
#                / 
#               /
#              /    pole: M = 1 kg
#             /     pole: R = 1 m
#            /
#     ______/_____  Cart: M = 1 kg
#    |            |
#    |____________|
#      O        O
#
# x = [p, pDot, theta, thetaDot]

state_dim = 4
action_dim = 1

def reference_traj(simNum:int) -> np.ndarray:
    # traj hyperparameters
    alpha = np.random.uniform(0.001, 0.02)
    xref1 = np.sin(np.arange(0,simNum) * alpha)[np.newaxis,:]
    alpha = np.random.uniform(0.001, 0.02)
    xref2 = np.sin(np.arange(0,simNum) * alpha)[np.newaxis,:] * 3
    alpha = np.random.uniform(0.001, 0.02)
    xref3 = np.sin(np.arange(0,simNum) * alpha)[np.newaxis,:] * 1.5
    alpha = np.random.uniform(0.001, 0.02)
    xref4 = np.sin(np.arange(0,simNum) * alpha)[np.newaxis,:] * 0.25
    alpha = np.random.uniform(0.001, 0.02)
    xref5 = np.sin(np.arange(0,simNum) * alpha)[np.newaxis,:] * (-2)
    xref = xref1 + xref2 + xref3 + xref4 +xref5
    xref = np.repeat(xref, 4, axis=0)
    xref[1:,:] = 0
    return xref

def useLQR(ang):
    if ang > 0:
        Normal = (ang % (2*np.pi)) * (180/np.pi)
        if Normal < 40 or Normal > 320:
            return True
        else:
            return False
    else:
        Normal = (ang % (-2*np.pi)) * (180/np.pi)
        if Normal < -320 or Normal > -40:
            return True
        else:
            return False

def controller(xref, x, swingupGain):
    R = 0.5
    m = 1.
    G = 9.81
    M = 1.
    l = 1.
    R = l/2
    C1 = 1 / (m + M)
    C2 = l*m / (m + M)
    xref = xref.reshape(4,1)
    x_ = x.reshape(-1,1)
    if useLQR(x[2]):
        if not __debug__:
            print("Enter Linear zone")
        # LQR gain
        Gain = - np.array([[1.0000, 2.6088, 52.9484, 16.5952]])
        u = - Gain @ (x_ - xref)
    else:
        # energy based nonlinear control
        if not __debug__:
            print("In NonLinear zone")
        # Self derived algorithm not working
        #Denominator = l * 4/3 - C2 * np.cos(x[2]) ** 2
        #A1 = G * np.sin(x[2]) / Denominator
        #A2 = -C1 * np.cos(x[2]) / Denominator
        #A3 = -C2 * x[3]**2 * np.sin(x[2]) * np.cos(x[2]) / Denominator
        #k = 2
        #u = ((-k*x[3] + 0.5*np.sin(x[2])*l*G)/(R**2) - A1 - A3) * 1/A2
        # Swing up algorithm comes from MIT underactuared robotics Lecture 3
        Ee = 0.5*m*R**2*x[3]**2 + 0.5*np.cos(x[2])*l*m*G - 0.5*l*m*G
        k = swingupGain
        u = k*x[3]*np.cos(x[2])*Ee
        # the control at state theta=pi is very small, try to compensate it
        if u < 1e-3 and u > 0:
            u += 1
        elif u > -1e-3 and u < 0:
            u -= 1
        u = np.array(u).reshape(1,1)
    return u

def pendulumGenerator(K:int, batch_size:int, nonlinear=True, batchNum=15) -> Tuple[torch.Tensor, torch.Tensor]:
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
    odeSolver = pendulum_ode_solver(nonlinear)

    # create tensor
    State = np.zeros([batch_size, K, state_dim])
    Action = np.zeros([batch_size, K, action_dim])
    for _ in range(batchNum): # each epoch has batchNum batches
        for b_j in range(batch_size):
            # tracking traj
            xref = reference_traj(simNum)
            # simulation start point
            pRand = np.random.uniform(-10,10)
            pDRand = np.random.uniform(-3,3)
            rRand = np.random.uniform(-2*np.pi,2*np.pi)
            rDRand = np.random.uniform(-3,3)
            x_0 = np.array([[pRand, pDRand, rRand, rDRand]]).reshape(4,1)

            res_x_sundials = [x_0]
            u_k = np.zeros([1,1])
            res_u = [u_k]
            SUGain = np.random.uniform(0.1, 1)
            if np.random.uniform(0,1) > 0.7:
                SUGain = 0
            try:
                for i in range(simNum-1):
                    u_k = controller(xref[:,i], x_0, SUGain)
                    res_u.append(u_k)
                    res_integrator = odeSolver(x0=x_0, p=u_k)
                    x_next = res_integrator['xf']
                    res_x_sundials.append(x_next)
                    x_0 = x_next.full()
            except:
                print("Integrator return unstable results.")
                return

            res_x_sundials = np.concatenate(res_x_sundials, axis=0).reshape(-1, 4)
            res_u = np.concatenate(res_u, axis=0).reshape(-1, 1)

            if not __debug__:
                plt.figure(figsize=[10,5])
                plt.plot(xref.T, 'cx')
                plt.plot(res_x_sundials[:,0], label='p')
                plt.plot(res_x_sundials[:,1], label='pd')
                plt.plot(res_x_sundials[:,2], label='theta')
                plt.plot(res_x_sundials[:,3], label='thetaDot')
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
        hf = h5py.File(path + '/Data/PendulumTraj/PendulumTraj.h5', 'w')
    elif TrainVali == 'vali':
        hf = h5py.File(path + '/Data/PendulumTraj/PendulumTrajVali.h5', 'w')
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
            hf = h5py.File(path + '/Data/PendulumTraj/PendulumTraj.h5', 'r')
        elif TrainVali == 'vali':
            hf = h5py.File(path + '/Data/PendulumTraj/PendulumTrajVali.h5', 'r')
    except:
        raise ValueError("No file under /Data/PendulumTraj folder")
    
    State = np.zeros([batch_size, K, state_dim])
    Action = np.zeros([batch_size, K, action_dim])
    
    b_inx = 0
    for key in hf.keys():
        data = hf.get(key)
        state = data[:K, :4]
        action = data[:K, 4:]
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
    RecordTraj(pendulumGenerator, 6400, 'train')
    RecordTraj(pendulumGenerator, 64, 'vali')