'''
Pendulum training results evaluation
Swing up using MPC in lifting space
'''

import multiprocessing
from multiprocessing import Manager
from collections import deque
import random
import os

from networks import ABC, Decoder, Encoder
from train import Trainer
from odeSysBuilder import pendulum_ode_solver
from odeSysBuilder import SpringSlider_ode_solver
from loss import LossFunc
from sliderTrajGenerator import reference_traj

import matplotlib.pyplot as plt
import casadi
import torch.optim as optim
from torch.optim import Adam
import torch
import numpy as np
import gym
import gym.monitoring as monitor
import h5py
import time

from Controller import MPC, FiniteLQR, KF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OnlineTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        if args.whichModel != None:
            modelpath = './Models/TrainedOfflineModel/' + args.whichModel
        # when read model, then the lifting dim is fixed
            self.ABC = torch.load(modelpath + '/ABC.pth')
            self.EnNet = torch.load(modelpath + '/EnNet.pth')
        self.ABC.cpu().train()
        self.EnNet.cpu().train()
        
        self.LossFunc = LossFunc()
        self.buffer = deque(maxlen=args.maxlen)
        self.meomeryManager = Manager().list()
        self.Pnum = multiprocessing.cpu_count()
        self.simNum = args.simNum
        self.updateNum = args.updateNum # 1 for 12 traj
        self.trainNum = args.trainNum
        self.whichOde = args.whichOde

        self.whichController = args.whichController
        with torch.no_grad():
            ABCweight = list(self.ABC.named_parameters())
            if self.whichController == 'MPC':
                self.MPC = MPC(ABCweight)
                if self.whichOde == 'slider':
                    self.odeSolver = SpringSlider_ode_solver()
                    self.ref = np.zeros([self.MPC.N, self.MPC.nxx])
                    self.ref[:,0] = 1 # step responce
                elif self.whichOde == 'pendulum':
                    self.odeSolver = pendulum_ode_solver()
                    self.ref = np.zeros([self.MPC.N, self.MPC.nxx])
                elif self.whichOde == 'gym':
                    self.odeSolver = [gym.make('Pendulum-v0') for _ in range(self.Pnum)]
                    self.ref = np.zeros([self.MPC.N, self.MPC.nxx])
                    self.ref[:,0] = 1 # cos theta = 1
                    self.ref[:,1] = 0 # sin theta = 0
            elif self.whichController == 'LQR':
                self.LQR = FiniteLQR(ABCweight, self.simNum)
                self.KF = KF(ABCweight)
                if self.whichOde == 'slider':
                    self.odeSolver = SpringSlider_ode_solver()
                    self.ref = np.zeros([self.LQR.nxx, 1])
                    self.ref[0,:] = 1 # step responce
                elif self.whichOde == 'pendulum':
                    self.odeSolver = pendulum_ode_solver()
                    self.ref = np.zeros([self.LQR.nxx, 1])
                elif self.whichOde == 'gym':
                    self.odeSolver = [gym.make('Pendulum-v0') for _ in range(self.Pnum)]
                    self.ref = np.zeros([self.LQR.nxx, 1])
                    self.ref[0,:] = 1
                    
    def simuOnce(self, Pi:int, x_0, video=False):
        ''' simulate the system once with control
            the main loop of conduct control
            the collected data will fill the meomeryManager
        '''
        simNum = self.simNum
        # initialize the reference trajectory for different controller
        if self.whichController == 'MPC':
            Ctrl = self.MPC
            xref = self.ref
        elif self.whichController == 'LQR':
            Ctrl = self.LQR
            xref = self.ref.reshape(self.LQR.nxx, 1)
            xref = torch.from_numpy(xref).float().cpu()
            xref = self.EnNet(xref.T)
            xref = xref.cpu().detach().numpy()
        
        #xref_tensor = torch.from_numpy(xref).float()
        #xref_tensor_lifting = self.EnNet(xref_tensor.T)
        #xref_lifting = xref_tensor_lifting.detach().numpy()
        if self.whichOde == 'gym':
            x_0 = self.odeSolver[Pi].reset()
            x_0 = x_0[:, np.newaxis]

        res_x_sundials = [x_0]
        x_0_tensor = torch.from_numpy(x_0.T).float()
        with torch.no_grad():
            x_0_tensor_lifting = self.EnNet(x_0_tensor)
        x_0_lifting = x_0_tensor_lifting.cpu().detach().numpy()
        u_k = np.zeros([self.control_dim, 1])
        res_u = [u_k]

        # record video for gym env
        if video:
            curr_time = int(time.time())
            video_path = './Plots/GymVideo/{}.mp4'.format(curr_time % 10000)
            video_recorder = monitor.VideoRecorder(
                        self.odeSolver[Pi], video_path, enabled=video_path is not None)
        ##### main loop #####
        for i in range(simNum-1):
            if video:
                self.odeSolver[Pi].unwrapped.render()
                video_recorder.capture_frame()
            
            x_0_lifting.reshape(Ctrl.nx, 1)
            u_k = Ctrl(x_0_lifting.T, xref, i)
            #if self.whichController == 'LQR':
                #pass
                # KF predict
                #lxpred, ppred = self.KF.predict(x_0_lifting, u_k)
                # KF gain
                #KG = self.KF.K_G()
            
            if self.whichOde == 'gym':
                if self.whichController == 'LQR':
                    u_k = np.clip(u_k, -2., 2.)
                x_next, _, done, _ = self.odeSolver[Pi].step(u_k)
                x_0 = x_next
            else:
                res_integrator = self.odeSolver(x0=x_0, p=u_k)
                x_next = res_integrator['xf']
                x_0 = x_next.full()
            res_x_sundials.append(x_next)
            res_u.append(u_k)
            x_0_tensor = torch.from_numpy(x_0.T).float()
            if self.whichController == 'LQR': # kalman filter not working
                # KF update
                #lx_new = self.KF.update(lxpred, KG, x_0) # x_next as measurment
                #x_0_lifting = lx_new

                with torch.no_grad():
                    x_0_tensor_lifting = self.EnNet(x_0_tensor)
                x_0_lifting = x_0_tensor_lifting.cpu().detach().numpy()
                
            elif self.whichController == 'MPC':
                with torch.no_grad():
                    x_0_tensor_lifting = self.EnNet(x_0_tensor)
                x_0_lifting = x_0_tensor_lifting.cpu().detach().numpy()
        # close the gym env video recorder
        if video:
            video_recorder.close()
            video_recorder.enabled = False
        res_x_sundials = np.concatenate(res_x_sundials, axis=0).reshape(-1, Ctrl.nxx)
        res_u = np.concatenate(res_u, axis=0).reshape(-1, 1)
        OneTraj = np.concatenate([res_x_sundials, res_u], axis=1)
        self.meomeryManager.append(OneTraj)

    def SampleTrajs(self):
        ''' Multiprocessing batch generation
        '''
        for Pi in range(self.Pnum):
            if self.state_dim == 4:
                pRand = np.random.uniform(-10,10)
                pDRand = np.random.uniform(-1,1)
                rRand = np.pi
                rDRand = 0
                x_0 = np.array([[pRand, pDRand, rRand, rDRand]]).reshape(self.state_dim,1)
            elif self.state_dim == 2:
                pRand = np.random.uniform(-2,2)
                pDRand = np.random.uniform(-1,1)
                x_0 = np.array([[pRand, pDRand]]).reshape(self.state_dim,1)
            elif self.state_dim == 3:
                x_0 = self.odeSolver[Pi].reset()
            exec('P{0} = multiprocessing.Process(target=self.simuOnce, args=(Pi, x_0))'.format(Pi))
        for i in range(self.Pnum):
            exec('P{}.start()'.format(i))
        for i in range(self.Pnum):
            exec('P{}.join()'.format(i))

        print('Sampling traj ... ')
        for n in self.meomeryManager:
            self.buffer.append(n)

        self.meomeryManager = Manager().list()

    def plot(self, step: int):
        ''' update nn using online data
            parameter:
                        step: the current training epoch
        '''
        lastTraj = self.buffer[-1]
        res_x = lastTraj[:,:self.state_dim]
        action = lastTraj[:,self.state_dim:]
        plt.figure(figsize=[10,10])
        plt.subplot(211)
        if self.state_dim == 4:
            plt.plot(res_x[:,0], label='p')
            plt.plot(res_x[:,1], label='pd')
            plt.plot(res_x[:,2], label='theta')
            plt.plot(res_x[:,3], label='thetaDot')
        elif self.state_dim == 2:
            plt.plot(res_x[:,0], label='p')
            plt.plot(res_x[:,1], label='pd')
            plt.plot(np.ones_like(res_x[:,1]), '-r', label='reference')
        elif self.state_dim == 3:
            plt.plot(np.ones_like(res_x[:,0]), '-r', label='reference')
            plt.plot(res_x[:,0], label='cos(theta)')
            plt.plot(res_x[:,1], label='sin(theta)')
            plt.plot(res_x[:,2], label='thetaDot')
        plt.grid()
        plt.legend()
        plt.subplot(212)
        plt.plot(action, label='u')
        plt.grid()
        #plt.plot(res_u, '-r')
        plt.legend()
        plt.savefig('./Plots/OnlineTrainingPlots/step_{}.jpg'.format(step), dpi=200)
        plt.close()

    def BufferBatchGenerator(self,  TrainVali='train'):
        ''' the batch generator from replay buffer
            parameters:
                        TrainVali:  choose the generator is for training or validation
        '''
        K = self.buffer[0].shape[0]
        State = torch.zeros(self.batch, K, self.state_dim)
        Action = torch.zeros(self.batch, K, self.control_dim)
        counter = 0

        for traj in self.buffer:
            state = traj[:, :self.state_dim]
            action = traj[:, self.state_dim:]
            state = torch.from_numpy(state).float().to(device)
            action = torch.from_numpy(action).float().to(device)

            Action[counter, :, :] = action
            State[counter, :, :] = state
            counter += 1
            if counter == self.batch:
                yield State, Action
                State = torch.zeros(self.batch, K, self.state_dim)
                Action = torch.zeros(self.batch, K, self.control_dim)
                counter = 0

    def train(self):
        ''' training based on current replay buffer
        '''
        # fill the replay buffer
        print('----------start fill the buffer--------')
        self.EnNet.to('cpu')
        #self.SampleTrajs()
        #self.SampleTrajs()
        #self.SampleTrajs()

        self._fill_buffer_from_record()
        
        self.ABC.to(device)
        self.EnNet.to(device)

        ABCParam = Trainer.setParams(self.ABC, 0) # don't impose weight decay on ABC matrix
        EnNetParam = Trainer.setParams(self.EnNet, self.decay)
        optimizer = Adam(ABCParam + EnNetParam, lr=self.lr)
        step = 0
        self.plot(step)
        while True:
            # split whole traj into self.k length short section
            print('----------start fitting the current buffer--------')
            for _ in range(self.trainNum):
                train_loss = 0
                for state, action in self.BufferBatchGenerator():
                    # to GPU/CPU tensor
                    state = state.to(device)
                    action = action.to(device)
                    batchNum = self.simNum // self.k
                    for i in range(batchNum):
                        state_i = state[:, i*self.k:(i+1)*self.k, :].to(device)
                        action_i = action[:, i*self.k:(i+1)*self.k, :].to(device)
                        optimizer.zero_grad()
                        Loss = self._calc_loss(state_i, action_i, self.k)
                        # backwards
                        Loss.backward()
                        optimizer.step()
                        train_loss += Loss.item()
                ctrlability = self.ABC.getCtrlRank().detach().numpy()
                print('------ Training loss: {0:.2f}, Controlability: {1} ------'.format(train_loss, ctrlability))
            

            self.ABC.cpu()
            self.EnNet.cpu()
            self._save_model(step)
            print('----------update ABC and sample new traj--------')
            # update MPC controller
            with torch.no_grad():
                ABCweight = list(self.ABC.named_parameters())
                if self.whichController == 'MPC':
                    self.MPC = MPC(ABCweight)
                elif self.whichController == 'LQR':
                    self.LQR = FiniteLQR(ABCweight, self.simNum)
                    self.KF = KF(ABCweight)

            for _ in range(self.updateNum):
                self.SampleTrajs()
            # save one video for gym
            if self.whichOde == 'gym':
                self.simuOnce(0, 0, True) # for gym: Pi=0, x0=0, not important
                self.odeSolver[0].close()

            self.plot(step)
            self.ABC.to(device)
            self.EnNet.to(device)
            step += 1

    def _save_model(self, step:int) -> None:
        ''' Helper function to save training model under ./Model folder
            input:
                    step:   step number for naming
        '''
        Name = 'Step{0}'.format(step)
        Path = './Models/OnlineTrainingModels/' + Name
        if not os.path.exists(Path):
            os.mkdir(Path)

            torch.save(self.ABC, Path + '/ABC.pth')
            #torch.save(self.DeNet, Path + '/DeNet.pth')
            torch.save(self.EnNet, Path + '/EnNet.pth')
        else:
            print('dir exits...')

    def _fill_buffer_from_record(self):
        ''' fill up the replay buffer from recording H5 files
        '''
        
        path = os.path.dirname(os.path.abspath(__file__))
        if self.state_dim == 3:
            hf = h5py.File(path + '/Data/GymTraj/GymTrajTrain.h5', 'r')
        elif self.state_dim == 4:
             hf = h5py.File(path + '/Data/PendulumTraj/PendulumTraj.h5', 'r')
        for key in hf.keys():
            data = hf.get(key)
            data = data[:self.simNum, :]
            self.buffer.append(data)
            if len(self.buffer) >= self.buffer.maxlen:
                break

