from train import BaseTrainer, Trainer
import tqdm
import time
import os
from typing import Callable, Generator, Tuple

import torch
import torch.optim as optim
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

from networks import Decoder, Encoder, ABC, LeastSquareABC
from loss import LossFunc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer3(Trainer):
    def __init__(self, args):
        super().__init__(args)
        # only act as parameters container
        self.ABC.A.requires_grad = False
        self.ABC.B.requires_grad = False
        self.ABC.C.requires_grad = False

    def train(self, datasetGenerator:Callable) -> None:
        '''Training function, including validation and logging training results
            input: 
                datasetGenerator: generator function of corresponding dataset
            output:
                None
        '''
        EnNetParam = Trainer.setParams(self.EnNet, self.decay)
        optimizer = Adam(EnNetParam, lr=self.lr)
        valiDataset = list(datasetGenerator(self.validate_k, 1, TrainVali='vali')) # 1 -> batchsize
        vali_loss_l = []
        ctrlability_l = []

        # main training loop
        for epoch in tqdm.tqdm(range(self.epoch)):
            train_loss = 0

            # learning weight decay
            for g in optimizer.param_groups:
                g['lr'] = self.lr * self.lr_decay ** epoch
            print("Decay learning rate: {0}".format(g['lr']))
            # update 10 times
            for _ in range(5):
                for state, action in datasetGenerator(self.validate_k, self.batch):
                    # to GPU/CPU tensor
                    state = state.to(device)
                    action = action.to(device)
                    Loss = self._calc_loss(state, action, self.k)
                    # backwards
                    Loss.backward()
                    optimizer.step()
                    train_loss += Loss.item()
            ######################### synchronized ABC each k step
            print('Update ABC ...')
            with torch.no_grad():
                Lstate = self.EnNet(state)
                BA, BB, BC = LeastSquareABC.calc_BatchABC(state, Lstate, action) # batch ABC
                Bar_A = BA.mean(0)
                Bar_B = BB.mean(0)
                Bar_C = BC.mean(0)
                self.ABC.A.data =  0.5*self.ABC.A.data + 0.5*Bar_A
                self.ABC.B.data =  0.5*self.ABC.B.data + 0.5*Bar_B
                self.ABC.C.data =  0.5*self.ABC.C.data + 0.5*Bar_C
            #########################
            # validate
            valiLoss, ctrlability = self._validate(valiDataset, epoch)
            vali_loss_l.append(valiLoss)
            ctrlability_l.append(ctrlability)
            self._save_model(valiLoss, epoch)
            print("Training Loss: %e Validating Loss: %e" % (train_loss, valiLoss))
        logging.info('The {0} horrizon, {1} lift dimension got {2:.3f} validataion error at {3}. episode. Maximun reachable ctrl rank is {4} at {5} epoch.'\
                                        .format(self.k, self.lift, np.min(vali_loss_l), np.argmin(vali_loss_l), np.max(ctrlability_l), np.argmax(ctrlability_l)))

        # plot training process
        self._plot_loss(vali_loss_l, ctrlability_l)
    
    def _calc_loss(self, state:torch.Tensor, action:torch.Tensor, k_input:int) -> torch.Tensor:
        ''' Loss calculation for training and validation
            Learning koopman loss as paper
            input:
                state:      one batch state from dataset generator
                action:     one batch action from dataset generator
                k_input:    No use
            output:
                Loss:       tensor with gradfn
        '''
        # state lifting
        Lstate = self.EnNet(state)
        # Dynamics K step simulation
        OneStepAheadPred = self.ABC.wholeTrajForward(Lstate, action) # OneStepAheadPred one step lags as Lstate
        OneStepAheadTarget = Lstate[:,1:,:]
        # K step simulation loss in lifting space
        L1 = self.LossFunc.KStepTransLoss(OneStepAheadPred, OneStepAheadTarget)
        # Decode K step
        OneStepAheadDecode = self.ABC.backmapping(OneStepAheadPred)
        OneStepAheadDecode_ws = self.ABC.backmapping(OneStepAheadTarget) # without simulation
        OneStepAheadDecodeTarget = state[:,1:,:]
        # Decode one step and K step Loss, gave it a factor 200 to let network guarantee decoding precision
        L2 = self.LossFunc.KStepDecodeLoss(OneStepAheadDecode, OneStepAheadDecodeTarget)
        L3 = self.LossFunc.KStepDecodeLoss(OneStepAheadDecode_ws, OneStepAheadDecodeTarget)

        # ctrlability loss no use
        Gramian = self.ABC.CtrlAbilityGramian().to(device)
        L4 = self.LossFunc.CtrlAbilityLoss(Gramian)
        L5 = self.LossFunc.InfinityLoss(OneStepAheadDecode, OneStepAheadDecodeTarget)
        L6 = self.LossFunc.MetricLoss(Lstate, state)
        # add all weighted loss
        #print('KStepTrans: {0:.2f}, KStepTransDecode: {1:.2f}, KStepPureDecode: {2:.2f}'.format(100*L1, L2, 2*L3))
        Loss = 20*L1 + 3*L2 + 3*L3 + L5 + L6

        return Loss

class Trainer2(BaseTrainer):
    ''' ABC non parametric least square version
        Trainer can't solve robot dynamical learning
        This is for panda learning
    '''
    def __init__(self, args):
        super().__init__(args)
        self.ABC = LeastSquareABC(args.lift, args.state_dim, args.control_dim).to(device)

    def train(self, datasetGenerator:Callable) -> None:
        
        EnNetParam = Trainer2.setParams(self.EnNet, self.decay)
        optimizer = Adam(EnNetParam, lr=self.lr)

        valiDataset = list(datasetGenerator(self.validate_k, 4, TrainVali='vali')) # Use the same dataset for validation
        vali_loss_l = []
        ctrlability_l = []
        train_loss = 0
        for epoch in tqdm.tqdm(range(self.epoch)):
            # learning weight decay
            for g in optimizer.param_groups:
                g['lr'] = self.lr * self.lr_decay ** epoch
            print("Decay learning rate: {0}".format(g['lr']))
            #batch main loop
            for state, action in datasetGenerator(self.validate_k, self.batch):
                # to GPU/CPU tensor
                state = state.to(device)
                action = action.to(device)
                Lstate = self.EnNet(state)
                A, B, C = LeastSquareABC.calc_BatchABC(state, Lstate, action)
                # split whole traj into self.k length short section, then get the loss
                section_length = self.validate_k // self.k
                Loss = 0
                for i in range(section_length):
                    state_i = state[:, i*self.k:(i+1)*self.k, :]
                    Lstate_i = Lstate[:, i*self.k:(i+1)*self.k, :]
                    action_i = action[:, i*self.k:(i+1)*self.k, :]
                    Loss_i = self._calc_loss(A, B, C, state_i, Lstate_i, action_i, self.k)
                    Loss += Loss_i
                    
                optimizer.zero_grad()
                Loss.backward() #retain_graph=True
                optimizer.step()
                # backwards
                self.ABC.lowpassfilterABC(A, B, C) # update parameter ABC
                
                train_loss += Loss
                print(self.ABC._ctrlRank)
            # validate
            valiLoss, ctrlability = self._validate(valiDataset, epoch)
            vali_loss_l.append(valiLoss)
            ctrlability_l.append(ctrlability)
            self._save_model(valiLoss, epoch)
            print("Training Loss: %e Validating Loss: %e" % (train_loss, valiLoss))
            train_loss = 0
            
        logging.info('The {0} horrizon, {1} lift dimension got {2:.3f} validataion error at {3}. episode. Maximun reachable ctrl rank is {4} at {5} epoch.'\
                                        .format(self.k, self.lift, np.min(vali_loss_l), np.argmin(vali_loss_l), np.max(ctrlability_l), np.argmax(ctrlability_l)))


    def _calc_loss(self, A, B, C, state, Lstate, action, k_input):
        ''' all the state Lstate action are short section of long traj
        '''
        # Dynamics K step simulation
        KStepPred = self.ABC(A, B, Lstate, action, k_input)
        KStepTarget = Lstate[:,0:k_input,:]
        # K step simulation loss in lifting space
        L1 = self.LossFunc.KStepTransLoss(KStepPred, KStepTarget)
        # Decode K step
        KStepDecode = self.ABC.backmapping(C, KStepPred)
        KStepDecode_ws = self.ABC.backmapping(C, KStepTarget) # without simulation
        KStepDecodeTarget = state[:,0:k_input,:]
        # Decode one step and K step Loss, gave it a factor 200 to let network guarantee decoding precision
        L2 = self.LossFunc.KStepDecodeLoss(KStepDecode, KStepDecodeTarget)
        L3 = self.LossFunc.KStepDecodeLoss(KStepDecode_ws, KStepDecodeTarget)

        # ctrlability loss no use
        Gramian = self.ABC.CtrlAbilityGramian(A, B).to(device)
        L4 = self.LossFunc.CtrlAbilityLoss(Gramian)
        # add all weighted loss
        #print('KStepTrans: {0:.2f}, KStepTransDecode: {1:.2f}, KStepPureDecode: {2:.2f}'.format(L1, L2, L3))
        Loss = L1 + L2 + 2*L3

        return Loss

    def _validate(self, valiDataset:list, epoch:int) -> Tuple[float, float]:
        ''' validation function
            input:
                valiDataset:        list for validation set
                epoch:              passing from training, for information printing
            output:
                total_loss:         total validation loss
                ctrlability:        controllability matrix rank number
        '''

        #self.DeNet.eval()
        self.EnNet.eval()
        self.ABC.eval()
        print("Start validation...")
        ctrlability = self.ABC.getCtrlRank().detach().numpy()
        print("Controlability matrix rank: ", ctrlability)
        with torch.no_grad():
            total_Loss = 0
            count = 0
            for state, action in valiDataset:
                state = state.to(device)
                action = action.to(device)
                Lstate = self.EnNet(state)
                A,B,C = LeastSquareABC.calc_BatchABC(state, Lstate, action)
                valiLoss = self._calc_loss(A, B, C, state, Lstate, action, self.validate_k)
                total_Loss += valiLoss.item()
                count += 1

            if self.state_dim == 3:
                self._plot_oneStepCartPole(state, action, self.validate_k, epoch)

            if self.state_dim == 14:
                self._plot_oneStepPanda(state, action, self.validate_k, epoch)

        #self.DeNet.train()
        self.EnNet.train()
        self.ABC.train()
        print("### Single step loss: {:.2f} ###".format(total_Loss / count))
        return total_Loss, ctrlability

    def _plot_oneStepCartPole(self, state:torch.Tensor, action:torch.Tensor, validate_k:int, epoch:int):
        ''' plot one step for validation
            input: 
                state:      one batch state from generator
                action:     one batch action from generator
                validate_k: how many steps need to be simulated
                epoch:      epoch number for img file name
        '''
        state_np = state.cpu().numpy()
        state_l = self.EnNet(state)
        A,B,C = LeastSquareABC.calc_BatchABC(state, state_l, action)
        state_next = self.ABC(A,B,state_l, action, validate_k)
        state_nextD = self.ABC.backmapping(C,state_next)
        res = state_nextD.cpu().detach().numpy()
        #### plot
        whichone = 1 # only plot the first example
        plt.figure(figsize=[12,8])
        for channel in range(self.state_dim):
            if channel == 0:
                plt.subplot(4,2,channel+1,xlabel="", ylabel="Position of cart".format(channel+1))
            elif channel == 1:
                plt.subplot(4,2,channel+1,xlabel="", ylabel="Velocity of cart".format(channel+1))
            elif channel == 2:
                plt.subplot(4,2,channel+1,xlabel="Step", ylabel="Angular of bar".format(channel+1))
            elif channel == 3:
                plt.subplot(4,2,channel+1,xlabel="Step", ylabel="Angular velocity of bar".format(channel+1))
            plt.plot(state_np[whichone, :, channel],'-c', label="target")
            plt.plot(res[whichone, :,channel],'-r', label="pred")
        plt.legend()
        plt.savefig('./Plots/OfflineTrainingPlots/valiPlot_ep_{0}.jpg'.format(epoch), dpi=200)
        plt.close()

    def _save_model(self, valiLoss:float, epoch:int) -> None:
        ''' Helper function to save training model under ./Model folder
            input:
                    valiLoss:   validation loss for naming
                    epoch:      epoch number for naming
        '''
        Name = 'Ep{0}_valiLoss_{1:.2f}'.format(epoch, valiLoss)
        Path = './Models/OfflineTrainingModels/' + Name
        if not os.path.exists(Path):
            os.mkdir(Path)
            torch.save(self.ABC, Path + '/ABC.pth')
            #torch.save(self.DeNet, Path + '/DeNet.pth')
            torch.save(self.EnNet, Path + '/EnNet.pth')
        else:
            print('dir exits...')