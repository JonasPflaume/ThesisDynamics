import logging
import tqdm
import time
import os
import random
from typing import Callable, Generator, Tuple

import torch
import torch.optim as optim
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

from networks import Decoder, Encoder, ABC, LeastSquareABC
from loss import LossFunc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BaseTrainer(object):
    def __init__(self, args:dict):
        # unpack hyperparameters
        self.batch = args.batch
        self.epoch = args.epoch
        self.dor = args.dor
        self.decay = args.decay
        self.lr = args.lr
        self.k = args.k
        self.lift = args.lift
        self.validate_k = args.validate_k

        self.state_dim = args.state_dim
        self.control_dim = args.control_dim
        self.lr_decay = args.lr_decay
        # initialize networks
        #self.DeNet = Decoder(self.lift, self.state_dim, self.dor).to(device)
        if args.pretrained:
            print('Use pretrained model...')
            modelpath = './Models/TrainedOfflineModel/' + args.pretrained
            self.EnNet = torch.load(modelpath + '/EnNet.pth').to(device).train()
            self.ABC = torch.load(modelpath + '/ABC.pth').to(device).train()
            #self.DeNet = torch.load(modelpath + '/DeNet.pth').to(device).train()
        else:
            self.EnNet = Encoder(self.state_dim, self.lift, self.dor).to(device)
            self.ABC = ABC(self.lift, self.control_dim, self.state_dim).to(device)
            #self.DeNet = Decoder(self.lift, self.state_dim, self.dor).to(device)
        self.LossFunc = LossFunc()

        logging.basicConfig(filename='./Logs/training_log', filemode='a', \
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',\
                            datefmt='%H:%M:%S',\
                            level=logging.INFO)

    def _plot_loss(self, valiLoss_l:list, ctrl_l:list) -> None:
        ''' helper function to draw the training results
            input:
                    valiLoss_l: validation loss list
                    ctrl_l:     controlability rank list
        '''
        plt.figure(figsize=[7,7])
        plt.plot(valiLoss_l, '-r')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid()
        plt.savefig('./Logs/validationLoss.jpg', dpi=200)
        plt.close()

        plt.figure(figsize=[7,7])
        plt.plot(ctrl_l, '-r')
        plt.xlabel("epoch")
        plt.grid()
        plt.savefig('./Logs/ctrlRank.jpg', dpi=200)
        plt.close()

    @staticmethod
    def setParams(network:torch.nn.Module, decay:float) -> list:
        ''' function to set weight decay
        '''
        params_dict = dict(network.named_parameters())
        params=[]
        weights=[]

        for key, value in params_dict.items():
            if key[-4:] == 'bias':
                params += [{'params':value,'weight_decay':0.0}]
            else:             
                params +=  [{'params': value,'weight_decay':decay}]
        return params

    def train(self):
        raise NotImplementedError

    def _validate(self):
        raise NotImplementedError

    def _calc_loss(self):
        raise NotImplementedError

    def _save_model(self):
        raise NotImplementedError


class Trainer(BaseTrainer):
    '''
    Trainer class for Deep koopman embedding, ABC parameters version
    Create instance by passing args dict, then call train() method
    '''
    def __init__(self, args:dict):
        super().__init__(args)

    def train(self, datasetGenerator:Callable) -> None:
        '''Training function, including validation and logging training results
            input: 
                datasetGenerator: generator function of corresponding dataset
            output:
                None
        '''
        ABCParam = Trainer.setParams(self.ABC, 0) # don't impose weight decay on ABC matrix
        EnNetParam = Trainer.setParams(self.EnNet, self.decay)
        #DeNetParam = Trainer.setParams(self.DeNet, self.decay)
        optimizer = Adam(ABCParam + EnNetParam, lr=self.lr)
        valiDataset = list(datasetGenerator(self.validate_k, 1, TrainVali='vali')) # 1 -> batchsize
        vali_loss_l = []
        ctrlability_l = []

        # AB pre-training (not a good idea)
        '''
        while True:
            optimizer.zero_grad()
            Gramian = self.ABC.CtrlAbilityGramian()
            ctrlLoss = self.LossFunc.CtrlAbilityLoss(Gramian) * 1e6
            ctrlLoss.backward()
            optimizer.step()
            print('############', self.ABC.getCtrlRank())
            
            if self.ABC.getCtrlRank() == self.lift * 0.8:
                print(self.ABC.A)
                break
        '''
        # main training loop
        for epoch in tqdm.tqdm(range(self.epoch)):
            train_loss = 0

            # learning weight decay
            for g in optimizer.param_groups:
                g['lr'] = self.lr * self.lr_decay ** epoch
            print("Decay learning rate: {0}".format(g['lr']))

            # split whole traj into self.k length short section
            for state, action in datasetGenerator(self.validate_k, self.batch):
                # to GPU/CPU tensor
                state = state.to(device)
                action = action.to(device)
                batchNum = self.validate_k // self.k
                Loss = 0
                state_b = [state[:, i*self.k:(i+1)*self.k, :] for i in range(batchNum)]
                action_b = [action[:, i*self.k:(i+1)*self.k, :] for i in range(batchNum)]
                random.shuffle(state_b)
                random.shuffle(action_b)
                for i in range(batchNum):
                    state_i = state_b[i]
                    action_i = action_b[i]

                    optimizer.zero_grad()
                    Loss_i = self._calc_loss(state_i, action_i, self.k)
                    Loss += Loss_i.item()
                    # backwards
                    Loss_i.backward()
                    optimizer.step()
                train_loss += Loss
            
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
            input:
                state:      one batch state from dataset generator
                action:     one batch action from dataset generator
                k_input:    k step number for AB net dynamical simulation
            output:
                Loss:       tensor with gradfn
        '''
        # state lifting
        Lstate = self.EnNet(state)
        # metrc loss
        L6 = self.LossFunc.MetricLoss(Lstate, state) 
        # Dynamics K step simulation
        KStepPred = self.ABC(Lstate, action, k_input)
        KStepTarget = Lstate[:,0:k_input,:]
        # K step simulation loss in lifting space
        L1 = self.LossFunc.KStepTransLoss(KStepPred, KStepTarget)
        # Decode K step
        KStepDecode = self.ABC.backmapping(KStepPred)
        KStepDecode_ws = self.ABC.backmapping(KStepTarget) # without simulation
        #KStepDecode = self.DeNet(KStepPred)
        #KStepDecode_ws = self.DeNet(KStepTarget) # without simulation
        KStepDecodeTarget = state[:,0:k_input,:]
        # Decode one step and K step Loss, gave it a factor 200 to let network guarantee decoding precision
        L2 = self.LossFunc.KStepDecodeLoss(KStepDecode, KStepDecodeTarget)
        L3 = self.LossFunc.KStepDecodeLoss(KStepDecode_ws, KStepDecodeTarget)

        # ctrlability loss no use
        Gramian = self.ABC.CtrlAbilityGramian().to(device)
        L4 = self.LossFunc.CtrlAbilityLoss(Gramian)
        L5 = self.LossFunc.InfinityLoss(KStepDecode, KStepDecodeTarget)
        
        # add all weighted loss
        # print('KStepTrans: {0:.2f}, KStepTransDecode: {1:.2f}, KStepPureDecode: {2:.2f}, InfinityLoss: {3:.2f}, MetricLoss: {4:.2f}'.format(L1, L2, L3, L5, L6))
        Loss = L1 + 3*L2 + 20*L3 + L6
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
                valiLoss = self._calc_loss(state, action, self.validate_k) 
                total_Loss += valiLoss.item()
                count += 1

            if self.state_dim == 3:
                self._plot_pendulum(state, action, self.validate_k, epoch)

            if self.state_dim == 14:
                self._plot_oneStepPanda(state, action, self.validate_k, epoch)

        #self.DeNet.train()
        self.EnNet.train()
        self.ABC.train()
        print("### Single step loss: {:.2f} ###".format(total_Loss / count))
        return total_Loss, ctrlability

    def _plot_pendulum(self, state:torch.Tensor, action:torch.Tensor, validate_k:int, epoch:int):
        ''' plot one step for validation
            input: 
                state:      one batch state from generator
                action:     one batch action from generator
                validate_k: how many steps need to be simulated
                epoch:      epoch number for img file name
        '''
        state_np = state.cpu().numpy()
        state_l = self.EnNet(state)

        state_next = self.ABC(state_l, action, validate_k)
        state_nextD = self.ABC.backmapping(state_next)
        res = state_nextD.cpu().detach().numpy()
        #### plot
        whichone = 0 # only plot the first example
        plt.figure(figsize=[12,8])
        for channel in range(self.state_dim):
            if channel == 0:
                plt.subplot(4,2,channel+1,xlabel="", ylabel="cos(theta)".format(channel+1))
            elif channel == 1:
                plt.subplot(4,2,channel+1,xlabel="", ylabel="sin(theta)".format(channel+1))
            elif channel == 2:
                plt.subplot(4,2,channel+1,xlabel="Step", ylabel="Dot(theta)".format(channel+1))
            elif channel == 3:
                plt.subplot(4,2,channel+1,xlabel="Step", ylabel="Angular velocity of bar".format(channel+1))
            plt.plot(state_np[whichone, :, channel],'-c', label="target")
            plt.plot(res[whichone, :,channel],'-r', label="pred")
            plt.grid()
        plt.legend()
        plt.savefig('./Plots/OfflineTrainingPlots/valiPlot_ep_{0}.jpg'.format(epoch), dpi=200)
        plt.close()

    def _plot_oneStepPanda(self, state:torch.Tensor, action:torch.Tensor, validate_k:int, epoch:int):
        ''' plot one step for validation
            input: 
                state:      one batch state from generator
                action:     one batch action from generator
                validate_k: how many steps need to be simulated
                epoch:      epoch number for img file name
        '''
        state_np = state.cpu().numpy()
        state_l = self.EnNet(state)

        state_next = self.ABC(state_l, action, validate_k)
        state_nextD = self.ABC.backmapping(state_next)
        res = state_nextD.cpu().detach().numpy()
        #### plot
        whichone = 0 # only plot the first example
        plt.figure(figsize=[12,8])
        for channel in range(self.state_dim):
            plt.subplot(7,2,channel+1,xlabel="", ylabel="Position of cart".format(channel+1))
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
