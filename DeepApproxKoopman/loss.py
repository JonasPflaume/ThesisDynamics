from torch.nn import MSELoss
import torch

class LossFunc:
    '''Leave here as a interface to change the definition of loss criterion'''
    __slots__ = ()

    @staticmethod
    def KStepTransLoss(Psi_pred, Psi_true):
        ''' Dynamic transition loss
            input:
                Psi_pred:   AB output in lifting space
                Psi_true:   Direct lifting results from Encoder
            output:
                Loss:       Loss tensor   
        '''
        criterion = MSELoss()
        Loss = criterion(Psi_pred, Psi_true)
        return Loss

    @staticmethod
    def KStepDecodeLoss(Xpred, Xtrue):
        ''' K step decode loss function
            input:
                Xpred:  decode results from AB transition results or directly from encoder
                Xtrue:  true trajectories in dataset
            output:
                Loss:   loss tensor
        '''
        Loss = (Xpred - Xtrue)**2
        Loss = 1 / Loss.numel() * torch.sum(Loss)
        return Loss

    @staticmethod
    def CtrlAbilityLoss(Gramian):
        ''' Controlability loss, positive definite loss
            input:
                    Gramian: n x n square matrix
            output:
                    Loss: calculated loss
        '''
        E, V = torch.symeig(Gramian, eigenvectors=True)
        Loss = torch.exp(-0.1 * (E - 5)) + torch.exp(0.1 * (E - 5))
        Loss = Loss.sum()
        return Loss

    @staticmethod
    def LsAbcLoss(A, B, C, state, Lstate, action, k):
        ''' least square ABC loss
            input:
                ABC:        batch system matrix
                state:      original state
                Lstate:     Lifting sate
                action:     torques
                k:          k step simulation
        '''
        SecNum = state.shape[1] // k
        for i in range(SecNum):
            psi_start = Lstate[:,i*k:i*k+1,:]
            # k step simulation
            for j in range(k):
                psi_pred = torch.bmm(A, mat2)


    @staticmethod
    def BatchCtrlAbilityLoss(Gramian):
        ''' Controlability loss, positive definite loss
            input:
                    Gramian: b x n x n square matrix
            output:
                    Loss: calculated loss
        '''
        E, V = torch.symeig(Gramian, eigenvectors=True)
        Loss = 0
        for Ei in E:
            Loss_i = 0.1 * Ei ** 2 - 2 * Ei + 10
            Loss_i = Loss_i.sum()
            Loss += Loss_i
        return Loss

    @staticmethod
    def InfinityLoss(Xpred, Xtrue):
        ''' K step decode loss function by infinity norm metric
            input:
                Xpred:  decode results from AB transition results or directly from encoder
                Xtrue:  true trajectories in dataset
            output:
                Loss:   loss tensor
        '''
        Loss = Xpred - Xtrue
        Loss = torch.norm(Loss, p=float('inf'))
        return Loss

    @staticmethod
    def MetricLoss(Lstate, state):
        K = Lstate.shape[1]
        idx_1 = 0
        idx_2 = 0
        while idx_1 == idx_2:
            idx_1 = torch.randint(0, K, (1,))
            idx_2 = torch.randint(0, K, (1,))

        x_i, x_j = state[:,idx_1,:], state[:,idx_2, :]
        X_i, X_j = Lstate[:,idx_1,:], Lstate[:,idx_2, :]

        dis_x = torch.norm(x_i - x_j, dim=(1,2))
        dis_X = torch.norm(X_i - X_j, dim=(1,2))

        Loss = dis_x - dis_X
        Loss = torch.norm(Loss)
        return Loss



