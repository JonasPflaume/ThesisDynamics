import torch
import torch.nn as nn
from typing import Tuple
import numpy as np

Activation = nn.LeakyReLU # activation function
enhance = 2               # increasing level of neural networks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ABC(nn.Module):
    ''' AB networks consists of only two torch parameter
    '''
    def __init__(self, LstateDim: int, ControlNum: int, stateDim:int):
        '''
            Arguments:
                LstateDim:  The dimension of lifting space
                ControlNum: Number of control
        '''
        super().__init__()
        self.LstateDim, self.ControlNum, self.stateDim = LstateDim, ControlNum, stateDim

        A, B, C = self.initABC(LstateDim, ControlNum)
        self.A = nn.parameter.Parameter(A, requires_grad=True)
        self.B = nn.parameter.Parameter(B, requires_grad=True)
        self.C = nn.parameter.Parameter(C, requires_grad=True)
        self._ctrlRank = torch.zeros(1,1)
        print(self)

    def initABC(self, LstateDim: int, ControlNum: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' Method for initialization of AB tensor
        '''
        a = torch.randn(LstateDim) * .01
        Disturb = torch.diag(a)
        V = torch.eye(LstateDim) + Disturb
        P = torch.rand(LstateDim, LstateDim)
        Q, R = torch.qr(P)
        A = Q @ V @ torch.inverse(Q)
        #A = torch.eye(LstateDim) + torch.randn(LstateDim, LstateDim) * 1e-3
        B = torch.randn(LstateDim, ControlNum) * .01
        C = torch.randn(self.stateDim, LstateDim) * .01
        return A, B, C

    def forward(self, x: torch.Tensor, u: torch.Tensor, stepNum: int) -> torch.Tensor:
        ''' forwad was conducted by picking the first state from batch, return a traj the same length as x and the same start
            Then carry out the linear system simulation for stepNum
            input:
                    x:          Batch of lifting space
                    u:          Batch of used torques from dataset
                    stepNum:    Linear system simulation steps
            output:
                    resX:       simulated trajectories have the same shape as input x
        '''
        batchSize = x.shape[0]
        A = self.A.repeat(batchSize, 1, 1)
        B = self.B.repeat(batchSize, 1, 1)
        startx = x[:, 0:1, :]
        startx = startx.permute(0,2,1) # change axis to do bmm
        resX = torch.zeros_like(x)
        resX = resX.permute(0,2,1)
        for i in range(stepNum):
            resX[:,:,i] = torch.squeeze(startx)
            ut = u[:,i,:]
            ut = torch.unsqueeze(ut, 1)
            ut = ut.permute(0,2,1)
            startx = torch.bmm(A, startx) + torch.bmm(B, ut)
        return resX.permute(0,2,1)

    def backmapping(self, xlift:torch.Tensor) -> torch.Tensor:
        ''' mapping lifting state back to statespace by C matrix
            input:
                    xlift: lifted state space
        '''
        batchSize = xlift.shape[0]
        C = self.C.repeat(batchSize, 1, 1)
        xlift = xlift.permute(0,2,1)
        x = torch.bmm(C, xlift)
        x = x.permute(0,2,1)
        return x

    def __repr__(self):
        return "ABC parameters with A: {0}x{0} B: {0}x{1} C: {0}x{2}".format(self.LstateDim, self.ControlNum, self.stateDim)

    def CtrlAbilityGramian(self) -> torch.Tensor:
        ''' method calculating the controlability gramian matrix
            output:
                    P:  Approximate controlability gramian matrix
                        (it tends to be positive definite and symmetric)
        '''
        CM = self.CtrAbilityMatrix()
        CM_T = torch.transpose(CM, 0, 1)
        P = torch.matmul(CM, CM_T)
        return P

    def CtrAbilityMatrix(self) -> torch.Tensor:
        ''' Classical controlability matrix rank
            output:
                    rank:   controlability matrix rank
        '''
        n = self.B.shape[0]
        r = self.B.shape[1]
        CM = torch.zeros(n, n*r)
        CM[:, 0:r] = self.B
        for i in range(1, n):
            CM[:, i*r:(i+1)*r] = torch.matrix_power(self.A, i) @ self.B
        self._ctrlRank = torch.matrix_rank(CM)
        return CM

    def getCtrlRank(self) -> int:
        ''' return the current calculated controlability rank
        '''
        return self._ctrlRank


    def freeze(self):
        ''' freeze the networks
        '''
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        ''' unfreeze the networks
        '''
        for p in self.parameters():
            p.requires_grad = True

    def wholeTrajForward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ''' forwad was conducted by picking the first state from batch, return the traj one step shorter than x
            Then carry out the linear system simulation
            input:
                    x:          Batch of lifting space
                    u:          Batch of used torques from dataset
                    stepNum:    Linear system simulation steps
            output:
                    resX:       simulated trajectories have the same shape as input x
        '''
        batchSize = x.shape[0]
        A = self.A.repeat(batchSize, 1, 1)
        B = self.B.repeat(batchSize, 1, 1)
        resX = torch.zeros(batchSize, x.shape[1]-1, x.shape[2]).to(device)
        resX = resX.permute(0,2,1)
        for i in range(resX.shape[2]):
            startx = x[:, i:i+1, :]
            startx = startx.permute(0,2,1) # change axis to do bmm
            ut = u[:,i,:]
            ut = torch.unsqueeze(ut, 1)
            ut = ut.permute(0,2,1)
            startx = torch.bmm(A, startx) + torch.bmm(B, ut)
            resX[:,:,i] = torch.squeeze(startx)
        return resX.permute(0,2,1)


class Encoder(nn.Module):
    '''Encoder network with dropout, LeakyReLU activation and xavier initialization'''
    def __init__(self, input_s, output, dor):
        super().__init__()
        self.L1 = nn.Linear(input_s, 6)
        self.L2 = nn.Linear(6, 12)
        self.L3 = nn.Linear(12, output)
        self.Act = Activation()
        #ingredients.append(Activation())
        

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        print('Encoder structure:', self)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.L1(x)
        x = self.Act(x)
        x = self.L2(x)
        x = self.Act(x)
        x = self.L3(x)
        return x

    def freeze(self):
        ''' freeze the networks
        '''
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        ''' unfreeze the networks
        '''
        for p in self.parameters():
            p.requires_grad = True

    def backmapping(self, x) -> torch.Tensor:
        batchsize = x.shape[0]
        L3_T = self.L3.weight.t().repeat(batchsize, 1, 1)
        L2_T = self.L2.weight.t().repeat(batchsize, 1, 1)
        L1_T = self.L1.weight.t().repeat(batchsize, 1, 1)
        x = torch.transpose(x, 1, 2)
        x = torch.bmm(L3_T, x)
        x = self.Act(x)
        x = torch.bmm(L2_T, x)
        x = self.Act(x)
        x = torch.bmm(L1_T, x)
        x = torch.transpose(x, 1, 2)
        return x

class Decoder(nn.Module):
    '''Decoder network with dropout, LeakyReLU activation and xavier initialization'''
    def __init__(self, input_s, output, dor):
        super().__init__()
        ingredients = []
        ingredients.append(nn.Linear(input_s, 6))
        ingredients.append(Activation())
        ingredients.append(nn.Linear(6, 12))
        ingredients.append(Activation())
        ingredients.append(nn.Linear(12, output))
        #ingredients.append(Activation())
        #ingredients.append(nn.Linear(40, output))
        self.net = nn.Sequential( *ingredients )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print('Decoder structure:', self)

    def forward(self, x):
        return self.net(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

class LeastSquareABC(ABC):
    ''' TODO
    '''
    def __init__(self, LiftDim, OriginalDim, actionDim):
        super().__init__(LiftDim, actionDim, OriginalDim)
        
        self.A.requires_grad = False
        self.B.requires_grad = False
        self.C.requires_grad = False

    @staticmethod
    def calc_BatchABC(state, Lstate, action):
        ''' get least square ABC
            input:
                    state:      states  [batch, K, xdof]
                    Lstate:     lifting states [batch, K, Lxdof]
                    action:     batch torques [batch, K, Udof]
        '''
        K = Lstate.shape[1]
        XLdof = Lstate.shape[2]
        Udof = action.shape[2]

        X = Lstate[:, :K-1, :].permute(0,2,1)
        Y = Lstate[:, 1:, :].permute(0,2,1)
        assert X.shape[1] == Y.shape[1]
        U = action[:,:K-1,:].permute(0,2,1)
        XU = torch.cat((X, U), 1)
        try:
            XUpinv = torch.pinverse(XU)
        except:
            print('Unstable pseudo inverse, add damping...')
            diag = torch.eye(XU.shape[2])
            diag = diag.reshape((1, XU.shape[2], XU.shape[2]))
            diag = diag.repeat(XU.shape[0], 1, 1)
            XUpinv = torch.bmm(torch.transpose(XU, 1, 2), XU) + diag.to(device) * 1e-3
            XUpinv = torch.bmm(torch.inverse(XUpinv), torch.transpose(XU, 1, 2))
        AB = torch.bmm(Y, XUpinv)
        _A = AB[:, :, :XLdof]
        _B = AB[:, :, XLdof:]
        assert _B.shape[2] == Udof

        try:
            Xpinv = torch.pinverse(X)
        except:
            print('Unstable pseudo inverse, add damping...')
            diag = torch.eye(X.shape[2])
            diag = diag.reshape((1, X.shape[2], X.shape[2]))
            diag = diag.repeat(X.shape[0], 1, 1)
            Xpinv = torch.bmm(torch.transpose(X, 1, 2), X) + diag.to(device) * 1e-3
            Xpinv = torch.bmm(torch.inverse(Xpinv), torch.transpose(X, 1, 2))
        X_ori = state[:, :K-1, :].permute(0,2,1)
        _C = torch.bmm(X_ori, Xpinv)

        return _A, _B, _C

    def forward(self, A: torch.Tensor, B: torch.Tensor, x: torch.Tensor, u: torch.Tensor, stepNum: int) -> torch.Tensor:
        ''' forwad was conducted by picking the first state from batch
            Then carry out the linear system simulation for stepNum
            input:
                    AB:        dynamics
                    x:          Batch of lifting space
                    u:          Batch of used torques from dataset
                    stepNum:    Linear system simulation steps
            output:
                    resX:       simulated trajectories have the same shape as input x
        '''
        batchSize = x.shape[0]
        startx = x[:, 0:1, :]
        startx = startx.permute(0,2,1) # change axis to do bmm
        resX = torch.zeros_like(x)
        resX = resX.permute(0,2,1)
        for i in range(stepNum):
            resX[:,:,i] = torch.squeeze(startx)
            ut = u[:,i,:]
            ut = torch.unsqueeze(ut, 1)
            ut = ut.permute(0,2,1)
            startx = torch.bmm(A, startx) + torch.bmm(B, ut)
        return resX.permute(0,2,1)

    def backmapping(self, C:torch.Tensor, xlift:torch.Tensor) -> torch.Tensor:
        ''' mapping lifting state back to statespace by C matrix
            input:
                    xlift: lifted state space
        '''
        batchSize = xlift.shape[0]
        xlift = xlift.permute(0,2,1)
        x = torch.bmm(C, xlift)
        x = x.permute(0,2,1)
        return x

    def __repr__(self):
        return "Implicit defined ABC parameters with A: {0}x{0} B: {0}x{1} C: {0}x{2}".format(self.LstateDim, self.ControlNum, self.stateDim)

    def lowpassfilterABC(self, A, B, C):
        with torch.no_grad():
            self.A.data  = A.mean(0)
            self.B.data  = B.mean(0)
            self.C.data  = C.mean(0)


    def CtrlAbilityGramian(self, A, B) -> torch.Tensor:
        ''' method calculating the controlability gramian matrix
            output:
                    P:  Approximate controlability gramian matrix
                        (it tends to be positive definite and symmetric)
        '''
        CM = self.CtrAbilityMatrix(A, B)
        CM_T = torch.transpose(CM, 1, 2)
        P = torch.bmm(CM, CM_T)
        return P

    def CtrAbilityMatrix(self, A, B) -> torch.Tensor:
        ''' Classical controlability matrix rank
            output:
                    rank:   controlability matrix rank
        '''
        n = self.B.shape[0]
        r = self.B.shape[1]
        b = A.shape[0]
        CM = torch.zeros(b, n, n*r)
        CM[:, :, 0:r] = B
        
        for i in range(1, n):
            CM[:, :, i*r:(i+1)*r] = torch.bmm(torch.matrix_power(A, i) , B)
        
        self._ctrlRank = torch.matrix_rank(CM[0])
        return CM

    def getCtrlRank(self) -> int:
        ''' return the current calculated controlability rank
        '''
        return self._ctrlRank
