from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import sigmoid_kernel

from functools import partial

import sys
sys.path.append('/home/jiayun/Desktop/oa-workspace/Exercises/')
sys.path.append('/home/jiayun/Desktop/oa-workspace/')
from a1_unconstrained_solver.solution import SolverUnconstrained
from sigmoid_activation import Sigmoid, activation, Cosine

class Regression(object):
    def __init__(self):
        raise NotImplementedError()
        
    def fit(self):
        raise NotImplementedError()
        
    def predict(self):
        raise NotImplementedError()
        

class CV_RR(Regression):
    def __init__(self, labd):
        self.labd = labd # regularization parameters
        
    def fit(self, Xtrain, Ytrain, K, verbose=True):
        # K fold validation parameter
        fold_len = len(Xtrain) // K
        fold_idx = [i*fold_len for i in range(K)]
        MSE = []
        VAR = []
        for labd in tqdm(self.labd):
            Loss = []
            for k in range(K):
                Xtrain_k = np.concatenate((Xtrain[:k*fold_len, :] ,Xtrain[(k+1)*fold_len:, :]), axis=0)
                Ytrain_k = np.concatenate((Ytrain[:k*fold_len, :] ,Ytrain[(k+1)*fold_len:, :]), axis=0)
                Xtrain_vali = Xtrain[k*fold_len:(k+1)*fold_len, :]
                Ytrain_vali = Ytrain[k*fold_len:(k+1)*fold_len, :]
                
                beta_k = self.get_beta(Xtrain_k, Ytrain_k, labd)
                # loss calculation
                pred_k = self.predict(Xtrain_vali, beta_k)
                Loss_k = (Ytrain_vali - pred_k) ** 2
                Loss_k = Loss_k.sum() / len(Xtrain_vali)
                Loss.append(Loss_k)
            Loss = np.array(Loss)
            l_hat = np.sum(Loss) / K
            MSE.append(l_hat)
            VAR.append(1/(K-1)*(np.sum(Loss**2)- K*l_hat**2 ))
        labd_idx = np.argmin(MSE)
        self.min_labd = self.labd[labd_idx]

        self.beta = self.get_beta(Xtrain, Ytrain, self.min_labd)
        ### show the MSE and variance of CV ###
        if verbose:
            print("Min MSE: ", MSE[labd_idx])
            MSE = np.array(MSE)
            VAR = np.array(VAR)
            plt.plot(self.labd, MSE, 'k-')
            plt.scatter(self.min_labd, MSE[labd_idx], c='r')
            plt.xscale('log')
            #plt.fill_between(self.labd, MSE + 1e-4*VAR, MSE - 1e-4*VAR) # too large to illustrate
            plt.show()

    def get_beta(self, Xtrain_k, Ytrain_k, labd):
        I = np.eye(Xtrain_k.shape[1])
        I[0,0] = 0
        return np.linalg.solve(Xtrain_k.T @ Xtrain_k + I*labd, Xtrain_k.T @ Ytrain_k)
        
    def predict(self, X, beta):
        return X @ beta
    

class CV_KRR(CV_RR):
    def __init__(self, labd, kernel:str, gamma=None, coef=None):
        super(CV_KRR, self).__init__(labd)
        self.kernel = kernel
        self.gamma = gamma # for RBF kernel
        self.coef = coef
    
    def get_beta(self, Xtrain_k, Ytrain_k, labd):
        K_gram = self.get_gram(Xtrain_k)
        I = np.eye(K_gram.shape[1])
        self.Xtrain = Xtrain_k
        return np.linalg.solve(K_gram + I*labd, Ytrain_k)
    
    def predict(self, Xvali, beta):
        k_gram = self.get_gram(self.Xtrain, Xvali)
        return k_gram.T @ beta
    
    def get_gram(self, X, X_=None):
        if self.kernel == 'rbf':
            func = partial(rbf_kernel, gamma=self.gamma) if self.gamma != None else rbf_kernel
        elif self.kernel == 'poly':
            func = polynomial_kernel
        elif self.kernel == 'sig':
            func = partial(sigmoid_kernel, gamma=self.coef) if self.coef != None else sigmoid_kernel
        else:
            print('Not supported kernel type!')
        if type(X_) != np.ndarray:
            return func(X)
        else:
            return func(X, X_)
        
class BoostRegressor(Regression):
    def __init__(self, depth=1, activation=activation, varphi=[7, 5, 0.01]):
        self.depth = depth
        self.solver = SolverUnconstrained()
        self.solver.verbose = False
        self.activation = activation
        self.varphi = varphi
        
    def fit(self, Xtrain, Ytrain):
        Y = np.copy(Ytrain)
        pred = np.zeros_like(Ytrain)
        res_beta_l = []
        pbar = tqdm(range(self.depth))
        for t in pbar:
            pbar.set_description('depth %i' % t)
            Y -= pred
            Problem = Sigmoid(Xtrain, Y, l=1e3, varphi=self.varphi)
            Problem.return_exact_newton = 0
            self.solver.setProblem(Problem)
            res_beta = self.solver.solve()
            res_beta_l.append(res_beta)
            
            pred = self.activation(Xtrain @ res_beta, Problem.varphi)
            pred = pred.reshape(Ytrain.shape)
        self.beta = np.concatenate(res_beta_l, axis=1)
        
    def predict(self, X):
        pred_l = []
        for t in range(self.depth):
            pred = self.activation(X @ self.beta[:,t].reshape(-1,1), [7, 5, 0.01])
            pred_l.append(pred)
            
        pred_l = np.concatenate(pred_l, axis=1)
        res = np.sum(pred_l, axis=1)
        return res