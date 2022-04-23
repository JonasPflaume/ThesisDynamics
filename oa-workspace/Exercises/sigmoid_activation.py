import numpy as np
import sys
sys.path.append('..')

import os
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/..'
sys.path.append(dir_path)

from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT

def activation(x, varphi):
    varphi1, varphi2, varphi3 = varphi
    res = varphi1 / (1 + np.exp(-varphi2*(x + varphi3)))
    res -= varphi1 / (1 + np.exp(-varphi2*varphi3))
    return res

class Sigmoid(MathematicalProgram):
    def __init__(self, X, Y, l, varphi):
        super().__init__()
        self.X = X # data matrix
        self.Y = Y # label, [N, 1]
        self.dof = 7
        self.feature_dim = X.shape[1]
        self.N = X.shape[0]
        self.l = l
        self.varphi = varphi # hand tuned
        self.return_exact_newton = 1

    def evaluate(self, x):
        x = x.reshape(-1,1)
        varphi = self.varphi
        varphi1, varphi2, varphi3 = varphi
        beta = x
        Xb = self.X @ beta
        A_Xb = (activation(Xb, varphi)).reshape(-1,1)
        inside = A_Xb - (self.Y).reshape(-1,1)
        phi = 1/2 * np.linalg.norm(inside) ** 2 + 1/2*self.l * np.linalg.norm(beta) ** 2
        
        denomi_exp = np.exp(-varphi2 * (Xb+varphi3)) # N x 1
        # get J_b
        p_phi_p_beta = varphi2*varphi1 / ((1 + denomi_exp)**2) * denomi_exp * (self.X) # N x p
        self.p_phi_p_beta = p_phi_p_beta
        J_b = (A_Xb.T - self.Y.T) @ p_phi_p_beta + self.l * beta.T
        J = J_b
        if self.return_exact_newton:
            scalar_term = denomi_exp * (2*varphi1/(1+denomi_exp)**3 + -varphi1/(1+denomi_exp)**2) # N x 1
            matrix_term = 0
            for i in range(self.N):
                matrix_term += scalar_term[i] * varphi2 ** 2 * np.outer(self.X[i], self.X[i]) * (A_Xb[i] - self.Y[i])
            self.pp_phi_pp_beta = matrix_term # p x p

        return np.array(phi).reshape(1,1), J

    def getDimension(self):
        return self.feature_dim

    def getFHessian(self, x):
        if self.return_exact_newton:
            H = self.p_phi_p_beta.T @ self.p_phi_p_beta + self.l * np.eye(self.feature_dim) + self.pp_phi_pp_beta
        else:
            H = self.p_phi_p_beta.T @ self.p_phi_p_beta + self.l * np.eye(self.feature_dim)
        return H

    def report(self, verbose):
        if verbose:
            print("Sigmoid activation...")

    def getInitializationSample(self):
        dim = self.getDimension()
        return np.random.uniform(low=-1,high=1,size=[dim,1])

    def getFeatureTypes(self):
        return [OT.f]

def activation_cos(x, amp):
    res = amp*np.cos(x)
    return res

class Cosine(MathematicalProgram):
    def __init__(self, X, Y, l, amp):
        super().__init__()
        self.X = X # data matrix
        self.Y = Y # label, [N, 1]
        self.dof = 7
        self.feature_dim = X.shape[1]
        self.N = X.shape[0]
        self.l = l
        self.amp = amp # hand tuned

    def evaluate(self, x):
        x = x.reshape(-1,1)
        amp = self.amp
        beta = x
        Xb = self.X @ beta
        A_Xb = (activation_cos(Xb, amp)).reshape(-1,1)
        inside = A_Xb - (self.Y).reshape(-1,1)
        phi = 1/2 * np.linalg.norm(inside) ** 2 + 1/2*self.l * np.linalg.norm(beta) ** 2
        # get J_b
        sinXb = np.sin(Xb)
        cosXb = np.cos(Xb)
        J = (self.Y.T - A_Xb.T) @ (sinXb * self.X) + self.l * beta.T

        self.H = 0
        scalar = sinXb **2 - (cosXb - self.Y.reshape(-1,1)) * cosXb
        self.H += self.X.T @ np.diag(scalar.squeeze()) @ self.X

        self.H += self.l * np.eye(self.feature_dim)

        return np.array(phi).reshape(1,1), J

    def getDimension(self):
        return self.feature_dim

    def getFHessian(self, x):
        
        return self.H

    def report(self, verbose):
        if verbose:
            print("cos activation...")

    def getInitializationSample(self):
        dim = self.getDimension()
        return np.random.uniform(low=-1,high=1,size=[dim,1])

    def getFeatureTypes(self):
        return [OT.f]
