import numpy as np
import sys
sys.path.append('..')

import os
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/..'
sys.path.append(dir_path)

from optimization_algorithms.interface.mathematical_program_traced import MathematicalProgramTraced
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT

class OptimalControl(MathematicalProgram):
    def __init__(self, A, B, N):
        super().__init__()
        self.A = A
        self.B = B
        self.state_dim = A.shape[1]
        self.control_dim = B.shape[1]
        self.dim = A.shape[1] * (N+1) + B.shape[1] * N
        self.N = N # control length

    def evaluate(self, x):
        # x here is stacked x(t) and u(t)
        phi = []
        x = x.reshape(-1,1)
        state = x[:self.state_dim*(self.N+1), :]
        control = x[self.state_dim*(self.N+1):, :]

        obj = np.sum( control ** 2 )
        eq1 = state[:self.state_dim, :]
        eq2 = state[self.state_dim*self.N:, :]

        phi.append(obj)
        phi.append(eq1)
        phi.append(eq2)

        for n in range(self.N):
            x_curr = state[n*self.state_dim:(n+1)*self.state_dim, :].reshape(-1,1)
            x_next = state[(n+1)*self.state_dim:(n+2)*self.state_dim, :].reshape(-1,1)
            u_curr = control[n*self.control_dim:(n+1)*self.control_dim, :].reshape(-1,1)

            eq_n = x_next - self.A @ x_curr - self.B @ u_curr
            for item in eq_n:
                phi.append(item)

        # phi has length: 3 + N
        # here can only regard the control dim as 1
        J_obj = [0. for _ in range(self.state_dim*(self.N+1))] + [2*control[i,:] for i in range(self.N)]
        J_eq1 = [1. for _ in range(self.state_dim)] + [0. for _ in range(self.dim - self.state_dim)]
        J_eq2 = [0. for _ in range(self.dim - self.state_dim)] + [1. for _ in range(self.state_dim)]

        # the jacobian of dynamical constraints is a [x*(N+1) + N*u] * x*N matrix
        J_dyn = np.zeros([self.state_dim*self.N, self.state_dim*(self.N+1)+self.control_dim*self.N])
        for n in range(self.N):
            rows = np.zeros([self.state_dim, self.state_dim*(self.N + 1) + self.control_dim * self.N])
            rows[:, self.state_dim*n:self.state_dim*(n+1)] = -self.A
            rows[:, self.state_dim*(n+1):self.state_dim*(n+2)] = np.eye(self.state_dim)
            rows[:, self.state_dim*(self.N+1) + n * self.control_dim : self.state_dim*(self.N+1) + (n+1) * self.control_dim] = -self.B

            J_dyn[self.state_dim*n:self.state_dim*(n+1), :] = rows


        J = np.concatenate([J_obj, J_eq1], axis=0)
        J = np.concatenate([J, J_eq2], axis=0)
        J = np.concatenate([J, J_dyn], axis=0)
        return phi, J

    def getDimension(self):
        return self.dim

    def getFHessian(self, x):
        H = 0
        return H

    def report(self, verbose):
        if verbose:
            print("Optimal Control Problem Definition")

    def getFeatureTypes(self):
        return [OT.f, OT.eq, OT.eq] + [OT.eq for _ in range(self.N)]

    def getInitializationSample(self):
        dim = self.getDimension()
        return np.ones([dim, 1]) # it could be from phase_1
