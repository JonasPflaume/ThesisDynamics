import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class LQR(MathematicalProgram):
    def __init__(self, K, A, B, Q, R, yf):
        # in case you want to initialize some class members or so...
        self.A, self.B = A, B
        self.K = K
        self.Q, self.R = Q, R
        self.yf = yf
        self.N = A.shape[0]
        self.counter_evaluate = 0

        H = []
        for _ in range(K):
            H.append(R)
            H.append(Q)
        H = LQR.build_block_diag(H)

        self.H = H

    def evaluate(self, x):
        self.counter_evaluate += 1
        A, B, Q, R, N, yf, K \
            = self.A, self.B, self.Q, self.R, self.N, self.yf, self.K
        OBJ_Matrix = self.H

        x = x.reshape(-1,1)
        yf = yf.reshape(-1,1)
        phi = []
        J = []
        # obj value
        obj = 1/2 * x.T @ OBJ_Matrix @ x
        phi.append(obj)
        # constraints value
        first_cons_v = self.getY(x, 1) - B @ self.getU(x, 0)
        phi.append(first_cons_v)

        for k in range(1, K):
            temp_c = self.getY(x, k+1) - A @ self.getY(x, k) - B @ self.getU(x, k)
            phi.append(temp_c)

        last_cons_v = self.getY(x, K) - yf
        phi.append(last_cons_v)
        phi = np.vstack(phi).squeeze()

        # Jacobian of obj
        J_obj = x.T @ OBJ_Matrix
        J.append(J_obj)
        
        # Jacobian of constraints
        J_cons_fist = [-B, np.eye(N)] + [np.zeros_like(B) for _ in range(2*K-2)]
        J_cons_fist = np.block(J_cons_fist)
        J.append(J_cons_fist)

        for k in range(1, K):
            J_cons_k = [np.zeros_like(B) for _ in range(2*k-1)] + [-A, -B, np.eye(N)]
            J_cons_k += [np.zeros_like(B) for _ in range(2*K - len(J_cons_k))]
            J_cons_k = np.block(J_cons_k)
            J.append(J_cons_k)

        J_cons_last = [np.zeros_like(B) for _ in range(2*K-1)] + [np.eye(N)]
        J_cons_last = np.block(J_cons_last)
        J.append(J_cons_last)

        J = np.concatenate(J)
            
        return phi, J

    def getFHessian(self, x):
        return self.H

    def getDimension(self):
        return 2*self.K*self.N

    def getInitializationSample(self):
        return np.zeros(self.getDimension())

    def getFeatureTypes(self):
        FT = [OT.f] + [OT.eq for _ in range((self.K+1)*self.N)]
        return FT

    @staticmethod
    def build_block_diag(matrix_list:list):
        res = []
        for i, item in enumerate(matrix_list):
            temp = [np.zeros_like(item) for _ in range(i)] + [item] + [np.zeros_like(item) for _ in range(len(matrix_list)-i-1)]
            res.append(temp)
        return np.block(res)

    def getY(self, x, idx):
        # x = (2kn, 1), idx match the index in given formula
        idx_in_x = 2 * idx - 1
        return x[idx_in_x*self.N:(idx_in_x+1)*self.N, :]

    def getU(self, x, idx):
        idx_in_x = 2 * idx
        return x[idx_in_x*self.N:(idx_in_x+1)*self.N, :]


# K = 11
# A = np.identity(2)
# B = 1.5 * np.identity(2)
# Q = 1.8 * np.identity(2)
# R = 1.9 * np.identity(2)
# yf = 2 * np.ones(2)

# problem = LQR(K, A, B, Q, R, yf)

# from a2_augmented_lagrangian.solution import SolverAugmentedLagrangian

# solver = SolverAugmentedLagrangian()

# solver.setProblem(problem)
# import time
# start = time.time()
# opt_x = solver.solve()
# end = time.time()

# print('the time consuming is: %.2f' % (end-start))
# print('optimal x: ', opt_x)