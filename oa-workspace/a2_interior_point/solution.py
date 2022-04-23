import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverInteriorPoint(NLPSolver):

    def __init__(self):
        self.tolerance = 1e-4
        self.verbose = 0
        self.backtracking = 1

    def solve(self):
        FT = self.problem.getFeatureTypes()
        self.obj_idx = [i for i, item in enumerate(FT) if item == 1]
        self.sos_idx = [i for i, item in enumerate(FT) if item == 2]
        self.inq_idx = [i for i, item in enumerate(FT) if item == 3]

        mu = 1
        mu_re = 0.5
        x0 = self.problem.getInitializationSample()
        x0 = x0.reshape(-1, 1)
        num_ineq = len(self.inq_idx)
        for i in range(1000):
            # run the unconstrained optimizer
            x1 = self.solveUC(x0, mu)
            if self.verbose:
                print("___ {0}th iteration ___".format(i), "current q: ", self.curr_q, "current x: ", x1)
            # duality criterion
            if mu * num_ineq < self.tolerance:
                break
            x0 = x1
            # reduce the mu
            mu *= mu_re

            if self.curr_q > 10000:
                print("Exceed the maximum allowed query time...")
                break
            
            # constraints violation check, only for debug
            if self.verbose:
                c, _, = self.problem.evaluate(x1.squeeze())
                if not np.all(c[self.inq_idx] < 0):
                    print('There are constaints violation!')
        
        print('Query time: %i' %self.curr_q)
        
        return x1

    def solveUC(self, x0, mu):
        tolerance = 1e-4
        damping = 0
        verbose = self.verbose
        damping_decrease = 0.1

        rho_a_in = 1.2
        rho_a_de = 0.5
        sigmax = 10
        rho_ls = 0.01
        lr = 1
        phi = float('Inf')

        def BTincrease(lr):
            return np.min([lr * rho_a_in, sigmax])

        def BTdecrease(lr, curr_x, direction):
            while True:
                lhs_x = curr_x + lr * direction
                c, J, _, _ = self.LogBarrierEvaluate(lhs_x, mu)
                lhs = c
                rhs, J, _, _ = self.LogBarrierEvaluate(curr_x, mu)
                grad = J.reshape(-1,1)
                direction = direction.reshape(-1,1)
                rhs += rho_ls * grad.T @ (lr * direction)
                if lr < 1e-16:
                    raise ValueError('Fail to find a proper lr...')
                if lhs > rhs or np.isnan(lhs):
                    lr *= rho_a_de
                else:
                    break
            return lr

        def get_direction(J, H):
            grad = J.reshape(-1,1)
            try:
                direction = -np.linalg.inv(H) @ grad
            except:
                direction = -grad

            if grad.T @ direction > 0:
                print('---PULLBACK---')
                direction = -grad
            return direction

        x0 = x0.reshape(-1,1)

        for i in range(1000):
            if self.verbose:
                print('step {}, f = {}'.format(i, phi))
            # 1. get the update direction
            phi, J, H, CT_NOT_VIO = self.LogBarrierEvaluate(x0, mu)
            direction = get_direction(J, H)

            # 2. run the BT linear search
            if self.backtracking:
                lr = BTdecrease(lr, x0, direction)
            
            # 4. break loop
            if np.linalg.norm(lr*direction) < tolerance and CT_NOT_VIO:
                x1 = x0
                break
            # 3. update x
            x1 = x0 + lr * direction
            
            x0 = x1
            if self.backtracking:
                lr = BTincrease(lr)
        return x1

    def LogBarrierEvaluate(self, x, mu):
        # return the uc phi, J and H
        obj_idx = self.obj_idx
        sos_idx = self.sos_idx
        inq_idx = self.inq_idx

        phi = []
        J = np.zeros([1, self.problem.getDimension()])
        H = np.zeros([self.problem.getDimension(), self.problem.getDimension()])

        phi0, J0 = self.problem.evaluate(x.squeeze())
        CT_NOT_VIO = np.all(phi0[inq_idx] < 0)
        try:
            H_f0 = self.problem.getFHessian(x.squeeze())
        except:
            H_f0 = 0

        # 1. get the phi_obj
        log_term = -1 * mu * np.sum(np.log(-phi0[inq_idx]))
        obj_term = np.sum(phi0[obj_idx])
        phi_obj = obj_term + log_term

        phi_obj += np.inner(phi0[sos_idx], phi0[sos_idx])
        phi.append([phi_obj])
        assert len(phi) == 1
        # 2. compute the J
        # obj
        for i in obj_idx:
            J += J0[i]
        # sos term
        Fsos = phi0[sos_idx].reshape(-1,1)
        DxDFsos = J0[sos_idx].reshape(len(sos_idx), self.problem.getDimension())
        J += 2 * (Fsos.T @ DxDFsos)
        # ieq term
        H_inq = 0
        for idx in inq_idx:
            J -= mu * 1/phi0[idx] * J0[idx]
            H_inq += mu / (phi0[idx]**2) * np.outer(J0[idx], J0[idx])
        assert J.shape[0] == 1
        
        # 3. compute the H
        H += H_f0
        H += 2 * DxDFsos.T @ DxDFsos
        H += H_inq
        return phi, J, H, CT_NOT_VIO

    @property
    def curr_q(self):
        if hasattr(self.problem, 'counter_evaluate'):
            return self.problem.counter_evaluate
        else:
            return 0
