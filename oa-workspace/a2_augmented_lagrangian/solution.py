import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverAugmentedLagrangian(NLPSolver):

    def __init__(self):
        self.theta = 1e-4
        self.epsi = 1e-4
        self.verbose = 0

    @property
    def curr_q(self):
        return self.problem.counter_evaluate

    def solve(self):
        FT = self.problem.getFeatureTypes()
        self.obj_idx = [i for i, item in enumerate(FT) if item == 1]
        self.sos_idx = [i for i, item in enumerate(FT) if item == 2]
        self.inq_idx = [i for i, item in enumerate(FT) if item == 3]
        self.eq_idx = [i for i, item in enumerate(FT) if item == 4]
        # initialize the parameters
        rho_mu = rho_v = 3
        mu = v = 1
        labd, kappa = np.zeros(len(self.inq_idx)), np.zeros(len(self.eq_idx))
        x0 = self.problem.getInitializationSample()
        x0 = x0.reshape(-1,1)
        for i in range(1000):
            if self.verbose:
                print("___%i iteration__ curr_q: %i" % (i, self.curr_q))
            # 1. solve the unconstrained optimization
            x1 = self.solveUC(x0, mu, labd, v, kappa)
            # 2. update the parameters
            phi, J = self.problem.evaluate(x1.squeeze())
            g_x, h_x = phi[self.inq_idx], phi[self.eq_idx]

            labd = np.maximum(labd + 2 * mu * g_x, np.zeros_like(labd))
            kappa += 2 * v * h_x
            # 3. * try to update the mu and v (LP:this will increase the complexity in inner loop)
            mu *= rho_mu
            v *= rho_v
            # 4. check the stop criterion
            if np.linalg.norm(x1-x0) < self.theta and np.all(g_x < self.epsi) and np.all(np.abs(h_x) < self.epsi):
                break

            if self.curr_q > 10000:
                print('Exceed the maximun query time...')
                break
            x0 = x1
        print("Query time: %i" % self.curr_q)
        return x1
        
    def solveUC(self, x0, mu, labd, v, kappa):
        tolerance = 1e-4
        damping = 0
        verbose = self.verbose
        damping_decrease = 0.1

        rho_a_in = 1.2
        rho_a_de = 0.5
        sigmax = 10
        rho_ls = 0.01
        lr = 1

        def BTincrease(lr):
            return np.min([lr * rho_a_in, sigmax])

        def BTdecrease(lr, curr_x, direction):
            while True:
                lhs_x = curr_x + lr * direction
                c, J, _= self.AulaEvaluate(lhs_x, mu, labd, v, kappa)
                lhs = c
                rhs, J, _= self.AulaEvaluate(curr_x, mu, labd, v, kappa)
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

            PULLBACK = False
            if grad.T @ direction > 0:
                if self.verbose:
                    print('---PULLBACK---')
                direction = -grad
                PULLBACK = True
            return direction, PULLBACK

        x0 = x0.reshape(-1,1)
        PB_counter = 0
        for i in range(1000):
            if self.verbose:
                print('step {}, x = {}'.format(i, x0))
            # 1. get the update direction
            phi, J, H = self.AulaEvaluate(x0, mu, labd, v, kappa)
            direction, PB = get_direction(J, H)
            # break loop if too many pullback
            if bool(PB):
                PB_counter += int(PB)
            else:
                PB_counter = 0
            # 2. run the BT linear search
            lr = BTdecrease(lr, x0, direction)
            
            # 4. break loop
            if np.linalg.norm(lr*direction) < tolerance or PB_counter > 5:
                x1 = x0
                break
            # 3. update x
            x1 = x0 + lr * direction
            
            x0 = x1
            lr = BTincrease(lr)
        return x1

    def AulaEvaluate(self, x, mu, labd, v, kappa):
        labd = labd.reshape(-1,1)
        kappa = kappa.reshape(-1,1)
        # return the uc phi, J and H
        obj_idx = self.obj_idx
        sos_idx = self.sos_idx
        inq_idx = self.inq_idx
        eq_idx = self.eq_idx

        phi = []
        J = np.zeros([1, self.problem.getDimension()])
        H = np.zeros([self.problem.getDimension(), self.problem.getDimension()])

        phi0, J0 = self.problem.evaluate(x.squeeze())
        try:
            H_f0 = self.problem.getFHessian(x.squeeze())
        except:
            H_f0 = 0
        # get the necessary value, grad and ggrad
        I_labd = np.logical_or(phi0[inq_idx] >= 0, labd.squeeze() > 0)
        I_labd = np.array(I_labd, dtype=int)
        I_labd = np.diag( I_labd )
        
        #print(I_labd)
        g_x = phi0[inq_idx].reshape(-1,1)
        h_x = phi0[eq_idx].reshape(-1,1)
        grad_f_x = J0[obj_idx].reshape(1,-1)
        grad_g_x = J0[inq_idx]
        grad_h_x = J0[eq_idx]
        # 1. get the phi_obj
        obj_term = np.sum(phi0[obj_idx])
        inq_term = (mu * I_labd @ g_x + labd).T @ g_x
        eq_term = (v * h_x + kappa).T @ h_x
        sos_term = np.inner(phi0[sos_idx], phi0[sos_idx])
        phi = obj_term + inq_term + eq_term + sos_term
        assert len(phi) == 1
        # 2. compute the J
        # obj
        if obj_idx:
            J += grad_f_x
        # sos term
        Fsos = phi0[sos_idx].reshape(-1,1)
        DxDFsos = J0[sos_idx].reshape(len(sos_idx), self.problem.getDimension())
        J += 2 * (Fsos.T @ DxDFsos)
        # ieq term
        J += (2 * mu * I_labd @ g_x + labd).T @ grad_g_x
        # eq term
        J += (2 * v * h_x + kappa).T @ grad_h_x
        assert J.shape[0] == 1
        # 3. compute the H
        H += H_f0
        H += 2 * DxDFsos.T @ DxDFsos
        # inq term
        H += 2 * mu * grad_g_x.T @ I_labd @ grad_g_x
        # eq term
        H += 2 * v * grad_h_x.T @ grad_h_x
        return phi, J, H