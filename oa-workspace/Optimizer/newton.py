import numpy as np
import sys
sys.path.append('..')
from optimization_algorithms.interface.nlp_solver import NLPSolver
from .backtracking import BackTracking

class Newton(NLPSolver):
    def __init__(self, problem, tolerance, damping=0.1, rho_a_0=0.5, rho_a_1=1.2, sigma_max=1, rho_ls=0.01, pullback=True):
        super().__init__()
        self.pullback = pullback
        self.setProblem(problem)
        self.backtracking = BackTracking(problem, rho_a_0, rho_a_1, sigma_max, rho_ls)
        self.tolerance = tolerance
        self.damping = damping

    def solve(self):
        x = self.problem.getInitializationSample()
        x = x.reshape(2,1)
        phi, J = self.problem.evaluate(x)
        H = self.problem.getFHessian(x)
        x0 = x
        lr = 1 # initialize as 1
        gradient = J[0].T
        direction = self.get_direction(x0, H, gradient)
        i = 0
        while True:
            ####
            lr = self.backtracking.decrease_lr(lr, x0, direction)
            ####
            x1 = x0 + direction * lr
            ####
            lr = self.backtracking.increase_lr(lr)
            ####
            if np.linalg.norm(lr * direction) < self.tolerance:
                break
            x0 = x1
            phi, J = self.problem.evaluate(x0)
            H = self.problem.getFHessian(x0)
            gradient = J[0].T
            direction = self.get_direction(x0, H, gradient)
            print('Current cost: {0:.4f} at step {1}'.format(phi[0], i))
            i += 1

        print("Finally converged at x: {}, final cost: {}".format(x0, phi[0]))

        return x0

    def get_direction(self, x, H, gradient):
        x = x.reshape(2,1)
        if self.pullback:
            try:
                direction = -np.linalg.inv(H + np.eye(H.shape[0]) * self.damping) @ gradient
            except:
                direction = -gradient / np.linalg.norm(gradient)
            finally:
                if gradient.T @ direction > 0:
                    print("Pull back!")
                    direction = -gradient / np.linalg.norm(gradient)
        else:
            direction = -np.linalg.inv(H + np.eye(H.shape[0]) * self.damping) @ gradient
        direction = direction.reshape(2,1)
        return direction