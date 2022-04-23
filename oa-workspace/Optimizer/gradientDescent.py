import numpy as np
import sys
sys.path.append('..')
from optimization_algorithms.interface.nlp_solver import NLPSolver
from .backtracking import BackTracking

class PlainGD(NLPSolver):
    def __init__(self, problem, tolerance):
        super().__init__()
        self.tolerance = tolerance
        self.problem = problem

    def solve(self):
        x = self.problem.getInitializationSample()
        x0 = x.reshape(2,1)
        phi, J = self.problem.evaluate(x0)
        lr = 0.1
        i = 0
        while True:
            direction = -J[0].reshape(2,1)
            x1 = x0 + lr * direction
            if np.linalg.norm(lr * J[0].reshape(-1,1)) < self.tolerance:
                break
            phi, J = self.problem.evaluate(x1)
            print('Current cost: {0} at step {1}'.format(phi[0], i))
            x0 = x1
            i += 1

        print("Finally converged at x: {}, final cost: {}".format(x0, phi[0]))

        return x0


class BackTrackingGD(NLPSolver):
    def __init__(self, problem, tolerance, rho_a_0=0.5, rho_a_1=1.2, sigma_max=10, rho_ls=0.01):
        super().__init__()
        self.setProblem(problem)
        self.backtracking = BackTracking(problem, rho_a_0, rho_a_1, sigma_max, rho_ls)
        self.tolerance = tolerance

    def solve(self):
        x = self.problem.getInitializationSample()
        x0 = x.reshape(2,1)
        phi, J = self.problem.evaluate(x0)
        lr = 10
        i = 0
        while True:
            ####
            direction = -J[0].reshape(2,1)
            lr = self.backtracking.decrease_lr(lr, x0, direction)
            ####
            x1 = x0 + lr * direction
            ####
            lr = self.backtracking.increase_lr(lr)
            ####
            if np.linalg.norm(lr * J[0].reshape(-1,1)) < self.tolerance:
                break
            phi, J = self.problem.evaluate(x1)
            print('Current cost: {0} at step {1}'.format(phi[0], i))
            x0 = x1
            i += 1

        print("Finally converged at x: {}, final cost: {}".format(x0, phi[0]))

        return x0