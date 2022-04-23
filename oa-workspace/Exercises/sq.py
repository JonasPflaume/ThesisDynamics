import numpy as np
import sys
sys.path.append('..')
from optimization_algorithms.interface.mathematical_program_traced import MathematicalProgramTraced
from optimization_algorithms.interface.mathematical_program import MathematicalProgram

class Sq(MathematicalProgram):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def evaluate(self, x):
        C = self.C
        phi = x.T @ C @ x
        J = 2 * C @ x
        J = J.reshape(1,-1)
        return [phi.squeeze()], J

    def getDimension(self):
        return 2

    def getFHessian(self, x):
        x = x.reshape(-1,1)
        C = self.C
        H = 2 * C
        return H

    def report(self, verbose):
        if verbose:
            print("Hello")




if __name__ == "__main__":
    C = np.array([[1,0],[0,10]])
    Problem = Sq(C)

    def holeFunc(x):
        C = np.array([[1,0],[0,10]])
        return x.T @ C @ x

    from Optimizer.gradientDescent import BackTrackingGD, PlainGD
    from Optimizer.newton import Newton
    tolerance = 1e-4
    Problem = MathematicalProgramTraced(Problem)

    solver = Newton(Problem, tolerance)
    solver.solve()

    from Plot.plot import plotFunc
    plotFunc(holeFunc, [-5,-5], [5,5], trace_xy=Problem.trace_x, trace_z=Problem.trace_phi)