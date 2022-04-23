import numpy as np
import sys
sys.path.append('..')
from optimization_algorithms.interface.mathematical_program_traced import MathematicalProgramTraced
from optimization_algorithms.interface.mathematical_program import MathematicalProgram

class Rastrigin(MathematicalProgram):
    def __init__(self, a, c):
        super().__init__()
        self.a = a
        self.c = c # condition number

    def evaluate(self, x):
        a = self.a
        c = self.c
        # the objective function
        x1 = x[0]
        x2 = x[1]
        feature = np.array([np.sin(a*x1), np.sin(a*c*x2), 2*x1, 2*c*x2]).reshape(-1,1)
        phi = feature.T @ feature
        # jacobian
        feature_jacobian = self.featureJacobian(x)
        J = 2 * feature_jacobian.T @ feature
        return [phi.squeeze()], [J.T]

    def featureJacobian(self, x):
        dim = max(x.shape)
        a = self.a
        c = self.c
        x1 = x[0]
        x2 = x[1]

        feature_jacobian = [a*np.cos(a*x1), 0, 2, 0, 0, a*c*np.cos(a*c*x2), 0, 2*c]
        feature_jacobian = np.array(feature_jacobian, dtype=float).reshape(-1, dim)
        return feature_jacobian

    def getDimension(self):
        return 2

    def getFHessian(self, x):
        feature_jacobian = self.featureJacobian(x)
        # approximate the hessian by using the feature jacobian
        H = 2 * feature_jacobian.T @ feature_jacobian
        return H

    def report(self, verbose):
        if verbose:
            print("Hello")

    def getInitializationSample(self):
        dim = self.getDimension()
        return np.random.uniform(low=-1,high=1,size=[dim,1])



if __name__ == "__main__":
    a = 5
    c = 3
    Problem = Rastrigin(a, c)

    def RastriginFunc(x):
        x1 = x[0]
        x2 = x[1]
        feature = np.array([np.sin(a*x1), np.sin(a*c*x2), 2*x1, 2*c*x2]).reshape(-1,1)
        phi = feature.T @ feature
        return phi

    from Optimizer.gradientDescent import BackTrackingGD, PlainGD
    from Optimizer.newton import Newton
    tolerance = 1e-4
    Problem = MathematicalProgramTraced(Problem)

    solver = Newton(Problem, tolerance)
    solver.solve()

    from Plot.plot import plotFunc
    plotFunc(RastriginFunc, [-1,-1], [1,1], trace_xy=Problem.trace_x, trace_z=Problem.trace_phi)