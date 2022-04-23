import numpy as np
import sys
sys.path.append('..')
from optimization_algorithms.interface.mathematical_program_traced import MathematicalProgramTraced
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT

from logbarrier import CircleConstrained

class SquarePenaltyProblem(MathematicalProgram):
    def __init__(self, mu, init_value):
        super().__init__()
        self.original = CircleConstrained()
        self.mu = mu
        self.init_value = init_value

    def evaluate(self, x):
        phi0, J0 = self.original.evaluate(x) # [f, g1, g2], jacobian
        J0 = J0.T
        phi0 = np.array(phi0, dtype=float)
        idx = phi0[1:] < 0
        phi0[1:][idx] = 0.
        phi = phi0[0] + self.mu * (phi0[1]**2 + phi0[2]**2)
        J = J0[0] + 2 * self.mu * (phi0[1] * J0[1] + phi0[2] * J0[2])
        J = J.reshape(1, -1)
        return [phi], J


    def getDimension(self):
        return self.original.getDimension()

    def getFHessian(self, x):
        return self.original.getFHessian(x)

    def report(self, verbose):
        if verbose:
            print("Hello")

    def getFeatureTypes(self):
        return [OT.f]

    def getInitializationSample(self):
        return self.init_value



if __name__ == "__main__":

    from Optimizer.gradientDescent import BackTrackingGD, PlainGD

    mu = 1.
    old_res = np.ones([2,1]) * 1000

    center_list = []

    for i in range(10000):
        if i == 0:
            Problem_ = SquarePenaltyProblem(mu, np.array([[0.5],[0.5]]))
        else:
            Problem_ = SquarePenaltyProblem(mu, curr_res.reshape(-1,1))

        tolerance = 1e-8
        Problem = MathematicalProgramTraced(Problem_)

        solver = BackTrackingGD(Problem, tolerance)
        print("<<<<<<<<<<<<<<<<<<< {}. iteration >>>>>>>>>>>>>>>>>>>".format(i))
        curr_res = solver.solve()

        curr_phi, _ = Problem_.original.evaluate(curr_res)

        if np.linalg.norm( old_res - curr_res ) < tolerance and curr_phi[1] < tolerance and curr_phi[2] < tolerance:
            break

        old_res = curr_res
        center_list.append(curr_res)

        mu *= 10


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    circle = plt.Circle((0, 0), 1., color='r', fill=False)

    for center in center_list:
        ax.scatter(center[0], center[1])
    ax.plot([0,0], [-1,1])
    ax.add_patch(circle)
    plt.show()