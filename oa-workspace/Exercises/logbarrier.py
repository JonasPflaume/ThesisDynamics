import numpy as np
import sys
sys.path.append('..')
from optimization_algorithms.interface.mathematical_program_traced import MathematicalProgramTraced
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT

class CircleConstrained(MathematicalProgram):
    def __init__(self, res):
        super().__init__()
        self.res = res

    def evaluate(self, x):
        phi1 = np.sum(x)
        phi2 = x.T @ x - 1
        phi3 = -x[0] + self.res

        J = np.array([[1, 2*x[0], -1],[1, 2*x[1], 0]], dtype=float)
        
        return np.array([phi1, phi2, phi3]), J

    def getDimension(self):
        return 2

    def getFHessian(self, x):
        H = 0
        return H

    def report(self, verbose):
        if verbose:
            print("Hello")

    def getFeatureTypes(self):
        return [OT.f, OT.ineq, OT.ineq]

    def getInitializationSample(self):
        dim = self.getDimension()
        return np.ones([dim, 1]) * 0.5

    def sensitivity(self, x, labd):
        x1 = x[0]
        x2 = x[1]
        l1 = labd[0]
        l2 = labd[1]
        phi, J = self.evaluate(x)
        J = J.T
        obj, g1, g2 = phi[0], phi[1], phi[2]
        DobjDx, Dg1Dx, Dg2Dx = J[0], J[1], J[2]

        # KKT matrix
        KKT_M = [2*l1, 0, 2*x1, -1, 0, 2*l2, 0, 2*x2, 2*l1*x1, 2*l1*x2, x1**2+x2**2-1, 0, -l2, 0, 0,-x1+self.res]
        KKT_M = np.array(KKT_M, dtype=float).reshape(4,4)
        # KKT vector
        KKT_V = [0,0,0,l2]
        KKT_V = np.array(KKT_V,dtype=float).reshape(4,1)

        # \partial(F)/\partial(theta)
        PFPT = - np.linalg.inv(KKT_M) @ KKT_V
        xTheta = PFPT[:2,:]
        labdTheta = PFPT[2:,:]
        return xTheta, labdTheta



class LogBarrierProblem(MathematicalProgram):
    def __init__(self, mu, init_value, res=0.):
        super().__init__()
        self.original = CircleConstrained(res)
        self.mu = mu
        self.init_value = init_value

    def evaluate(self, x):
        phi0, J0 = self.original.evaluate(x)
        J0 = J0.T # (phi:3,x:2)
        DobjDx = J0[0]
        Dg1Dx = J0[1]
        Dg2Dx = J0[2]
        obj = phi0[0]
        con1 = phi0[1]
        con2 = phi0[2]
        phi = obj - self.mu * np.log(-con1) - self.mu * np.log(-con2)
        J = DobjDx.T - self.mu * (1/con1 * Dg1Dx.T + 1/con2 * Dg2Dx.T)
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


def get_labd(x, mu, res):
    term1 = - mu / (x.T @ x - 1)
    term2 = - mu / (-x[0] + res)
    return np.array([[term1],[term2]])


if __name__ == "__main__":

    from Optimizer.gradientDescent import BackTrackingGD, PlainGD
    import time

    mu = 1.
    old_res = np.ones([2,1]) * 1000

    center_list = []
    labda_list = []
    start = time.time()
    res = 0.5
    # main loop
    for i in range(10000):
        if i == 0:
            Problem = LogBarrierProblem(mu, np.array([[0.99],[0.]]), res=res)
        else:
            Problem = LogBarrierProblem(mu, curr_res.reshape(-1,1), res=res)

        tolerance = 1e-5
        Problem = MathematicalProgramTraced(Problem)

        solver = BackTrackingGD(Problem, tolerance)
        print("<<<<<<<<<<<<<<<<<<< {}. iteration >>>>>>>>>>>>>>>>>>".format(i))
        curr_res = solver.solve()

        if np.linalg.norm( old_res - curr_res ) < 1e-8:
            break
        
        curr_labd = get_labd(curr_res, mu, res)
        old_res = curr_res
        center_list.append(curr_res)

        labda_list.append(curr_labd)
        mu *= 0.5

    end = time.time()
    print('### Time consuming: ', end - start)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    circle = plt.Circle((0, 0), 1., color='r', fill=False)

    for center in center_list:
        ax.scatter(center[0], center[1])
    ax.plot([0+res,0+res], [-1,1])
    ax.add_patch(circle)
    plt.show()


    fig, ax = plt.subplots()

    for labd in labda_list:
        ax.scatter(labd[0], labd[1])
    plt.show()
