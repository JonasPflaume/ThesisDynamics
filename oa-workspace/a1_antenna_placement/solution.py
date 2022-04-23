import math
import sys
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class AntennaPlacement(MathematicalProgram):
    """
    """

    def __init__(self, P, w):
        """
        Arguments
        ----
        P: list of 1-D np.arrays
        w: 1-D np.array
        """
        # in case you want to initialize some class members or so...
        self.dim = 2
        self.P = P
        self.w = w

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """

        # add the main code here! E.g. define methods to compute value y and Jacobian J
        # y = ...
        # J = ...
        y = 0
        num = len(self.P)
        for j in range(num):
            y -= self.w[j] * np.exp(-np.linalg.norm(x-self.P[j])**2)
        
        J = 0
        for j in range(num):
            J += self.w[j] * np.exp(-np.linalg.norm(x-self.P[j])**2) * (2*x.T - 2*self.P[j].T)
        return np.array(y).reshape(1,) , J[np.newaxis, :]

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return self.dim

    def getFHessian(self, x):
        """
        See Also
        ------
        MathematicalProgram.getFHessian
        """
        # add code to compute the Hessian matrix
        H = 0
        num = len(self.P)
        x = x.reshape(-1,1)
        for j in range(num):
            temp_P = self.P[j].reshape(-1,1)
            exp_term = np.exp(-np.linalg.norm(x-temp_P) ** 2)
            H -= self.w[j] * exp_term * (2 * x - 2 * temp_P) @ (2 * x - 2 * temp_P).T
            H += 2 * self.w[j] * exp_term * np.eye(self.dim)
        return H

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        x0 = 0
        num = len(self.P)
        for i in range(num):
            x0 += self.P[i]
        x0 *= 1/num
        return x0

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types

        """
        return [OT.f]
