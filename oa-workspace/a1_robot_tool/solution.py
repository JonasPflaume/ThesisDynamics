import sys
import math
import numpy as np

sys.path.append("..")
from optimization_algorithms.interface.mathematical_program import MathematicalProgram
from optimization_algorithms.interface.objective_type import OT


class RobotTool(MathematicalProgram):
    """
    """

    def __init__(self, q0, pr, l):
        """
        Arguments
        ----
        q0: 1-D np.array
        pr: 1-D np.array
        l: float
        """
        # in case you want to initialize some class members or so...
        self.q0 = q0
        self.pr = pr
        self.l = l

    def evaluate(self, x):
        """
        See also:
        ----
        MathematicalProgram.evaluate
        """
        
        x = x.reshape(-1,1)
        q0 = self.q0.reshape(-1,1)
        pr = self.pr.reshape(-1,1)
        # add the main code here! E.g. define methods to compute value y and Jacobian J
        cos = np.cos
        sin = np.sin
        q1, q2, q3 = x[0], x[1], x[2]
        p1q = cos(q1) + 0.5 * cos(q1+q2) + 1/3 * cos(q1+q2+q3)
        p2q = sin(q1) + 0.5 * sin(q1+q2) + 1/3 * sin(q1+q2+q3)
        pq = np.array([p1q, p2q])
        pq = pq.reshape(-1,1)
        #y = np.linalg.norm(pq-pr) ** 2 + self.l * np.linalg.norm(x - q0) ** 2
        
        j_11 = - sin(q1) - 1/2 * sin(q1+q2) - 1/3 * sin(q1+q2+q3)
        j_12 = - 1/2 * sin(q1+q2) - 1/3 * sin(q1+q2+q3)
        j_13 = - 1/3 * sin(q1+q2+q3)
        j_21 = cos(q1) + 1/2 * cos(q1+q2) + 1/3 * cos(q1+q2+q3)
        j_22 = 1/2 * cos(q1+q2) + 1/3 * cos(q1+q2+q3)
        j_23 = 1/3 * cos(q1+q2+q3)
        jacobian_f = np.array([[j_11, j_12, j_13],[j_21, j_22, j_23]])
        jacobian_f = jacobian_f.squeeze()

        # y is a 1-D np.array of dimension m
        # J is a 2-D np.array of dimensions (m,n)
        # where m is the number of features and n is dimension of x
        m = len(self.getFeatureTypes())
        n = len(x)
        y = np.zeros([m, ])
        y[0:2] = (pq - pr).squeeze()
        y[2:] = (np.sqrt(self.l)*(x-q0)).squeeze()
        
        J = np.zeros([m, n])
        J[:2, :] = jacobian_f
        J[2:, :] = np.sqrt(self.l) * np.eye(n)
        
        return  y, J

    def getDimension(self):
        """
        See Also
        ------
        MathematicalProgram.getDimension
        """
        # return the input dimensionality of the problem (size of x)
        return 3

    def getInitializationSample(self):
        """
        See Also
        ------
        MathematicalProgram.getInitializationSample
        """
        return self.q0

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.sos] * 5
