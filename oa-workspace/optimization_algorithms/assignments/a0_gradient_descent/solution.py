import numpy as np
import sys
sys.path.append("..")

from optimization_algorithms.interface.nlp_solver import NLPSolver


class Solver0(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """

        # in case you want to initialize some class members or so...

    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """

        # write your code here

        # use the following to get an initialization:
        x = self.problem.getInitializationSample()

        # use the following to query the problem:
        phi, J = self.problem.evaluate(x)
        # phi is a vector (1D np.array); use phi[0] to access the cost value (a float number). J is a Jacobian matrix (2D np.array). Use J[0] to access the gradient (1D np.array) of the cost value.

        # now code some loop that iteratively queries the problem and updates x til convergenc....

        # finally:
        return x
