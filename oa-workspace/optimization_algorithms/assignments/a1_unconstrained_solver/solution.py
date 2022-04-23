import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT


class SolverUnconstrained(NLPSolver):

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
        # get feature types
        # ot[i] inidicates the type of feature i (either OT.f or OT.sos)
        # there is at most one feature of type OT.f
        ot = self.problem.getFeatureTypes()

        # use the following to query the problem:
        phi, J = self.problem.evaluate(x)
        H = self.problem.getFHessian(x)  # if necessary
        # phi is a vector (1D np.array); J is a Jacobian matrix (2D np.array).

        # now code some loop that iteratively queries the problem and updates x til convergenc....

        # finally:
        return x
