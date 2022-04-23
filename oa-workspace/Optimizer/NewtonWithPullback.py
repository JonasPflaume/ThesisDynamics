import numpy as np
import sys

sys.path.append("..")
from optimization_algorithms.interface.nlp_solver import NLPSolver
from optimization_algorithms.interface.objective_type import OT

def getCJH(problem, x):
        # helper function to get the CJH value

        types = problem.getFeatureTypes()
        index_f = [i for i, x in enumerate(types) if x == OT.f]
        assert( len(index_f) <= 1 )
        index_r = [i for i, x in enumerate(types) if x == OT.sos]

        phi, J = problem.evaluate(x)
        try:
            H = problem.getFHessian(x)  # if necessary
        except:
            H = None

        # calculate the Jacobian and hessian in different situation
        Jacobian = np.zeros([1, problem.getDimension()])
        if len(index_f) > 0:
            Jacobian += J[index_f]
            if len(index_r) > 0:
                Jacobian += 2 * J[index_r].T @ phi[index_r]
                H += 2 * J[index_r].T @ J[index_r]
        else:
            if len(index_r) > 0:
                H = 2 * J[index_r].T @ J[index_r]
                Jacobian = 2 * J[index_r].T @ phi[index_r]

        c = 0
        if len(index_f) > 0 :
            c += phi[index_f][0]
        if len(index_r) > 0 :
            c += phi[index_r].T @ phi[index_r]

        return c, Jacobian, H # CJH (cost, Jacobian, Hessian)

class BackTracking(object):
    '''
    Backtracking class
    '''
    def __init__(self, problem, rho_a_0, rho_a_1, sigma_max, rho_ls):
        self.rho_a_0 = rho_a_0
        self.rho_a_1 = rho_a_1
        self.sigma_max = sigma_max
        self.rho_ls = rho_ls
        self.problem = problem

    def decrease_lr(self, lr, curr_x, direction):

        types = self.problem.getFeatureTypes()
        index_f = [i for i, x in enumerate(types) if x == OT.f]
        assert( len(index_f) <= 1 )
        index_r = [i for i, x in enumerate(types) if x == OT.sos]

        while True:
            lhs_x = curr_x + lr * direction
            c, Jacobian, H = getCJH(self.problem, lhs_x)
            lhs = c

            phi, J = self.problem.evaluate(curr_x)

            c, Jacobian, H = getCJH(self.problem, curr_x)
            gradient = Jacobian.T
            
            rhs = c + self.rho_ls * gradient.reshape(-1,1).T @ (lr * direction)

            if lhs < rhs:
                return lr
            if lr < 1e-16:
                print("Fail to get a proper lr..")
                return
            lr *= self.rho_a_0

    def increase_lr(self, lr):
        increased = self.rho_a_1 * lr
        return np.min([increased, self.sigma_max])


class NewtonWithPullback(NLPSolver):

    def __init__(self):
        """
        See also:
        ----
        NLPSolver.__init__
        """

        # in case you want to initialize some class members or so...
        self.tolerance = 1e-8
        self.damping = 1e-6
        self.verbose = 1
        # the NLPSolver interface provide the setProblem method

    def solve(self):
        """

        See Also:
        ----
        NLPSolver.solve

        """
        # write your code here
        # backtracking factors initialization
        rho_a_0 = 0.5
        rho_a_1 = 1.2
        sigma_max = 1.
        rho_ls = 0.01
        backtracking = BackTracking(self.problem, rho_a_0, rho_a_1, sigma_max, rho_ls)

        # use the following to get an initialization:
        x = self.problem.getInitializationSample()

        # get feature types
        # ot[i] inidicates the type of feature i (either OT.f or OT.sos)
        # there is at most one feature of type OT.f
        # use the following to query the problem:
        # phi is a vector (1D np.array); J is a Jacobian matrix (2D np.array). 

        c, Jacobian, H = getCJH(self.problem, x)
        gradient = Jacobian.T
        try:
            direction = self.get_direction(x, H, gradient)
        except:
            print("Increse the damping ...")
            self.damping *= 10  # if fail to converge, increse the damping
        finally:
            direction = self.get_direction(x, H, gradient)
            
        x0 = x
        lr = 1.
        # now code some loop that iteratively queries the problem and updates x til convergenc....
        for i in range(100):
            
            lr = backtracking.decrease_lr(lr, x0, direction)

            x1 = x0 + direction * lr

            lr = backtracking.increase_lr(lr)

            if np.linalg.norm(lr * direction) < self.tolerance:
                break

            x0 = x1

            c, Jacobian, H = getCJH(self.problem, x0)
            gradient = Jacobian.T
            
            try:
                direction = self.get_direction(x0, H, gradient)
            except:
                print("Increse the damping ...")
                self.damping *= 10  # if fail to converge, increse the damping

            if self.verbose:
                print('Current cost: {0:.4f} at step {1}'.format(c, i))
            

        # finally:
        print("Finally converged at x: {}, final cost: {}".format(x0, c))
        x = x0
        return x

    def get_direction(self, x, H, gradient):
        x = x.reshape(self.problem.getDimension(), 1)
        if H is not None:
            try:
                direction = -np.linalg.inv(H + np.eye(H.shape[0]) * self.damping) @ gradient
            except:
                direction = -gradient / np.linalg.norm(gradient)
            finally:
                if gradient.T @ direction > 0:
                    print("Pull back!")
                    direction = -gradient / np.linalg.norm(gradient)
        else:
            direction = -gradient / np.linalg.norm(gradient)
        #try:
        direction = direction.reshape(self.problem.getDimension(),)
        #except:
            #print(direction.shape)
        return direction