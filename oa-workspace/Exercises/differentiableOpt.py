## utilize the log barrier method from exercise 5
import sys
sys.path.append('..')
from optimization_algorithms.interface.mathematical_program_traced import MathematicalProgramTraced
from logbarrier import LogBarrierProblem
import numpy as np
from Optimizer.gradientDescent import BackTrackingGD, PlainGD
import time

def get_labd(x, mu, res):
    term1 = - mu / (x.T @ x - 1)
    term2 = - mu / (-x[0] + res)
    return np.array([[term1],[term2]])

def LogBarrierMainLoop(res):
    mu = 1.
    old_res = np.ones([2,1]) * 1000

    center_list = []
    labda_list = []
    start = time.time()
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

        if np.linalg.norm( old_res - curr_res ) < 1e-16:
            break
        
        curr_labd = get_labd(curr_res, mu, res)
        old_res = curr_res
        center_list.append(curr_res)

        labda_list.append(curr_labd)
        mu *= 0.5

    end = time.time()
    print('### Time consuming: ', end - start)

    ### get the kkt matrix ###
    curr_xTheta, curr_labdTheta = Problem.mathematical_program.original.sensitivity(curr_res, curr_labd)

    return curr_res, curr_labd, curr_xTheta, curr_labdTheta



if __name__ == '__main__':
    res = 1/np.sqrt(2) + 0.1
    x_star, labd_star, xTheta, labdTheta = LogBarrierMainLoop(res)
    variant = 1e-6

    x_star_, labd_star_, xTheta_, labdTheta_ = LogBarrierMainLoop(res+variant)
    print(xTheta)
    print((x_star_ - x_star) / variant)


