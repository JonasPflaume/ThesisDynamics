import numpy as np

class BackTracking(object):
    def __init__(self, problem, rho_a_0, rho_a_1, sigma_max, rho_ls):
        self.rho_a_0 = rho_a_0
        self.rho_a_1 = rho_a_1
        self.sigma_max = sigma_max
        self.rho_ls = rho_ls
        self.problem = problem

    def decrease_lr(self, lr, curr_x, direction):
        while True:
            phi, J = self.problem.evaluate(curr_x + lr * direction)
            lhs = phi[0]
            phi, J = self.problem.evaluate(curr_x)
            rhs = phi[0] + self.rho_ls * J[0].reshape(-1,1).T @ (lr * direction)
            if lhs < rhs:
                return lr
            if lr < 1e-32:
                print("Fail to get a proper lr..")
                return
            lr *= self.rho_a_0

    def increase_lr(self, lr):
        increased = self.rho_a_1 * lr
        return np.min([increased, self.sigma_max])


if __name__ == "__main__":
    A = BackTracking(1, 1, 1.2, 2, 0.01)
    lr = A.increase_lr(1.2)
    print(lr)
