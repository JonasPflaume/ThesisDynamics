import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import namedtuple

point = namedtuple('point', ['x', 'y'])

class Problem:
    def __init__(self, n):
        self.x0 = np.ones([n,1]) * 2
        self.C = np.diag([10**((i-1)/(n-1)) for i in range(1, n+1)])
        self.dim = n

    def evaluate(self, x):
        return x.T @ self.C @ x

class One_L_ES:
    def __init__(self, Problem, labd, sigma, tolerance=1e-3):
        self.Problem = Problem
        self.sigma = sigma
        self.labd = labd
        self.buffer = []
        self.x_history = []
        self.y_history = []
        self.tolerance = tolerance

    def solve(self):
        x0 = self.Problem.x0
        y0 = self.Problem.evaluate(x0)
        for i in range(2000):
            self.buffer.append(point(x=x0, y=y0))
            self.x_history.append(x0)
            self.y_history.append(y0)
            mean = x0
            for j in range(self.labd):
                x_samp = np.random.normal(mean, self.sigma)
                x = point(x=x_samp, y=self.Problem.evaluate(x_samp))
                self.buffer.append(x)

            self.buffer = sorted(self.buffer, key=lambda x: x.y, reverse=False)
            if np.linalg.norm(x0 - self.buffer[0].x) < self.tolerance and np.linalg.norm(x0 - self.buffer[0].x) != 0:
                break
            x0 = self.buffer[0].x
            y0 = self.buffer[0].y
            self.buffer.clear()

        return x0

    def plot_his(self):
        plt.figure(figsize=[5,5])
        his = np.concatenate(self.x_history, axis=1)
        plt.plot(his[0,:], his[1,:])
        plt.grid()
        plt.show()

    def plot_value(self):
        plt.figure(figsize=[5,5])
        y_his = np.array(self.y_history).squeeze()
        plt.plot(y_his)
        plt.grid()
        plt.show()


if __name__ == '__main__':

    P = Problem(100)

    Solver = One_L_ES(P, 100, 0.02)
    x = Solver.solve()
    Solver.plot_his()
    Solver.plot_value()
