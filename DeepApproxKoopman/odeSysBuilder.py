'''
Ode numerical solver using casadi
If other dynamical system need to be add here,
then new ode_solver need to be written.
'''
import sys
import numpy as np
import casadi
from PandaDynamics.M_matrix import M_panda
from PandaDynamics.C_matrix import C_panda
from PandaDynamics.G_vector import G_panda
from PandaDynamics.dynamical_param import get_dyna # get dynamical parameters

PandaDof = 7 # panda degree of freedom

#
#                / 
#               /
#              /    pole: M = 1 kg
#             /     pole: R = 1 m
#            /
#     ______/_____  Cart: M = 1 kg
#    |            |
#    |____________|
#      O        O
#
# x = [p, pDot, theta, thetaDot]

def pendulum_ode_solver(nonlinear:bool=True) -> casadi.integrator:
    '''
    cart and mounted pendulum system
    nonlinear: if True, the ode system will be nonlinear version
    '''
    Mp = 1
    Mc = 1
    l = 1
    g = 9.81

    c1 = 1/(Mp+Mc)
    c2 = l*Mp/(Mp+Mc)

    nx = 4  # state space \in R^4 []
    na = 1
    dt = 0.05
    u = casadi.SX.sym('u', na)
    x = casadi.SX.sym("x", 4)
    if nonlinear:
        Sth = casadi.sin(x[2])
        Cth = casadi.cos(x[2])
        dx0_nonlinear = x[1]
        dx2_nonlinear = x[3]
        dx3_nonlinear = g*Sth+Cth*(-c1*u-c2*x[3]**2*Sth)
        dx3_nonlinear = dx3_nonlinear/((4/3)*l - c2*Cth**2)
        dx1_nonlinear = c1*u + c2*(x[3]**2*Sth - dx3_nonlinear*Cth)
        x_dot = casadi.vertcat(dx0_nonlinear, dx1_nonlinear, dx2_nonlinear, dx3_nonlinear)
    else:
        A = np.array([[0, 1, 0, 0], [0, 0, -c2*g/(l*4/3-c2), 0], [0, 0, 0, 1], [0, 0, g/(l*4/3-c2), 0]])
        B = np.array([[0],[c1+c1*c2/(l*4/3-c2)], [0], [-c1/(l*4/3-c2)]])
        x_dot = A@x + B@u
    # dynamical system defined as a callable function
    system = casadi.Function("sys", [x,u], [x_dot])
    ode = {'x': x, 'ode': x_dot, 'p': u}
    opts = {'tf': dt}
    ode_solver = casadi.integrator('F', 'idas', ode, opts)

    return ode_solver


def SpringSlider_ode_solver() -> casadi.integrator:
    '''
    Spring slider ode builder, for a nolinear spring system
    '''
    alpha = 0.5  # nonlinear spring constant
    m = 2        # slider mass


    nx = 2  # state space \in R^4 []
    na = 1
    dt = 0.05
    u = casadi.SX.sym('u', na)
    x = casadi.SX.sym("x", nx)

    dx0 = x[1]
    dx1 = - alpha/m * x[0] ** 3 + 1/m * u
    x_dot = casadi.vertcat(dx0, dx1)
    
    # dynamical system defined as a callable function
    system = casadi.Function("sys", [x, u], [x_dot])
    ode = {'x': x, 'ode': x_dot, 'p': u}
    opts = {'tf': dt}
    ode_solver = casadi.integrator('F', 'idas', ode, opts)

    return ode_solver

def panda_ode_solver() -> casadi.integrator:
    ''' panda robot ode builder
    '''
    param = get_dyna()
    q_dq = casadi.SX.sym("q_dq", 2*PandaDof)
    M = M_panda(param, q_dq[:PandaDof], casadi.cos, casadi.sin)
    C = C_panda(param, q_dq[:PandaDof], q_dq[PandaDof:], casadi.cos, casadi.sin)
    G = G_panda(param, q_dq[:PandaDof], casadi.cos, casadi.sin)

    M = casadi.vertcat(*M)
    G = casadi.vertcat(*G)
    C = casadi.vertcat(*C)

    M = casadi.reshape(M, PandaDof, PandaDof)
    G = casadi.reshape(G, PandaDof, 1)
    C = casadi.reshape(C, PandaDof, PandaDof)

    G = casadi.simplify(G)
    C = casadi.simplify(C)
    M = casadi.simplify(M)

    M_func = casadi.Function('M', [q_dq[:PandaDof]], [M])
    C_func = casadi.Function('M', [q_dq], [C])
    G_func = casadi.Function('M', [q_dq[:PandaDof]], [G])

    nx = 2 * PandaDof  # state space \in R^14, first 7 dim are q, last 7 dim are qDot
    na = PandaDof
    u = casadi.SX.sym('u', na)
    dt = 0.01
    q = q_dq[:PandaDof]
    dq = q_dq[PandaDof:]
    q_dq_dot_1 = dq
    q_dq_dot_2 = casadi.inv(M) @ (u - C @ dq - G)

    q_dq_dot = casadi.vertcat(q_dq_dot_1, q_dq_dot_2)

    # dynamical system defined as a callable function
    system = casadi.Function("sys", [q_dq, u], [q_dq_dot])

    ode = {'x': q_dq, 'ode': q_dq_dot, 'p': u}
    opts = {'tf': dt}
    ode_solver = casadi.integrator('F', 'idas', ode, opts)
    return ode_solver
