import casadi
import numpy as np

class MPC:
    ''' sloppy implemented MPC controller
    '''
    def __init__(self, ABCweight):
        self.A = ABCweight[0][1].cpu().detach().numpy()
        self.B = ABCweight[1][1].cpu().detach().numpy()
        self.C = ABCweight[2][1].cpu().detach().numpy()
        self.nx = self.A.shape[1]
        self.nu = self.B.shape[1]
        self.nxx = self.C.shape[0]
        Q = np.eye(self.nxx) * 5
        if self.nxx == 4:
            Q[0,0] = 1
            Q[1,1] = 1e-4
            Q[2,2] = 200
            Q[3,3] = 1e-4
        elif self.nxx == 2:
            Q[0,0] = 20
            Q[1,1] = 0
        elif self.nxx == 3:
            Q[0,0] = 5000
            Q[1,1] = 5000
            Q[2,2] = 1e-4
        R = np.eye(self.B.shape[1])
        self.N = 20
        x = casadi.SX.sym("x", self.nx, 1)
        u = casadi.SX.sym("u",self.nu, 1)
        x_next = self.A @ x + self.B @ u
        self.system = casadi.Function("sys",[x, u],[x_next])

        x_ref = casadi.SX.sym("y_ref",self.nxx,1)
        cost = (self.C@x - x_ref).T @ Q @ (self.C@x - x_ref) + u.T @ R @ u
        self.costFunc = casadi.Function("cost", [x, u, x_ref], [cost])
        t_cost = (self.C@x - x_ref).T @ Q @ (self.C@x - x_ref)
        self.terminal_cost_fcn = casadi.Function("t_cost", [x, x_ref], [t_cost])

        ## test the functional solver production ##
        lb_X,lb_U,ub_X,ub_U,X,U,lb_g,ub_g,g,J,Xref = self.casadi_setting(self.nx, self.nu)
        # Casadi solver
        self.lbx = casadi.vertcat(*lb_X, *lb_U)
        self.ubx = casadi.vertcat(*ub_X, *ub_U)
        x = casadi.vertcat(X, U)
        g = casadi.vertcat(*g)
        self.lbg = casadi.vertcat(*lb_g)
        self.ubg = casadi.vertcat(*ub_g)
        prob = {'f':J,'x':x,'g':g, 'p':Xref}
        opts = {'ipopt.print_level':0, 'print_time':0, 'ipopt.sb':'yes'}
        self.solver = casadi.nlpsol('solver', 'ipopt', prob, opts)
        

    def __call__(self, x, xref, step):
        ''' MPC controller
            step for function position
        '''
        self.lbx[:self.nx] = x
        self.ubx[:self.nx] = x
        res = self.solver(lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg,p=xref)
        res_X = res['x'][:(self.N+1)*self.nx].full().reshape(self.N+1, self.nx)
        res_U = res['x'][(self.N+1)*self.nx:].full().reshape(self.N, self.nu)

        u = res_U[0, :].reshape(self.nu, 1)
        return u

    def casadi_setting(self, nx, nu): # reference_signal:3*15 output*horizon
        # state constraints
        N = self.N
        lb_x = -1e12*np.ones((nx,1))
        ub_x = 1e12*np.ones((nx,1))
        # delta_u changing rate per step constraints
        lb_u = -400*np.ones((nu,1))
        ub_u = 400*np.ones((nu,1))
        
        X = casadi.SX.sym("X",(N+1)*nx,1)
        U = casadi.SX.sym("U",N*nu,1)
        Xref = casadi.SX.sym("r", N, self.nxx)

        J = 0
        lb_X = [] # lower bound for X.
        ub_X = [] # upper bound for X
        lb_U = [] # lower bound for delta_U
        ub_U = [] # upper bound for delta_U
        g = []    # constraint expression g
        lb_g = []  # lower bound for constraint expression g
        ub_g = []  # upper bound for constraint expression g
        
        for k in range(N):
            # Retrieve parameters
            x_k = X[k*nx:(k+1)*nx,:]
            x_k_next = X[(k+1)*nx:(k+2)*nx,:]
            u_k = U[k*nu:(k+1)*nu,:]
            
            # objective
            J += self.costFunc(x_k, u_k, casadi.reshape(Xref[k,:], self.nxx, 1))
            
            # equality constraints (system equation)
            x_k_next_calc = self.system(x_k, u_k)
            
            g.append(x_k_next - x_k_next_calc)
            lb_g.append(np.zeros((nx,1)))
            ub_g.append(np.zeros((nx,1)))

            # set the constraints of input and states
            lb_X.append(lb_x)
            ub_X.append(ub_x)
            lb_U.append(lb_u)
            ub_U.append(ub_u)
        
        # add the terminal cost 
        x_terminal = X[N*nx:(N+1)*nx,:]
        J += self.terminal_cost_fcn(x_terminal, casadi.reshape(Xref[-1,:], self.nxx, 1))
        lb_X.append(lb_x)
        ub_X.append(ub_x)
        
        return lb_X,lb_U,ub_X,ub_U,X,U,lb_g,ub_g,g,J,Xref

class FiniteLQR:
    ''' sloppy implementation of a finite horizon LQR controller
    '''
    def __init__(self, ABCweight, simNum):
        self.A = ABCweight[0][1].cpu().detach().numpy()
        self.B = ABCweight[1][1].cpu().detach().numpy()
        self.C = ABCweight[2][1].cpu().detach().numpy()
        self.nx = self.A.shape[1]
        self.nu = self.B.shape[1]
        self.nxx = self.C.shape[0]
        self.N = simNum

        self.__LQR_startup()

    def __LQR_startup(self):
        N = self.N
        A = self.A
        B = self.B
        C = self.C
        Q = np.eye(self.nx) * 5
        if self.nxx == 4:
            Q[0,0] = 20
            Q[1,1] = 0
            Q[2,2] = 20
            Q[3,3] = 0
        elif self.nxx == 2:
            Q[0,0] = 20
            Q[1,1] = 0
        elif self.nxx == 3:
            Q[0,0] = 500
            Q[1,1] = 500
            Q[2,2] = 1e-3
        R = np.eye(self.B.shape[1])

        # Create a list of N + 1 elements
        P = [None] * (N + 1)
        Qf = Q
        # LQR via Dynamic Programming
        P[N] = Qf
        # For i = N, ..., 1
        for i in range(N, 0, -1):
            # Discrete-time Algebraic Riccati equation to calculate the optimal 
            # state cost matrix
            P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
    
        # Create a list of N elements
        self.K = [None] * N
    
        # For i = 0, ..., N - 1
        for i in range(N):
            # Calculate the optimal feedback gain K
            self.K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
    

    def __call__(self, x, xref, step):
        ''' xref for function argument
        '''
        return self.K[step] @ (x - xref.T)


class KF:
    ''' sloppy implementation of kalman filter
    '''
    def __init__(self, ABCweight):
        covariance = 1
        self.A = ABCweight[0][1].cpu().detach().numpy()
        self.B = ABCweight[1][1].cpu().detach().numpy()
        self.C = ABCweight[2][1].cpu().detach().numpy()
        self.nx = self.A.shape[1]
        self.nu = self.B.shape[1]
        self.nxx = self.C.shape[0]
        self.P = np.eye(self.A.shape[0],self.A.shape[1])
        self.Q = np.ones((self.A.shape[0],self.A.shape[1]))*covariance
        self.R = np.eye(self.C.shape[0],self.C.shape[0])*covariance

    def predict(self,x,u):
        x = x.reshape(-1,1)
        A = self.A
        B = self.B
        P = self.P
        Q = self.Q
        xpred = A@x + (B@u).reshape(A.shape[0],1)
        Ppred = A@P@A.T + Q
        return xpred, Ppred

    def K_G(self):
        # kalman gain calculating
        P = self.P
        C = self.C
        R = self.R
        K = P@C.T@np.linalg.inv(C@P@C.T+R)
        return K

    def update(self,xpred,K,z):
        z = z.reshape(-1,1)
        #update the x and P matrix
        C = self.C
        P = self.P
        R = self.R
        A = self.A
        z = z.reshape(C.shape[0],1)
        xnew = xpred + K@(z-(C@xpred).reshape(C.shape[0],1))
        temp = np.eye(A.shape[0]) - K@C
        Pnew = temp@P@temp.T + K@R@K.T
        self.P = Pnew
        return xnew.T