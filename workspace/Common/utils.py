from itertools import chain

import sympybotics
import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from .regressor import H_panda_regressor

from sympybotics.dynamics.rne import rne_forward, rne_backward
from sympybotics.utils import identity
from copy import copy, deepcopy
from sympybotics.geometry import Geometry

def load_robot_model():

    pi = sympy.pi
    q = sympybotics.robotdef.q
    # load dynamical model by using modified DH parameters
    panda_def = sympybotics.RobotDef ( 'Panda',
    [ ( 0,        0,       0.333, 'q'),
    ( '-pi/2 ', 0,       0,     'q'),
    ( 'pi/2',   0,       0.316, 'q'),
    ( 'pi/2' ,  0.0825 , 0,     'q'),
    ( '-pi/2' , -0.0825, 0.384, 'q'),
    ( 'pi/2' ,  0 ,      0 ,    'q'),
    ( 'pi/2' ,  0.088 ,  0,     'q')],
    dh_convention='modified' )
    panda_def.gravityacc = sympy.Matrix([ 0.0 , 0.0 , -9.81 ])
    rbt = sympybotics.RobotDynCode(panda_def,  verbose=True)

    return rbt, panda_def

def read_data(path):
    if not path.endswith('dat'):
        return
    data = []
    with open(path, 'r') as F:
        d = F.readlines()
        for i in d:
            k = i.rstrip().split(" ")
            k = list(filter(lambda x: x!="", k))
            if len(k) == 99 or len(k) == 57 or len(k)==113 or len(k)==120 or len(k)==127 or len(k)==106:
                data.append(k)
            else:
                pass
                #print("Skipped one line broken data...")
    data = np.array(data, dtype=float)
    return data

# old numerical grad calculator
def numerical_grad_nd(x, delta=1e-3):
    grad = np.zeros_like(x)
    for idx in range(x.shape[0]):
        if idx != 0:
            temp = x[idx,:] - x[idx-1,:]
            grad_t = temp/delta
            grad[idx,:] = grad_t
        else:
            grad[idx,:] = np.zeros_like(x[idx,:])
    return grad

def numerical_grad_nd_fourier(x, delta=1e-3):
    n, channel = x.shape
    L = n * delta
    DF = np.zeros(x.shape)
    for idx in range(channel):
        x_fft = np.fft.fft(x[:,idx])
        kappa = (2*np.pi/L) * np.arange(0,n)
        kappa = np.fft.fftshift(kappa)
        dfhat = np.complex(0,1) * kappa * x_fft
        df = np.real(np.fft.ifft(dfhat))
        DF[:,idx] = df
    return DF

def ButterWorthFilter(qDot_real, tau_real, time):
    # 4th order butterworth filter
    sos = signal.butter(4, 2*np.pi, 'lp', fs=len(time)/(time[-1] - time[1]), output='sos') 

    qDot_real_filtered = np.zeros_like(qDot_real)
    tau_real_filtered = np.zeros_like(tau_real)

    for i in range(7):
        qDot_real_filtered[:,i] = signal.sosfiltfilt(sos, qDot_real[:,i])
        tau_real_filtered[:,i] = signal.sosfiltfilt(sos, tau_real[:,i])

    return qDot_real_filtered, tau_real_filtered

def lowpassfilter(x, ratio):
    temp = np.zeros([1,7])
    res = []
    for i, item in enumerate(x):
        item = item.reshape(1,-1)
        if i == 0:
            temp += item
            res.append(temp)
        else:
            temp = (1 - ratio) * item + ratio * temp
            res.append(temp)
    res = np.concatenate(res, axis=0)
    assert len(res) == len(x)
    return res

def friction(x, speed):
    phi1j, phi2j, phi3j = x
    tau = phi1j/(1+np.exp(-phi2j*(speed+phi3j))) - phi1j/(1+np.exp(-phi2j*phi3j))
    return tau


def friction_dof(X, Speed):
    ''' X is matrix, each row is a x
    '''
    Speed = Speed.reshape(-1,1)
    tau = np.zeros_like(Speed)
    for i in range(X.shape[0]):
        x = X[i, :]
        speed = Speed[i,:]
        tau_i = friction(x, speed)
        tau[i,:] = tau_i
    return tau

def stack_regressor(time, q_real, qDot_real, qDDot_inf, tau_real_filtered, calc_friction_tau=False, opt_x=None):
    # start to stack the regressor
    sample_number = int(len(time))

    Yreg = np.zeros((sample_number*7, 43))
    Stack_tau = np.zeros((sample_number*7, 1))
    if calc_friction_tau:
        friction_torque = np.zeros((sample_number, 7))

    for i in range(sample_number):
        qi = q_real[i]
        qdi = qDot_real[i]
        qddi = qDDot_inf[i]
        Hi = H_panda_regressor(qi,qdi,qddi)
        Hi = np.array(Hi).reshape(7,43)
        Yreg[7*i:7*(i+1), :] = Hi
        Stack_tau[7*i:7*(i+1), :] = tau_real_filtered[i].reshape(7,1)
        if calc_friction_tau:
            for j in range(7): # loop joints
                friction_torque[i, j] = friction(opt_x[j], qdi[j])

    reg_rank = np.linalg.matrix_rank(Yreg)
    reg_cond = np.linalg.cond(Yreg)
    print("The stacked regressor has {} rank and it's condition number is {}.".format(reg_rank, reg_cond))
    if calc_friction_tau:
        return Yreg, Stack_tau, friction_torque

    return Yreg, Stack_tau

def get_index_interval(q, M, G, threshold=5e-2):
    # group the static pose idx
    idx = []
    idx_group = []
    q_0 = q[0]
    counter = 0
    for i in range(q.shape[0]):
        L2_distance = np.linalg.norm(q_0 - q[i])
        if L2_distance < threshold:
            idx.append(i)
            counter += 1
        if not L2_distance < threshold and counter > 8:
            idx_group.append(idx)
            idx = []
            counter = 0
            idx.append(i)
            q_0 = q[i]
            
    # get the middle one and start stacking m, G
    s_stack = np.zeros([len(idx_group)*(28+7),1])
    q_position = []
    for i, k in enumerate(idx_group):
        idx_middle = k[49]
        q_position.append(q[idx_middle])
        mk = M[idx_middle, :]
        gk = G[idx_middle, :].reshape(-1,1)
        assert gk.shape == (7,1)
        
        mk = np.array(mk).reshape(7,7)
        mk = mk[np.tril_indices(7)].reshape(-1,1)
        assert mk.shape == (28,1)
        
        s_stack[i*35:(i+1)*35,:] = np.concatenate((mk, gk))
    # return corresponding configurations and stacked data
    return np.array(q_position), s_stack


def rne(rbtdef, geom, ifunc=None):
    '''Generate joints generic forces/torques equation.'''

    fw_results = rne_forward(rbtdef, geom, ifunc)
    invdyn = rne_backward(rbtdef, geom, fw_results, ifunc=ifunc)

    return invdyn

def ltri_extract(SMatrix):
    '''Extract the lower triagle part of a matrix'''
    
    idx = np.tril_indices(SMatrix.shape[0])
    size = SMatrix.shape[0]
    l_terms = sympy.zeros((size+1)*size/2, 1)
    counter = 0
    for i,j in zip(idx[0],idx[1]):
        l_terms[counter] = SMatrix[i,j]
        counter += 1
    return l_terms

def ie_regressor(rbtdef, geom, ifunc=None):
    '''Generate inverse engineering regression matrix.'''
    
    # ifunc could be used to log
    if not ifunc:
        ifunc = identity
        
    # deep copy the robot definition for G and M generation, rbtdef contain the dynamical parameters
    rbtdeftmp = deepcopy(rbtdef)
    rbtdeftmp_G = deepcopy(rbtdef)

    dynparms = rbtdef.dynparms()
    num_dyn = len(dynparms)
    
    # Initialization of container
    M_row_n = int((rbtdef.dof+1)*rbtdef.dof/2)
    Ys = sympy.zeros(M_row_n + rbtdef.dof, num_dyn) # 35*70
    M = sympy.zeros(M_row_n, num_dyn) # 28*70
    G = sympy.zeros(rbtdef.dof, num_dyn) # 7*70
    
    M_temp = sympy.zeros(rbtdef.dof, rbtdef.dof) # 7*7
    
    # start to build the matrix
    for p, parm in enumerate(dynparms):
        
        ## build G ##
        # set parm equal parameter in rbtdef to 1 and the otherwise to 0
        for i in range(rbtdef.dof):
            rbtdeftmp_G.Le[i] = list(map(
                lambda x: 1 if x == parm else 0, rbtdef.Le[i]))
            rbtdeftmp_G.l[i] = sympy.Matrix(rbtdef.l[i]).applyfunc(
                lambda x: 1 if x == parm else 0)
            rbtdeftmp_G.m[i] = 1 if rbtdef.m[i] == parm else 0
            rbtdeftmp_G.Ia[i] = 1 if rbtdef.Ia[i] == parm else 0
            rbtdeftmp_G.fv[i] = 1 if rbtdef.fv[i] == parm else 0
            rbtdeftmp_G.fc[i] = 1 if rbtdef.fc[i] == parm else 0
            rbtdeftmp_G.fo[i] = 1 if rbtdef.fo[i] == parm else 0
            
        # set dq and ddq to 0, which gravity terms aren't relevant to.
        rbtdeftmp_G.dq = sympy.zeros(rbtdeftmp.dof, 1)
        rbtdeftmp_G.ddq = sympy.zeros(rbtdeftmp.dof, 1)
        rbtdeftmp_G.frictionmodel = None
        geomtmp_G = Geometry(rbtdeftmp_G)
        G_temp = rne(rbtdeftmp_G, geomtmp_G, ifunc)
        G[:,p] = G_temp
            
        ## build M ##
        # set parm equal parameter in rbtdef to 1 and the otherwise to 0
        for i in range(rbtdef.dof):
            rbtdeftmp.Le[i] = list(map(
                lambda x: 1 if x == parm else 0, rbtdef.Le[i]))
            rbtdeftmp.l[i] = sympy.Matrix(rbtdef.l[i]).applyfunc(
                lambda x: 1 if x == parm else 0)
            rbtdeftmp.m[i] = 1 if rbtdef.m[i] == parm else 0
            rbtdeftmp.Ia[i] = 1 if rbtdef.Ia[i] == parm else 0
            rbtdeftmp.fv[i] = 1 if rbtdef.fv[i] == parm else 0
            rbtdeftmp.fc[i] = 1 if rbtdef.fc[i] == parm else 0
            rbtdeftmp.fo[i] = 1 if rbtdef.fo[i] == parm else 0
            
        # M irrelevant to gravity constant
        rbtdeftmp.gravityacc = sympy.zeros(3, 1)
        rbtdeftmp.frictionmodel = None
        # M irrelevant to angular velocity
        rbtdeftmp.dq = sympy.zeros(rbtdeftmp.dof, 1)
        
        # build M column by column by setting corresponding ddq to 1.
        for k in range(M_temp.rows):
            rbtdeftmp.ddq = sympy.zeros(rbtdeftmp.dof, 1)
            rbtdeftmp.ddq[k] = 1
            geomtmp = Geometry(rbtdeftmp)

            fw_results = rne_forward(rbtdeftmp, geomtmp, ifunc)
            Mcoli = rne_backward(rbtdeftmp, geomtmp, fw_results, ifunc=ifunc)
            # It's done like this since M is symmetric:
            M_temp[:, k] = (M_temp[k, :k].T).col_join(Mcoli[k:, :])
        M[:, p] = ltri_extract(M_temp)
    # concatenate the results
    Ys[:M_row_n, :] = M
    Ys[M_row_n:,:] = G
    return Ys, G, M

def plot_result(time, tau):
    plt.figure(figsize=[12,12])
    for channel in range(7):
        plt.subplot(4,2,channel+1,xlabel="time[s]", ylabel="Torque of {}. joint[n/m]".format(channel+1))
        plt.plot(time, tau[:,channel],'-c', label="u_real")
    plt.legend()
    plt.show()
    
def plot_test_results(time, tau_real, tau_approx_RE):
    plt.figure(figsize=[12,12])
    for channel in range(7):
        plt.subplot(4,2,channel+1,xlabel="time[s]", ylabel="Torque of {}. joint[n/m]".format(channel+1))
        plt.plot(time, tau_real[:,channel],'-c', label="u_real")
        plt.plot(time, tau_approx_RE[:,channel],'-r', label="u_ie")
    plt.legend()
    plt.show()

def plot_test_results_with_friction(time, tau_real, tau_inf_nofric, tau_inf_withfric, *args):
    plt.figure(figsize=[12,12])
    for channel in range(7):
        if channel==5 or channel==6:
            plt.subplot(4,2,channel+1,xlabel="time[s]", ylabel="Torque of {}. joint[n/m]".format(channel+1))
        else:
            plt.subplot(4,2,channel+1,xlabel="", ylabel="Torque of {}. joint[n/m]".format(channel+1))
        plt.plot(time, tau_real[:,channel],'-c', linewidth=1, label="u_real")
        plt.plot(time, tau_inf_nofric[:,channel],'-b', linewidth=1, label="u_inf_no_friction")
        plt.plot(time, tau_inf_withfric[:,channel],'-r', linewidth=1, label="u_inf_with_friction")
        
        if args:
            item = args[0]
            plt.plot(time, item[:, channel],'-g', linewidth=1, label="other stuff")
    plt.legend()
    plt.savefig('evaluation.jpg', dpi=200)
    plt.show()
    
def calc_tau(M_func, C_func, G_func, q, dq, ddq, opt_x=None, dyna_func='original'):
    # function to calculate the predicted torque each joint
    # input opt_x to calc torque with friction compensation
    if dyna_func == 'original':
    	param = get_dyna()
    elif dyna_func == 'cma':
        param = get_dyna_CMA()

    from numpy import cos, sin
    import numpy as np
    
    if type(opt_x)==np.ndarray:
        friction_torque = np.zeros((len(q), 7))
    tau = np.zeros(q.shape)
    for idx in range(len(q)):
        q_i = q[idx].reshape(7,1)
        dq_i = dq[idx].reshape(7,1)
        ddq_i = ddq[idx].reshape(7,1)
        M = M_func(param, q_i, cos, sin)
        G = G_func(param, q_i, cos, sin)
        C = C_func(param, q_i, dq_i, cos, sin)
        
        M = np.array(M, dtype=float).reshape(7,7)
        G = np.array(G, dtype=float).reshape(7,1)
        C = np.array(C, dtype=float).reshape(7,7)
        
        tau_i = M@ddq_i + C@dq_i + G
        tau[idx,:] = np.squeeze(tau_i)
        if type(opt_x)==np.ndarray:
            for j in range(7): # loop joints
                    friction_torque[idx, j] = friction(opt_x[j], dq_i[j])
    if type(opt_x)==np.ndarray:            
        return tau, friction_torque
    else:
        return tau

def get_dyna():
    global m1 
    global m2 
    global m3 
    global m4 
    global m5 
    global m6 
    global m7 
    global c1x 
    global c1y 
    global c1z 
    global c2x 
    global c2y 
    global c2z 
    global c3x 
    global c3y 
    global c3z 
    global c4x 
    global c4y 
    global c4z 
    global c5x 
    global c5y 
    global c5z 
    global c6x 
    global c6y 
    global c6z 
    global c7x 
    global c7y 
    global c7z 
    global I1xx
    global I1xy
    global I1xz
    global I1yy
    global I1yz
    global I1zz
    global I2xx
    global I2xy
    global I2xz
    global I2yy
    global I2yz
    global I2zz
    global I3xx
    global I3xy
    global I3xz
    global I3yy
    global I3yz
    global I3zz
    global I4xx
    global I4xy
    global I4xz
    global I4yy
    global I4yz
    global I4zz
    global I5xx
    global I5xy
    global I5xz
    global I5yy
    global I5yz
    global I5zz
    global I6xx
    global I6xy
    global I6xz
    global I6yy
    global I6yz
    global I6zz
    global I7xx
    global I7xy
    global I7xz
    global I7yy
    global I7yz
    global I7zz
        
    m1 = 3.8303
    m2 = 5.7113
    m3 = 3.5256
    m4 = 1.1148
    m5 = 1.7786
    m6 = 1.8548
    m7 = 2.0418
    c1x = -0.0073
    c1y = 0.0274
    c1z = -0.1957
    c2x = -0.0014
    c2y = -0.0349
    c2z = -0.0023
    c3x = 0.0541
    c3y = 0.0063
    c3z = -0.0317
    c4x = -0.0749
    c4y = 0.1298
    c4z = -0.0012
    c5x = -0.0057
    c5y = 0.0231
    c5z = -0.1728
    c6x = 0.0284
    c6y = 0.0051
    c6z = -0.0206
    c7x = -7.1399e-04
    c7y = -8.3331e-05
    c7z = 0.0899
    I1xx = 0.4216
    I1xy = 4.5679e-04
    I1xz = 0.0045
    I1yy = 0.4142
    I1yz = -0.0560
    I1zz = 0.0080
    I2xx = 0.0281
    I2xy = 0.0024
    I2xz = 0.0182
    I2yy = 0.0372
    I2yz = -4.9687e-04
    I2zz = 0.0351
    I3xx = 0.0664
    I3xy = -0.0019
    I3xz = -0.0144
    I3yy = 0.0769
    I3yz = -0.0107
    I3zz = 0.0151
    I4xx = 0.0348
    I4xy = 0.0222
    I4xz = 0.0052
    I4yy = 0.0272
    I4yz = -0.0043
    I4zz = 0.0495
    I5xx = 0.0743
    I5xy = -4.1791e-04
    I5xz = -0.0021
    I5yy = 0.0703
    I5yz = 0.0034
    I5zz = 0.0058
    I6xx = 0.0150
    I6xy = -0.0033
    I6xz = 0.0042
    I6yy = 0.0105
    I6yz = 0.0025
    I6zz = 0.0114
    I7xx = 0.0110
    I7xy = 8.7968e-04
    I7xz = 0.0024
    I7yy = 0.0143
    I7yz = -3.9522e-04
    I7zz = 0.0071
    param = [eval('[I{0}xx, I{0}xy, I{0}xz, I{0}yy, I{0}yz, I{0}zz, c{0}x, c{0}y, c{0}z, m{0}]'.format(i)) 
                 for i in range(1,8)]

    param = list(chain(*param))

    return param

def get_dyna_CMA():
    global m_1 
    global m_2 
    global m_3 
    global m_4 
    global m_5 
    global m_6 
    global m_7 
    global l_1x 
    global l_1y 
    global l_1z 
    global l_2x 
    global l_2y 
    global l_2z 
    global l_3x 
    global l_3y 
    global l_3z 
    global l_4x 
    global l_4y 
    global l_4z 
    global l_5x 
    global l_5y 
    global l_5z 
    global l_6x 
    global l_6y 
    global l_6z 
    global l_7x 
    global l_7y 
    global l_7z 
    global L_1xx
    global L_1xy
    global L_1xz
    global L_1yy
    global L_1yz
    global L_1zz
    global L_2xx
    global L_2xy
    global L_2xz
    global L_2yy
    global L_2yz
    global L_2zz
    global L_3xx
    global L_3xy
    global L_3xz
    global L_3yy
    global L_3yz
    global L_3zz
    global L_4xx
    global L_4xy
    global L_4xz
    global L_4yy
    global L_4yz
    global L_4zz
    global L_5xx
    global L_5xy
    global L_5xz
    global L_5yy
    global L_5yz
    global L_5zz
    global L_6xx
    global L_6xy
    global L_6xz
    global L_6yy
    global L_6yz
    global L_6zz
    global L_7xx
    global L_7xy
    global L_7xz
    global L_7yy
    global L_7yz
    global L_7zz

    x = [ 3.38350078e+00,  4.96603125e+00,  2.51838204e+00,  2.96389742e+00,
        1.78683913e+00,  1.43909552e+00,  1.76025069e+00, -2.40761490e-02,
       -3.22193822e-02, -9.74076573e-02, -5.72375602e-03, -1.18735134e-01,
        1.02541487e-02,  9.52453372e-02,  2.02114911e-02, -7.91660509e-02,
       -1.41949140e-01,  1.50000000e-01, -3.53531585e-03, -1.00414608e-02,
        3.27962456e-02, -5.00000000e-02,  7.93819086e-02, -3.16060750e-02,
       -4.42387008e-02, -1.55499535e-03, -4.65960449e-04,  1.42969533e-01,
        6.59100729e-01,  6.59100719e-01,  1.46835732e-08,  3.72707822e-09,
       -1.95793586e-05,  7.30913856e-05,  1.31291314e-02,  3.81714222e-02,
        3.02080446e-02,  1.37137440e-11,  1.70687448e-02,  2.84313116e-11,
        3.92762590e-02,  4.36115588e-02,  4.33591947e-03,  1.17870447e-05,
       -1.06869766e-02,  2.02480994e-11,  1.69028411e-02,  3.52039149e-02,
        3.10162671e-02,  1.22554593e-02,  3.30204772e-03,  3.46718837e-11,
        1.07669993e-02,  8.05128889e-04,  9.96187045e-03,  5.30207597e-08,
       -1.50795475e-08,  2.83206445e-03,  1.02840741e-02,  7.91397888e-03,
        1.80920172e-02,  2.19062553e-03, -5.11594203e-05,  7.12338557e-04,
        1.11221545e-02,  1.40381121e-02,  3.35729277e-03,  7.57917169e-04,
        2.36640907e-03,  7.28699743e-10]
        
    m_1, m_2, m_3, m_4, m_5, m_6, m_7 = x[0:7]
    l_1x, l_1y, l_1z = x[7:10]
    l_2x, l_2y, l_2z = x[10:13]
    l_3x, l_3y, l_3z = x[13:16]
    l_4x, l_4y, l_4z = x[16:19]
    l_5x, l_5y, l_5z = x[19:22]
    l_6x, l_6y, l_6z = x[22:25]
    l_7x, l_7y, l_7z = x[25:28]
    L_1xx, L_1yy, L_1zz, L_1xy, L_1xz, L_1yz = x[28:34]
    L_2xx, L_2yy, L_2zz, L_2xy, L_2xz, L_2yz = x[34:40]
    L_3xx, L_3yy, L_3zz, L_3xy, L_3xz, L_3yz = x[40:46]
    L_4xx, L_4yy, L_4zz, L_4xy, L_4xz, L_4yz = x[46:52]
    L_5xx, L_5yy, L_5zz, L_5xy, L_5xz, L_5yz = x[52:58]
    L_6xx, L_6yy, L_6zz, L_6xy, L_6xz, L_6yz = x[58:64]
    L_7xx, L_7yy, L_7zz, L_7xy, L_7xz, L_7yz = x[64:70]
    
    param = [eval('[L_{0}xx, L_{0}xy, L_{0}xz, L_{0}yy, L_{0}yz, L_{0}zz, l_{0}x, l_{0}y, l_{0}z, m_{0}]'.format(i)) 
                 for i in range(1,8)]

    param = list(chain(*param))

    return param



