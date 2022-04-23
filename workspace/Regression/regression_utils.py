### For horizon prediction ###
import sys
sys.path.append('..')
from Common.utils import *
import numpy as np

def output_poly_param(torque, order, horizon):
    ## take torque and output the poly parameter ##
    horizon = torque.shape[0]
    x = np.arange(0, horizon) * 0.01
    X = np.zeros([horizon, order])
    for i in range(order):
        X[:, i] = x**i
        
    out_param = np.linalg.inv(X.T@X)@X.T@torque
    return out_param.reshape(-1,)

def subsample(state, sample_num):
    start_q_qdot = state[0:2,:]
    interval = (state.shape[0]-2) // sample_num
    sample_idx = [2 + i*interval for i in range(sample_num)]
    sample_state = state[sample_idx]
    sample_state = np.concatenate([start_q_qdot, sample_state], axis=0)
    return sample_state.reshape(-1,)

def paramToTorqueTraj(param, horizon):
    # param must be the shape of (order, dof)
    order = param.shape[0]
    x = np.arange(0, horizon) * 0.01
    X = np.zeros([horizon, order])
    for i in range(order):
        X[:, i] = x**i
    
    return X@param

def build_dataset(D, poly_dim, sample_num, feature_dim, horizon, feature=1, n_knot=4):
    # feature choice, 1 for poly, 2 for spline
    dof = D.q_dim
    train_len = len(D)
    horizon = D.horizon
    Ytrain_f = np.zeros([train_len, horizon, dof])
    Xtrain_f = np.zeros([train_len, 603, dof])
    
    subsample_dim = 2*dof + sample_num*dof          # should be 2*7 + n*7, n is subsample points
    Ytrain = np.zeros([train_len, poly_dim*dof])
    Xtrain = np.zeros([train_len, subsample_dim])
    
    for idx, data in enumerate(D):
        if idx == train_len:
            break
            
        state, torque = data
        Xtrain[idx, :] = subsample(state, sample_num)
        Ytrain[idx, :] = output_poly_param(torque, poly_dim, horizon)
        Xtrain_f[idx, :, :] = state
        Ytrain_f[idx, :, :] = torque
    if feature == 1:
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(feature_dim)
        Xtrain = poly.fit_transform(Xtrain)
    elif feature == 2:
        from sklearn.preprocessing import SplineTransformer
        spline = SplineTransformer(degree=poly_dim, n_knots=n_knot)
        Xtrain = spline.fit_transform(Xtrain)
    else:
        raise ValueError()
    return Ytrain, Xtrain, Ytrain_f, Xtrain_f

def read_residual_data(root, horizon, WithErrorTraj=False, return_UG_error=False, ratio=0.9):
    '''
    root is a list of trajectory address
    '''
    error_list = []
    errorH_list = []
    errorH_feature_list = []
    x_list = []
    xref_list = []
    xH_list = []
    xHref_list = []
    UG_error_list = []
    for path in root:
        dataset = read_data(path)

        time = dataset[:,0]

        indx = max(np.where(time<38)[0])
        time = time[:indx]
        q_real = dataset[:indx,1:1+7]
        q_ref = dataset[:indx,8:8+7]
        qDot_real = dataset[:indx,15:15+7]
        u_cmd = dataset[:indx, 22:22+7]
        tau_real = dataset[:indx,29:29+7]
        Gra = dataset[:indx,36:36+7]
        u_G = u_cmd + Gra
        G_dat = dataset[:indx,36:36+7]
        C_dat = dataset[:indx, 43:43+7]
        M_dat = dataset[:indx, 50:50+49]
        
        qDot_real_filtered, tau_real_filtered = ButterWorthFilter(qDot_real, tau_real, time)

        qDDot_inf = numerical_grad_nd(qDot_real_filtered) # numerical derivative to get joints acceleration.

        qD_ref = numerical_grad_nd(q_ref)
        qD_ref_filtered, _ = ButterWorthFilter(qD_ref, tau_real, time)
        qDD_ref = numerical_grad_nd(qD_ref_filtered)

        ## have to calculate MCG torques by hand
        tau_MCG = []
        for i in range(len(tau_real)):
            tau_i = M_dat[i,:].reshape(7,7) @ qDDot_inf[i, :].reshape(7,1)
            tau_i += C_dat[i,:].reshape(7,1)
            tau_i += G_dat[i,:].reshape(7,1)
            tau_MCG.append(tau_i)

        tau_MCG = np.array(tau_MCG).reshape(len(tau_real), 7)

        error = tau_real_filtered - tau_MCG
        UG_error = u_G - tau_MCG
        error_list.append(error)
        UG_error_list.append(UG_error)
        qd = qDot_real_filtered
        x = np.concatenate([q_real, qd, qDDot_inf], axis=1)
        xref = np.concatenate([q_ref, qD_ref, qDD_ref], axis=1)
        xref_list.append(xref)
        x_list.append(x)

        # create the horizon datasets
        for i in range(horizon, len(q_real)):
            q_real_h_i = q_real[i-horizon+1:i+1, :]
            qD_real_h_i = qDot_real[i-horizon+1:i+1, :]
            qDDot_inf_h_i = qDDot_inf[i-horizon+1:i+1, :]
            if WithErrorTraj:
                error_h_i = error[i-horizon+1:i+1, :]
                errorH_feature_list.append(error_h_i.reshape(1,-1))
            xHref_i = np.concatenate([q_ref[i-horizon+1:i+1, :], qD_ref[i-horizon+1:i+1, :],\
                                      qDD_ref[i-horizon+1:i+1, :]], axis=1)
            xHref_list.append(xHref_i.reshape(1,-1))
            xH_i = np.zeros_like(xHref_i)
            temp = np.concatenate([q_real_h_i, qD_real_h_i, qDDot_inf_h_i], axis=1) # length = 1
            xH_i[:, :] = temp
            xH_list.append(xH_i.reshape(1,-1))
        errorH_list.append(error[horizon:, :])

    x = np.concatenate(x_list)
    xref = np.concatenate(xref_list)
    xH = np.concatenate(xH_list, axis=0)
    xHref = np.concatenate(xHref_list, axis=0)
    error = np.concatenate(error_list)
    UG_error = np.concatenate(UG_error_list)
    errorH = np.concatenate(errorH_list)
    if WithErrorTraj:
        errorH_feature = np.concatenate(errorH_feature_list)
        return x, xref, xH, xHref, error, errorH, errorH_feature
    else:
        if return_UG_error:
            return x, xref, xH, xHref, error, errorH, UG_error
        else:
            return x, xref, xH, xHref, error, errorH

def read_traj(path):
    dataset = read_data(path)

    time = dataset[:,0]

    indx = max(np.where(time<38)[0])
    time = time[:indx]
    q_real = dataset[:indx,1:1+7]
    q_ref = dataset[:indx,8:8+7]
    qDot_real = dataset[:indx,15:15+7]
    u_cmd = dataset[:indx, 22:22+7]
    tau_real = dataset[:indx,29:29+7]
    Gra = dataset[:indx,36:36+7]
    u_G = u_cmd + Gra
    G_dat = dataset[:indx,36:36+7]
    C_dat = dataset[:indx, 43:43+7]
    M_dat = dataset[:indx, 50:50+49]

    qDot_real_filtered, tau_real_filtered = ButterWorthFilter(qDot_real, tau_real, time)
    qDot_real_filtered, u_G_filtered = ButterWorthFilter(qDot_real, u_G, time)

    qDDot_inf = numerical_grad_nd(qDot_real_filtered) # numerical derivative to get joints acceleration.

    qD_ref = numerical_grad_nd(q_ref)
    qD_ref_filtered, _ = ButterWorthFilter(qD_ref, tau_real, time)
    qDD_ref = numerical_grad_nd(qD_ref_filtered)

    ## have to calculate MCG torques by hand
    tau_MCG = []
    for i in range(len(tau_real)):
        tau_i = M_dat[i,:].reshape(7,7) @ qDDot_inf[i, :].reshape(7,1)
        tau_i += C_dat[i,:].reshape(7,1)
        tau_i += G_dat[i,:].reshape(7,1)
        tau_MCG.append(tau_i)

    tau_MCG = np.array(tau_MCG).reshape(len(tau_real), 7)
    return time, q_real, qDot_real, qDot_real_filtered, qDDot_inf, tau_real, tau_real_filtered, tau_MCG

def build_dataset_RR(D, horizon):
    ''' build a simple block to one dataset
    '''
    datanum = len(D)
    dof = 7
    horizon = D.horizon
    Ytrain = np.zeros([datanum, dof])
    Xtrain = np.zeros([datanum, 21*horizon])
    for i, (state, action) in enumerate(D):
        Ytrain[i, :] = action
        Xtrain[i, :] = state.reshape(-1)
    return Ytrain, Xtrain