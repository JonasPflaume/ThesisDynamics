import sys
sys.path.append('..')
from Common.Dataset_generator import DatasetsNMResidualBlockToOne, DatasetsNMResidual

import numpy as np
import glob

from regression import CV_RR
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from DeepNMR.trainAinference import train
from Common.utils import read_data, numerical_grad_nd, ButterWorthFilter
from Common.utils import friction_dof
from DeepNMR.trainAinference import inference, load_model

def build_dataset_RR(D, horizon, feature=1):
    ''' build a simple block to one dataset
    '''
    datanum = len(D)
    dof = D[0][0].shape[1]
    horizon = D.horizon
    Ytrain = np.zeros([datanum, dof])
    Xtrain = np.zeros([datanum, feature_num*horizon, dof])
    for i, (state, action) in enumerate(D):
        Ytrain[i, :] = action
        Xtrain[i, :, :] = state
        
    return Ytrain, Xtrain

def feature(X):
    aug = X
    #aug = np.concatenate([aug, X ** 2], axis=1)
    #aug = np.concatenate([aug, np.cos(0.5*X)], axis=1)
    #aug = np.concatenate([aug, np.sin(X)], axis=1)
    #aug = np.concatenate([aug, np.sin(0.5*X)], axis=1)
    
    poly = PolynomialFeatures(1)
    X = poly.fit_transform(aug)
    return X


horizon_L = [1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,70,80,90,100]
dof = 7
epoch = 250
VALIDATE = 1
TRAIN = 1

error_RR = []
error_MLP = []
error_RNN = []
error_LSTM = []
error_GRU = []
error_CNN = []

runtime_RR = []
runtime_MLP = []
runtime_RNN = []
runtime_LSTM = []
runtime_GRU = []
runtime_CNN = []

if TRAIN:
    for horizon in horizon_L:
        print('Current horizon: ', horizon)

        D = DatasetsNMResidualBlockToOne('../data/trajectories', horizon, False, False, subsample=1, filter=False)
        # path:str, horizon:int, overlapping:bool, shuffle_dataset:bool, subsample: sampling gap, filter: smoothness

        feature_num = D[1][0].shape[0] // horizon

        Ytrain, Xtrain_noExpand = build_dataset_RR(D, horizon)
        # build dataset from given datastructure Datasets.

        length = Xtrain_noExpand.shape[0]
        Xtrain = Xtrain_noExpand.reshape(length, -1)
        Xtrain = feature(Xtrain)

        labd = np.logspace(-5, 6, 10)
        exec('RR_{} = CV_RR(labd)'.format(horizon))
        exec('RR_{}.fit(Xtrain, Ytrain, 4, verbose=False)'.format(horizon))

        # train(Xtrain_noExpand, Ytrain, model_type='MLP', epoch=epoch ,batch_size=128, lr=1e-4)
        # train(Xtrain_noExpand, Ytrain, model_type='RNN', epoch=epoch ,batch_size=128, lr=1e-4)
        # train(Xtrain_noExpand, Ytrain, model_type='LSTM', epoch=epoch ,batch_size=128, lr=1e-4)
        # train(Xtrain_noExpand, Ytrain, model_type='GRU', epoch=epoch ,batch_size=128, lr=1e-4)
        # train(Xtrain_noExpand, Ytrain, model_type='CNN', epoch=epoch ,batch_size=128, lr=1e-4)

if VALIDATE:
    for horizon in horizon_L:
        path = '../data/test_trajectory/traj7_z.panda.dat'
        dataset = read_data(path)

        time = dataset[:,0]

        indx = max(np.where(time<40)[0])
        time = time[:indx]
        q_real = dataset[:indx,1:1+7]
        q_ref = dataset[:indx,8:8+7]
        qDot_real = dataset[:indx,15:15+7]
        u_cmd = dataset[:indx, 22:22+7]
        tau_real = dataset[:indx,29:29+7]
        Gra = dataset[:indx,36:36+7]
        u_G = u_cmd + Gra

        # get numerical derived qDDot
        qDot_real_filtered, _ = ButterWorthFilter(qDot_real, tau_real, time)
        qDDot_inf = numerical_grad_nd(qDot_real_filtered)

        qDotRef_inf = numerical_grad_nd(q_ref)
        qDDotRef_inf = numerical_grad_nd(qDotRef_inf)

        residual = u_G - tau_real
        
        parameter = np.load('../Common/fric_param.npy')
        length = len(q_real)
        target_test = u_G - tau_real

        ### sigmoidal friction model ###
        # pred_torque_fc = []
        # for i in range(length):
        #     speed = qDot_real_filtered[i]
        #     tau_fc = friction_dof(parameter, speed)
        #     pred_torque_fc.append(tau_fc)
            
        # pred_torque_fc = np.array(pred_torque_fc)
        # pred_torque_fc = pred_torque_fc.reshape(-1, dof)

        # MAE_fc = np.abs(pred_torque_fc - target_test).sum() / len(pred_torque_fc)

        ### block to one ridge regression ###
        pred_torque = []
        Xtest = []
        for i in range(length):
            if i < horizon: # if horizon = 1, history contains the current state, residual will be zero
                q_history = np.concatenate([np.zeros([horizon-i-1, dof]), q_real[:i+1, :]])
                qd_history = np.concatenate([np.zeros([horizon-i-1, dof]), qDot_real_filtered[:i+1, :]])
                residual_history = np.concatenate([np.zeros([horizon-i, dof]), residual[:i, :]])
            else:
                q_history = q_real[i-horizon+1:i+1, :]
                qd_history = qDot_real_filtered[i-horizon+1:i+1, :]
                residual_history = residual[i-horizon:i, :]
            if length - i - 1 < horizon:
                q_refH = np.concatenate([q_ref[i+1:i+1+horizon, :], np.zeros([horizon - length + i + 1, dof])])
                qd_ref = np.concatenate([qDotRef_inf[i+1:i+1+horizon, :], np.zeros([horizon - length + i + 1, dof])])
                qdd_ref = np.concatenate([qDDotRef_inf[i+1:i+1+horizon, :], np.zeros([horizon - length + i + 1, dof])])
            else:
                q_refH = q_ref[i+1:i+1+horizon, :]
                qd_ref = qDotRef_inf[i+1:i+1+horizon, :]
                qdd_ref = qDDotRef_inf[i+1:i+1+horizon, :]

            X = [q_history, qd_history, residual_history]#, q_refH, qd_ref, qdd_ref]
            X = np.concatenate(X)
            Xtest.append(X)
        # feature transformation
        Xtest_noExpand = np.array(Xtest)
        Xtest = Xtest_noExpand.reshape(-1, Xtest_noExpand.shape[1]*Xtest_noExpand.shape[2])
        Xtest = feature(Xtest)

        exec('pred_torque_RR = RR_{0}.predict(Xtest, RR_{0}.beta)'.format(horizon))

        # manuel defined threshold
        #idx = np.abs(pred_torque_RR) > 3.
        #pred_torque_RR[idx] = 0

        pred_torque_RR = pred_torque_RR.reshape(-1, dof)

        ER_RR = np.abs(pred_torque_RR - target_test).sum() / len(pred_torque_RR)

        # file_list = glob.glob('/home/jiayun/git/workspace/Regression/DeepNMR/picked_model/*')
        # path_r = '/home/jiayun/git/workspace/Regression/DeepNMR/picked_model/'
        # for file_i in file_list:
        #     if file_i.startswith(path_r + 'MLP_HR_{}_'.format(horizon)):
        #         MLP_file = file_i
        #     elif file_i.startswith(path_r + 'RNN_HR_{}_'.format(horizon)):
        #         RNN_file = file_i
        #     elif file_i.startswith(path_r + 'LSTM_HR_{}_'.format(horizon)):
        #         LSTM_file = file_i
        #     elif file_i.startswith(path_r + 'GRU_HR_{}_'.format(horizon)):
        #         GRU_file = file_i
        #     elif file_i.startswith(path_r + 'CNN_HR_{}_'.format(horizon)):
        #         CNN_file = file_i
        # # MLP validate
        
        # MLP_model = load_model(MLP_file)
        # pred_torque_MLP = inference(MLP_model, Xtest_noExpand)

        # ER_MLP = np.abs(pred_torque_MLP - target_test).sum() / len(pred_torque_MLP)

        # # RNN validate
        
        # RNN_model = load_model(RNN_file)
        # pred_torque_RNN = inference(RNN_model, Xtest_noExpand)

        # ER_RNN = np.abs(pred_torque_RNN - target_test).sum() / len(pred_torque_RNN)

        # # LSTM validate
        # LSTM_model = load_model(LSTM_file)
        # pred_torque_LSTM = inference(LSTM_model, Xtest_noExpand)

        # ER_LSTM = np.abs(pred_torque_LSTM - target_test).sum() / len(pred_torque_LSTM)

        # # GRU validate
        # GRU_model = load_model(GRU_file)
        # pred_torque_GRU = inference(GRU_model, Xtest_noExpand)

        # ER_GRU = np.abs(pred_torque_GRU - target_test).sum() / len(pred_torque_GRU)

        # CNN validate
        # CNN_model = load_model(CNN_file)
        # pred_torque_CNN = inference(CNN_model, Xtest_noExpand)

        # ER_CNN = np.abs(pred_torque_CNN - target_test).sum() / len(pred_torque_CNN)


        ##### runing time test #####
        import time
        #start = time.time()
        #for _ in range(1000):
        #    RR.predict(Xtest[1,:], RR.beta)
        #end = time.time()
        #RR_time = end - start

        # start = time.time()
        # for _ in range(1000):
        #     inference(MLP_model, Xtest_noExpand[1:2])
        # end = time.time()
        # MLP_time = end - start

        # start = time.time()
        # for _ in range(1000):
        #     inference(RNN_model, Xtest_noExpand[1:2])
        # end = time.time()
        # RNN_time = end - start

        # start = time.time()
        # for _ in range(1000):
        #     inference(LSTM_model, Xtest_noExpand[1:2])
        # end = time.time()
        # LSTM_time = end - start

        # start = time.time()
        # for _ in range(1000):
        #     inference(GRU_model, Xtest_noExpand[1:2])
        # end = time.time()
        # GRU_time = end - start

        # start = time.time()
        # for _ in range(1000):
        #     inference(CNN_model, Xtest_noExpand[1:2])
        # end = time.time()
        # CNN_time = end - start


        error_RR.append(ER_RR)
        # error_MLP.append(ER_MLP)
        # error_RNN.append(ER_RNN)
        # error_LSTM.append(ER_LSTM)
        # error_GRU.append(ER_GRU)
        # error_CNN.append(ER_CNN)

        #runtime_RR.append(RR_time)
        # runtime_MLP.append(MLP_time)
        # runtime_RNN.append(RNN_time)
        # runtime_LSTM.append(LSTM_time)
        # runtime_GRU.append(GRU_time)
        # runtime_CNN.append(CNN_time)


record_f = open("record.txt", "w")

record_f.write(str(horizon_L) + '\n')

record_f.write(str(error_RR) + '\n')
# record_f.write(str(error_MLP) + '\n')
# record_f.write(str(error_RNN) + '\n')
# record_f.write(str(error_LSTM) + '\n')
# record_f.write(str(error_GRU) + '\n')
# record_f.write(str(error_CNN) + '\n')

#record_f.write(str(runtime_RR) + '\n')
# record_f.write(str(runtime_MLP) + '\n')
# record_f.write(str(runtime_RNN) + '\n')
# record_f.write(str(runtime_LSTM) + '\n')
# record_f.write(str(runtime_GRU) + '\n')
# record_f.write(str(runtime_CNN) + '\n')

record_f.close()
