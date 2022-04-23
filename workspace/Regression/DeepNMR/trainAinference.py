import sys
import os
import shutil
import glob
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ABS_PATH)

from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from models import MLP, RNN, LSTM, GRU, CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_generator(Xtrain, Ytrain, batch_size):
    ''' get the Xtrain and Ytrain np.ndarray object to generate training batch tensor
        Xtrain (sampleNum, feature_dim)
        Ytrain (sampleNum, 1)
    '''
    data_length = Xtrain.shape[0]
    batch_num = data_length // batch_size
    # shuffle the dataset
    idx = np.arange(0, data_length)
    np.random.shuffle(idx)
    Xtrain = Xtrain[idx]
    Ytrain = Ytrain[idx]

    for i in range(batch_num):
        X_b = Xtrain[i*batch_size:(i+1)*batch_size, :, :]
        Y_b = Ytrain[i*batch_size:(i+1)*batch_size, :]

        yield torch.from_numpy(X_b).float().to(device), torch.from_numpy(Y_b).float().to(device)


def validate(vali_input, vali_target, Net, Criterion):
    ''' validate the current episode model
    '''
    with torch.no_grad():
        vali_input = torch.from_numpy(vali_input).float().to(device)
        vali_target = torch.from_numpy(vali_target).float().to(device)
        predict = Net(vali_input)
        Loss = Criterion(predict, vali_target)
    return Loss.detach().cpu().numpy()

def train(X, Y, batch_size=256, model_type='MLP', epoch=100, dor=0.5, decay=1e-3, lr=5e-5, training_data=0.85, hidden_dim=10, layers=2):
    ''' main loop
    '''
    logging.basicConfig(filename='training_log', filemode='a', \
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',\
                            datefmt='%H:%M:%S',\
                            level=logging.INFO)

    data_length = X.shape[0]
    Horizon = X.shape[1] // 3
    training_points = int(data_length * training_data)
    Xtrain = X[:training_points, :, :]
    Xvali = X[training_points:, :, :]
    Ytrain = Y[:training_points, :]
    Yvali = Y[training_points:, :]

    if model_type == 'MLP':
        Net = MLP(Xtrain.shape[1]*Xtrain.shape[2], Ytrain.shape[1], dor).to(device) # X feature, Y feature, drop out rate
    elif model_type == 'RNN':
        Net = RNN(Xtrain.shape[2], Ytrain.shape[1], hidden_dim, layers, dor).to(device)
    elif model_type == 'LSTM':
        Net = LSTM(Xtrain.shape[2], Ytrain.shape[1], hidden_dim, layers, dor).to(device)
    elif model_type == 'GRU':
        Net = GRU(Xtrain.shape[2], Ytrain.shape[1], hidden_dim, layers, dor).to(device)
    elif model_type == 'CNN':
        Net = CNN(Xtrain.shape[1], Ytrain.shape[1], hidden_dim).to(device)
    else:
        raise ValueError('Model Not implemented!')
    print(Net)
    optimizer = Adam(Net.parameters(), lr=lr, weight_decay=decay)
    Criterion = MSELoss()
    optimizer.zero_grad()

    vali_loss_l = []
    train_loss_l = []
    pbar = tqdm(range(epoch))
    for ep in pbar:
        
        Loss_total = 0
        Step_counter = 0
        vali_loss = validate(Xvali, Yvali, Net, Criterion)
        for x_b, y_b in batch_generator(Xtrain, Ytrain, batch_size):
            optimizer.zero_grad()

            output = Net(x_b)
            Loss = Criterion(output, y_b)
            Loss.backward()
            Loss_total += Loss.item()
            Step_counter += 1
            optimizer.step()
        model_name = '/model/{}_HR_{}_EP_{}.pth'.format(model_type ,Horizon, ep)
        ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        torch.save(Net, ABS_PATH+model_name)
        vali_loss_l.append(vali_loss)
        train_loss_l.append(Loss_total / Step_counter)

        pbar.set_description('train loss: {:.4f}, vali loss: {:.4f}'.format(Loss_total / Step_counter, vali_loss))
    
    suffix = np.argmin(vali_loss_l)
    logging.info('{} model, {} horizon got {:.3f} validataion error at {}. episode.'.format(model_type, Horizon, np.min(vali_loss_l), suffix))
    shutil.move(ABS_PATH + '/model/{}_HR_{}_EP_{}.pth'.format(model_type ,Horizon, suffix), ABS_PATH + '/picked_model/{}_HR_{}_EP_{}.pth'.format(model_type ,Horizon, suffix))
    # clear model space
    files = glob.glob(ABS_PATH + '/model/*')
    for f in files:
        os.remove(f)


def inference(model, X):
    ''' sloppy inference implementation
    '''
    model.eval()
    model.to('cpu')
    X = torch.from_numpy(X).float().to('cpu')
    pred = model(X)
    return pred.detach().cpu().numpy()


def load_model(addr):
    ''' load the full model, instead of parameters dict
    '''
    return torch.load(addr)
