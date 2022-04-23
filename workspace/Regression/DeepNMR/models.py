import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    '''
        Base MLP networks
    '''
    def __init__(self, input_dim, output_dim, dor):
        super(MLP, self).__init__()
        layer_factor = np.sqrt(input_dim/50) # gurantee the output is 50

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, int(input_dim/layer_factor)),
            nn.Dropout(p=dor),
            nn.LeakyReLU(),
            nn.Linear(int(input_dim/layer_factor), int(input_dim/(layer_factor**2))),
            nn.Dropout(p=dor),
            nn.LeakyReLU(),
            nn.Linear(int(input_dim/(layer_factor**2)), output_dim)
        )
        # initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.net(x)


class RNN(nn.Module):
    '''
        Simple RNN model
    '''
    def __init__(self, input_dim, output_dim, hidden_dim, layers, dor):
        super(RNN, self).__init__()

        if layers > 1:
            self.dor = dor
        else:
            self.dor = 0.
        
        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=layers,
                          batch_first=True,
                          dropout=self.dor)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :] # pick the last time step of rnn output
        out = self.fc(out)

        return out

class LSTM(nn.Module):
    ''' LSTM model
    '''
    def __init__(self, input_dim, output_dim, hidden_dim, layers, dor, bidirectional=False):
        super(LSTM, self).__init__()

        self.bidirectional = bidirectional
        if layers > 1:
            self.dor = dor
        else:
            self.dor = 0.

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=layers,
                            batch_first=True,
                            dropout=self.dor,
                            bidirectional=bidirectional)

        if self.bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class GRU(nn.Module):
    '''Gat e Recurrent Unit'''
    def __init__(self, input_dim, output_dim, hidden_dim, layers, dor):
        super(GRU, self).__init__()

        if layers > 1:
            self.dor = dor
        else:
            self.dor = 0.

        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=layers,
                          dropout=self.dor,
                          batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class CNN(nn.Module):
    ''' 1 Dim CNN '''
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=30, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(in_channels=30, out_channels=10, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten(),

            nn.Linear(10 * (output_dim - 2), output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

    def forward(self, x):
        out = self.cnn(x)
        return out
