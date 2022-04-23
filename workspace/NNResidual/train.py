import sys
sys.path.append('..')
from tqdm import tqdm
import logging

from Common.Dataset_generator import NNDatasets
from Common.utils import plot_test_results_with_friction
from network import NetBase
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batch_generator(Dataset, batch_size):
    input_b = np.zeros([batch_size, Dataset.q_dim * 4]) # include u from MCG
    target_b = np.zeros([batch_size, Dataset.q_dim])
    count = 0
    for state, action in Dataset:
        input_b[count, :] = np.concatenate([state, action[0, :][np.newaxis, :]], axis=0).reshape(-1,)
        target_b[count, :] = action[0, :] - action[1, :] # u_pd - u_measure
        count += 1
        if count == batch_size:
            yield torch.from_numpy(input_b).float(), torch.from_numpy(target_b).float()
            count = 0
            input_b = np.zeros([batch_size, Dataset.q_dim * 4]) # include u from MCG
            target_b = np.zeros([batch_size, Dataset.q_dim])

def preprocessing_data(Dataset):

    input_b = np.zeros([len(Dataset), Dataset.q_dim * 4])
    target_b = np.zeros([len(Dataset), Dataset.q_dim])
    for i, (state, action) in enumerate(Dataset):
        input_b[i,:] = np.concatenate([state, action[0, :][np.newaxis, :]], axis=0).reshape(-1,)
        target_b[i,:] = action[0, :] - action[1, :] # pd_tau - mea_tau
    input_b = torch.from_numpy(input_b).float()
    target_b = torch.from_numpy(target_b).float()
    return input_b, target_b

def validate(vali_input, vali_target, Net, criterion):
    Net.eval()
    with torch.no_grad():
        output = Net(vali_input)
        Loss = criterion(output, vali_target)
    Net.train()
    return Loss.item()

def evaluate(time, test_input, test_target, real_tau, pd_tau, Net, criterion, prefix):
    Net.eval()
    with torch.no_grad():
        output = Net(test_input)
        Loss = criterion(output, test_target)
    Net.train()

    output = output.cpu().numpy()
    tau_inf_withfric = real_tau + output
    plt.figure(figsize=[12,12])
    for channel in range(7):
        plt.subplot(4,2,channel+1,xlabel="time[s]", ylabel="Torque of {}. joint[n/m]".format(channel+1))
        plt.plot(time, real_tau[:,channel],'-c', linewidth=.7, label="u_measure")
        plt.plot(time, pd_tau[:,channel],'-b', linewidth=.7, label="u_pd")
        plt.plot(time, tau_inf_withfric[:,channel],'-r', linewidth=.7, label="u_real + friction")
    plt.legend()
    plt.savefig('./Evaluation_Plot/{}_evaluation.jpg'.format(prefix), dpi=200)
    
def main(args):
    logging.basicConfig(filename='training_log', filemode='a', \
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',\
                            datefmt='%H:%M:%S',\
                            level=logging.INFO)
    Dtrain = NNDatasets('../data/trajectories', 1, False, True) #initialize dataset
    Dvali = NNDatasets('../data/test_trajectory', 1, False, True)
    Dtest = NNDatasets('../data/8_figure', 1, False, False)
    vali_input, vali_target = preprocessing_data(Dvali)
    test_input, test_target = preprocessing_data(Dtest)

    if args.Ntype == 'NB':
        Net = NetBase(Dtrain.input_size, Dtrain.output_size, args.dor, args.hidden)
        print(Net)
    else:
        raise ValueError('NN not found!')

    optimizer = Adam(Net.parameters(), lr=args.lr, weight_decay=args.decay)
    Criterion = MSELoss()
    optimizer.zero_grad()

    vali_loss_l = []
    train_loss_l = []
    for ep in tqdm(range(args.epoch)):
        Loss_total = 0
        Step_counter = 0
        vali_loss = validate(vali_input, vali_target, Net, Criterion)
        for input_b, target_b in batch_generator(Dtrain, args.batch):
            # state = q, qd, qdd; torques = tau_cmd, tau_real
            optimizer.zero_grad()

            output = Net(input_b)
            Loss = Criterion(output, target_b)
            Loss.backward()
            Loss_total += Loss.item()
            Step_counter += 1
            optimizer.step()
        torch.save(Net, './Model/HU_{}_DR_{}_EP_{}.pth'.format(args.hidden, args.decay, ep))
        vali_loss_l.append(vali_loss)
        train_loss_l.append(Loss_total / Step_counter)
    
    logging.info('The {} hidden units, {} weight decay got {:.3f} validataion error at {}. episode.'.format(args.hidden, args.decay, np.min(vali_loss_l), np.argmin(vali_loss_l)))

    evaluate(Dtest.time_list, test_input, test_target, Dtest.tau_real, Dtest.tau_u_G, Net, Criterion, str(args.hidden)+'_'+str(args.decay))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Hyper parameters of NN training...')
    parser.add_argument('--batch', type=int, required=True, help='batch size')
    parser.add_argument('--Ntype', type=str, required=True, help='NB for baseNet')
    parser.add_argument('--epoch', type=int, required=True, help='training length')
    parser.add_argument('--dor', type=float, required=True, help='drop out rate')
    parser.add_argument('--hidden', type=int, required=True, help='MLP hidden unit number')
    parser.add_argument('--decay', type=float, required=True, help='weight decay')
    parser.add_argument('--lr', type=float, required=True, help='learning rate')
    args = parser.parse_args()

    main(args)
