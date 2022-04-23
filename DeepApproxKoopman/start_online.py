from onlineTrain import OnlineTrainer
import argparse
from utils import reset_workspace

# necessary parameters
parser = argparse.ArgumentParser(description='Hyper parameters of NN training...')
parser.add_argument('--batch', type=int, required=True, help='batch size')
parser.add_argument('--epoch', type=int, required=True, help='training length')
parser.add_argument('--dor', type=float, required=True, help='drop out rate')
parser.add_argument('--decay', type=float, required=True, help='weight decay')
parser.add_argument('--lr', type=float, required=True, help='learning rate')
parser.add_argument('--k', type=int, required=True, help='AB predict length')
parser.add_argument('--lift', type=int, required=True, help='lifting state dimension')
parser.add_argument('--state_dim', type=int, required=True, help='original state dimension')
parser.add_argument('--control_dim', type=int, required=True, help='control input dimension')
parser.add_argument('--validate_k', type=int, required=True, help='How many step ahead in validate')
parser.add_argument('--lr_decay', type=float, required=True, help='Learning rate decay.')
parser.add_argument('--maxlen', type=int, required=True, help='maximal length of replay buffer, better be multiple of 12')
parser.add_argument('--simNum', type=int, required=True, help='simulation number of online training.')
parser.add_argument('--trainNum', type=int, required=True, help='for online training, each epoch update the neural networks trainNum times')
parser.add_argument('--updateNum', type=int, required=True, help='for online training, each epoch how many times sample from environment (multiprocessing: 1 for 12 traj)')
parser.add_argument('--whichOde', type=str, required=True, help='which ode environment')
parser.add_argument('--whichController', type=str, required=True, help='which controller')
parser.add_argument('--whichModel', type=str, required=True, help='the folder name of offline pretrained model')
parser.add_argument('--pretrained', type=str, help='folder name containing pretrained model')


args = parser.parse_args()

# remove all the records last training
reset_workspace('online')

# create online trainer
OT = OnlineTrainer(args)
# start training
OT.train()
