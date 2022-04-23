from utils import read_data, reset_workspace
from train import Trainer
from implicitTrain import Trainer2, Trainer3
#from PendulumTrajGenerator import pendulumGenerator
from odeSysBuilder import pendulum_ode_solver
import argparse

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
parser.add_argument('--implicit', type=str, required=True, help='Use implicit ABC')
parser.add_argument('--pretrained', type=str, help='folder name containing pretrained model')
args = parser.parse_args()

if args.state_dim == 4:
    from PendulumTrajGenerator import RecordTrajGenerator
elif args.state_dim == 2:
    from sliderTrajGenerator import RecordTrajGenerator
elif args.state_dim == 3:
    from GymEnvGenerator import RecordTrajGenerator
elif args.state_dim == 14:
    from PandaGenerator import RecordTrajGenerator


reset_workspace('offline')

# create offline trainer
if args.implicit == '1':
    print('Use implicit trainer')
    T = Trainer3(args)
else:
    print('Use parametric trainer')
    T = Trainer(args)
# passing pendulum trajectories generator
T.train(RecordTrajGenerator)
