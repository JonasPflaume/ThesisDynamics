import sys
sys.path.append('..')
from Controller import FiniteLQR
import gym
from gym import monitoring as monitor
import torch
import numpy as np
import time
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rc('font', **{'size': 16})
plt.rcParams.update({
  "text.usetex": True
})

simNum = 100
# initialize the reference trajectory for different controller
modelpath = '../Models/DoneModel/' + '1'
env = gym.make('Pendulum-v0')
ABC = torch.load(modelpath + '/ABC.pth').cpu()
EnNet = torch.load(modelpath + '/EnNet.pth').cpu()
xref = np.zeros([3, 1])
xref[0,:] = 1
xref = torch.from_numpy(xref).float().cpu()
xref = EnNet(xref.T)
xref = xref.cpu().detach().numpy()

x_0 = env.reset()
x_0 = x_0[:, np.newaxis]
print(x_0.shape)
res_x = [x_0]
x_0_tensor = torch.from_numpy(x_0.T).float()
with torch.no_grad():
    x_0_tensor_lifting = EnNet(x_0_tensor)
x_0_lifting = x_0_tensor_lifting.cpu().detach().numpy()
u_k = np.zeros([1, 1])
res_u = [u_k]

ABCweight = list(ABC.named_parameters())
Ctrl = FiniteLQR(ABCweight, simNum)

# record video for gym env
curr_time = int(time.time())
video_path = './{}.mp4'.format(curr_time % 10000)
video_recorder = monitor.VideoRecorder(
            env, video_path, enabled=video_path is not None)
##### main loop #####
for i in range(simNum-1):
    env.unwrapped.render()
    video_recorder.capture_frame()
    
    x_0_lifting.reshape(25, 1)
    u_k = Ctrl(x_0_lifting.T, xref, i)

    x_next, _, _, _ = env.step(u_k)
    x_0 = x_next

    res_x.append(x_next.reshape(3,1))
    res_u.append(u_k)
    x_0_tensor = torch.from_numpy(x_0.T).float()


    with torch.no_grad():
        x_0_tensor_lifting = EnNet(x_0_tensor)
    x_0_lifting = x_0_tensor_lifting.cpu().detach().numpy()


video_recorder.close()
video_recorder.enabled = False
res_x = np.concatenate(res_x, axis=1)
res_u = np.concatenate(res_u, axis=0)
time = np.linspace(0, simNum * 0.05, simNum)

plt.figure(figsize=[10,8])
plt.subplot(211)
plt.grid()
plt.plot(time, res_x[0], '-r', label=r'$cos(\theta)$')
plt.plot(time, res_x[2], '-b', label=r'$\dot{\theta}$')
plt.ylabel('rad/s')
plt.legend()
plt.subplot(212)
plt.grid()
plt.plot(time, np.clip(res_u, -2, 2), '-r', label=r'action $\tau$')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Action [N]')
plt.savefig("GymEvaluation.jpg", dpi=200)
plt.show()