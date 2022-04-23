import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plotFunc(f, bounds_lo, bounds_up, trace_xy = None, trace_z = None):
    x = np.linspace(bounds_lo[0], bounds_up[0], 100)
    y = np.linspace(bounds_lo[1], bounds_up[1], 100)
    xMesh, yMesh = np.meshgrid(x, y, indexing='ij')
    zMesh = np.zeros_like(xMesh)
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            entry = np.array([xMesh[i,j], yMesh[i,j]]).reshape(-1,1)
            zMesh[i,j] = f(entry)

    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(xMesh, yMesh, zMesh, cmap=cm.coolwarm)
    if trace_xy is not None and trace_z is not None:
        trace_xy = np.concatenate(trace_xy, axis=1).T
        trace_z = np.concatenate(trace_z)[:,np.newaxis]
    trace_z = trace_z.squeeze()
    if len(trace_z.shape) == 1:
        trace_z = trace_z.reshape(-1,1)
    if trace_xy is not None: ax1.plot(trace_xy[:,0], trace_xy[:,1], trace_z[:,0], 'ko-')
    fig.colorbar(surf)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f')
    ax2 = fig.add_subplot(122)
    surf2 = plt.contourf(xMesh, yMesh, zMesh, cmap=cm.coolwarm)
    if trace_xy is not None: ax2.plot(trace_xy[:,0], trace_xy[:,1], 'ko-')
    fig.colorbar(surf2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.show()
