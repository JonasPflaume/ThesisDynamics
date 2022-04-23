## Helping functions ##
import numpy as np
import os
import glob

def read_data(path:str) -> np.ndarray:
    ''' read data from *.data file
        return a time*99 numpy array
    '''
    if not path.endswith('dat'):
        raise ValueError("This repo contains no .dat file!")
    data = []
    with open(path, 'r') as F:
        d = F.readlines()
        for i in d:
            k = i.rstrip().split(" ")
            k = list(filter(lambda x: x!="", k))
            if len(k) == 99 or len(k)==106:
                data.append(k)
            else:
                print("Skipped one line broken data...")
    data = np.array(data, dtype=float)
    return data

def friction(x:tuple, speed:float) -> float:
    ''' return friction torque for one specific joint given identified parameters tuple x '''

    phi1j, phi2j, phi3j = x
    tau = phi1j/(1+np.exp(-phi2j*(speed+phi3j))) - phi1j/(1+np.exp(-phi2j*phi3j))
    return tau


def reset_workspace(workspace='offline'):
    ''' delete all the results images and models in folders
    '''
    root = os.path.dirname(os.path.abspath(__file__))
    offline_imagedir = root + '/Plots/OfflineTrainingPlots/*'
    offline_imagefiles = glob.glob(offline_imagedir)
    online_imagedir = root + '/Plots/OnlineTrainingPlots/*'
    online_imagefiles = glob.glob(online_imagedir)
    online_model_dir = root + '/Models/OnlineTrainingModels'
    offline_model_dir = root + '/Models/OfflineTrainingModels'
    online_video_dir = root + '/Plots/GymVideo/*'
    
    if workspace == 'online':
        for root, dirs, files in os.walk(online_model_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        for n in online_imagefiles:
            os.remove(n)
        
        for root, dirs, files in os.walk(online_video_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    elif workspace == 'offline':
        for root, dirs, files in os.walk(offline_model_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

        for n in offline_imagefiles:
            os.remove(n)

