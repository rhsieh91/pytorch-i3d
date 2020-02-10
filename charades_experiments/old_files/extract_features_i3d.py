import os
import pdb
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pytorch_i3d import InceptionI3d
from pytorch_sife import SIFE
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.tensorboard import SummaryWriter

from data_loader_jpeg import *

# from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from collections import OrderedDict

import nonechucks as nc

# ----------------- Modify these --------------------

NUM_ACTIONS = 157
NUM_FEATURES = 1024
BATCH_SIZE = 16
FEATURES_SAVE_PATH = '/vision/u/samkwong/pytorch-i3d/charades_experiments/i3d_features'

""" baseline i3d params """
IS_BASELINE = True # use baseline i3d
DATA_PARALLEL = True # model trained using nn.DataParallel

def extract_data(model, test_loader):
    # Move model to CPU/GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('Using device:', device)
    model = model.to(device=device) # move model parameters to CPU/GPU

    # Extract features and ground truth labels
    print('Starting feature extraction with batch size = {}'.format(BATCH_SIZE))
    inputs_features = np.empty((0, NUM_FEATURES)) # to hold all inputs' feature arrays

    i = 0
    for data in test_loader:
        print("Extracting features from batch {}".format(i))
        i += 1
        inputs = data[0]
        inputs = inputs.to(device=device, dtype=torch.float32) 
        with torch.no_grad():
            features = model.extract_features(inputs)

        print('Features shape =', features.shape)
        features = features.squeeze()
        
        features = features.cpu().detach().numpy()
        print('Inputs Features shape =', inputs_features.shape)
        print('Features shape =', features.shape)
        inputs_features = np.append(inputs_features, features, axis=0)

    print('inputs_features shape = {}'.format(inputs_features.shape))
    return inputs_features

def get_test_loader(model):
    print('Getting test_loader')
    # Transforms
    SPATIAL_TRANSFORM = Compose([
        Resize((224, 224)),
        ToTensor()
        ])
    
    # Load dataset
    vf = VideoFolder(root="/vision/group/Charades_RGB/Charades_v1_rgb",
                          csv_file_input="/vision/group/Charades/annotations/Charades_v1_train.csv",
                          csv_file_actions_labels="/vision/u/samkwong/pytorch-i3d/charades_experiments/data/annotations/Charades_v1_actions.csv",
                          csv_file_scene_labels="/vision/u/samkwong/pytorch-i3d/charades_experiments/data/annotations/Charades_v1_scenes.csv",
                          clip_size=128,
                          nclips=1,
                          step_size=1,
                          is_val=True, # True means don't randomly offset clips (i.e. don't augment dataset)
                          transform=SPATIAL_TRANSFORM,
                          loader=default_loader)

    #vf = nc.SafeDataset(vf) # skip over any noisy samples

    print('Size of training set = {}'.format(len(vf)))
    test_loader = DataLoader(vf, 
                              batch_size=BATCH_SIZE,
                              shuffle=False, 
                              num_workers=2,
                              pin_memory=True)

    print('Aqcuired test_loader!')
    return test_loader

# ------------------------------------------------------------

if __name__ == '__main__':

    i3d = InceptionI3d(NUM_ACTIONS, in_channels=3)
    model = i3d
    # model.replace_logits(NUM_ACTIONS)

    test_loader = get_test_loader(model)
    inputs_features = extract_data(model, test_loader)
    print('Saving features')
    np.save(FEATURES_SAVE_PATH, inputs_features)
    
