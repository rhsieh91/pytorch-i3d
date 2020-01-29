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
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from collections import OrderedDict

# ----------------- Modify these --------------------

NUM_ACTIONS = 157
# NUM_SCENES = 2
NUM_FEATURES = 1024
BATCH_SIZE = 1
FEATURES_SAVE_PATH = '/vision/group/Charades_single_action/i3d_features'

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

    for i, data in enumerate(test_loader):
        print("Extracting features from batch {}".format(i))
        inputs = data[0]
        inputs = inputs.to(device=device, dtype=torch.float32) 
        with torch.no_grad():
            features = model.extract_features(inputs)

        features = features.squeeze()
        features = features.cpu().detach().numpy()
        inputs_features = np.append(inputs_features, features, axis=0)

    print('inputs_features shape = {}'.format(inputs_features.shape))
    print('Saving features')
    return inputs_features

def get_test_loader(model):
    # Transforms
    SPATIAL_TRANSFORM = Compose([
        Resize((224, 224)),
        ToTensor()
        ])
    
    # Load dataset
    vf = VideoFolder(root="/vision/group/Charades_single_action/single_action_rgb",
                          csv_file_input="/vision/group/Charades_single_action/Charades_single_action_train.csv",
                          csv_file_action_labels="/vision/group/Charades_single_action/Charades_v1_actions.csv",
                          csv_file_scene_labels="/vision/group/Charades_single_action/Charades_v1_scenes.csv",
                          clip_size=64,
                          nclips=1,
                          step_size=1,
                          is_val=True, # True means don't randomly offset clips (i.e. don't augment dataset)
                          transform=SPATIAL_TRANSFORM,
                          loader=default_loader)

    print('Size of training set = {}'.format(len(vf)))
    test_loader = DataLoader(vf, 
                              batch_size=BATCH_SIZE,
                              shuffle=False, 
                              num_workers=2,
                              pin_memory=True)

    return test_loader

# ------------------------------------------------------------

if __name__ == '__main__':

    i3d = InceptionI3d(NUM_ACTIONS, in_channels=3)
    model = i3d
    # model.replace_logits(NUM_ACTIONS)

    test_loader = get_test_loader(model)
    inputs_features = extract_data(model, test_loader)
    np.save(FEATURES_SAVE_PATH, inputs_features)
    