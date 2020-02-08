# Contributer: piergiaj
# Modified by Samuel Kwong

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset
import pickle


def run(max_steps=64e3, mode='rgb', root='/vision/group/Charades_RGB/Charades_v1_rgb', split='./data/annotations/charades.json', batch_size=1, load_model='', save_dir='./features_charades_i3d'):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    print('Making train dataset...')
    if os.path.exists('./data/charades_train_dataset_original.pickle'):
        pickle_in = open('./data/charades_train_dataset_original.pickle')
        dataset = pickle.load(pickle_in)
    else:
        dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
        pickle_out = open('./data/charades_train_dataset_original.pickle', 'wb')
        pickle.dump(dataset, pickle_out)
        pickle_out.close()
    print('Finished making train dataset.')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    print('Making validation dataset...')
    if os.path.exists('./data/charades_val_dataset_original.pickle'):
        pickle_in = open('./data/charades_val_dataset_original.pickle')
        val_dataset = pickle.load(pickle_in)
    else:
        val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
        pickle_out = open('./data/charades_val_dataset_original.pickle', 'wb')
        pickle.dump(val_dataset, pickle_out)
        pickle_out.close()
    print('Finished making val dataset.')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}


if __name__ == '__main__':
    run(mode='rgb', root='/vision/group/Charades_RGB/Charades_v1_rgb', split='./data/annotations/charades.json', batch_size=1, load_model='./models/rgb_imagenet.pt', save_dir='./features_charades_i3d')
