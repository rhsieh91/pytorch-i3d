# Contributer: piergiaj
# Modified by Samuel Kwong

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
#import argparse
#
#parser = argparse.ArgumentParser()
#parser.add_argument('-mode', type=str, help='rgb or flow')
#parser.add_argument('-load_model', type=str)
#parser.add_argument('-root', type=str)
#parser.add_argument('-gpu', type=str)
#parser.add_argument('-save_dir', type=str)
#
#args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
#from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset
import pickle


def run(max_steps=64e3, mode='rgb', root='/vision/group/Charades_RGB/Charades_v1_rgb', split='./data/annotations/charades.json', batch_size=4, load_model='', save_dir='./features_charades_i3d'):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device: {}'.format(device))

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    print('Getting train dataset...')
    if os.path.exists('./data/train_dataset_original.pickle'):
        pickle_in = open('./data/train_dataset_original.pickle', 'rb')
        dataset = pickle.load(pickle_in)
    else:
        dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
        pickle_out = open('./data/train_dataset_original.pickle', 'wb')
        pickle.dump(dataset, pickle_out)
        pickle_out.close()
    print('Got train dataset.')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print('Getting validation dataset...')
    if os.path.exists('./data/val_dataset_original.pickle'):
        pickle_in = open('./data/val_dataset_original.pickle', 'rb')
        val_dataset = pickle.load(pickle_in)
    else:
        val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
        pickle_out = open('./data/val_dataset_original.pickle', 'wb')
        pickle.dump(val_dataset, pickle_out)
        pickle_out.close()
    print('Got val dataset.')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    print('Loading model...')
    i3d.load_state_dict(torch.load(load_model))
    print('Finished loading model.')
    i3d.replace_logits(157)
    i3d.cuda()
    
    if torch.cuda.device_count() > 1:
        print('Multiple GPUs detected: {}'.format(torch.cuda.device_count()))
        i3d = nn.DataParallel(i3d)
    else:
        print('Using single GPU')
    i3d = i3d.to(device=device)

    for phase in ['train', 'val']:
        print('Extracting {} set features'.format(phase))
        i3d.train(False)  # Set model to evaluate mode
                    
        # Iterate over data.
        for i,data in enumerate(dataloaders[phase]):
            print('Batch: {}'.format(i))
            # get the inputs
            inputs, labels, name = data
            inputs = inputs.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                features = i3d.module.extract_features(inputs.cuda())
                for f,n in zip(features,name):
                    if os.path.exists(os.path.join(save_dir, n+'.npy')):
                        continue
                    np.save(os.path.join(save_dir, n), f.permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':
    #run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
    run(mode='rgb', root='/vision/group/Charades_RGB/Charades_v1_rgb', split='./data/annotations/charades.json', batch_size=8, load_model='./models/rgb_imagenet.pt', save_dir='./features_charades_i3d')
