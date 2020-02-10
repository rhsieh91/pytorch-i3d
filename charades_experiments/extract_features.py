# Contributer: piergiaj
# Modified by Samuel Kwong

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)

args = parser.parse_args()
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


def run(max_steps=64e3, mode='rgb', root='/vision/group/Charades_RGB/Charades_v1_rgb', split='./data/annotations/charades.json', batch_size=16, load_model='', save_dir='./features_charades_i3d'):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device: {}'.format(device))

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    print('Making train dataset...')
    if os.path.exists('./data/charades_train_dataset_original.pickle'):
        pickle_in = open('./data/charades_train_dataset_original.pickle', 'rb')
        dataset = pickle.load(pickle_in)
    else:
        dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
        pickle_out = open('./data/charades_train_dataset_original.pickle', 'wb')
        pickle.dump(dataset, pickle_out)
        pickle_out.close()
    print('Finished making train dataset.')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print('Making validation dataset...')
    if os.path.exists('./data/charades_val_dataset_original.pickle'):
        pickle_in = open('./data/charades_val_dataset_original.pickle', 'rb')
        val_dataset = pickle.load(pickle_in)
    else:
        val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
        pickle_out = open('./data/charades_val_dataset_original.pickle', 'wb')
        pickle.dump(val_dataset, pickle_out)
        pickle_out.close()
    print('Finished making val dataset.')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)    

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
    
    ## Note:
    ## nn.DataParallel has been giving problems
    ## Problem 1: AttributeError: 'DataParallel' object has no attribute 'extract_features'
    ## Problem 2: Supposedly only works with batch_size > 1? But we batch_size > 1 poses dataloader problems atm...
    #if torch.cuda.device_count() > 1:
    #    print('Multiple GPUs detected: {}'.format(torch.cuda.device_count()))
    #    i3d = nn.DataParallel(i3d)
    #else:
    #    print('Using single GPU')
    i3d = i3d.to(device=device)

    for phase in ['train', 'val']:
        print('Extracting {} set features'.format(phase))
        i3d.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        for i, data in enumerate(dataloaders[phase]):
            print('Batch: {}'.format(i))
            # get the inputs
            inputs, labels, name = data
            #inputs = inputs.to(device=device, dtype=torch.float32)
            print(inputs.shape)
            if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
                continue

            b,c,t,h,w = inputs.shape
            if t > 1600:
                features = []
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)
                    #ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                    with torch.no_grad():
                        #features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                        ip = torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda()
                        features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                # wrap them in Variable
                #inputs = Variable(inputs.cuda(), volatile=True)
                with torch.no_grad():
                    features = i3d.extract_features(inputs.cuda())
                np.save(os.path.join(save_dir, name[0]), features.squeeze(0).permute(1,2,3,0).data.cpu().numpy())


if __name__ == '__main__':
    # need to add argparse
    #run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
    run(mode='rgb', root='/vision/group/Charades_RGB/Charades_v1_rgb', split='./data/annotations/charades.json', batch_size=1, load_model='./models/rgb_imagenet.pt', save_dir='./features_charades_i3d')
