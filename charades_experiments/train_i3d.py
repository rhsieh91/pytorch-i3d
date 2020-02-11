# Contributor: piergiaj
# Modified by Samuel Kwong

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import pickle 

parser = argparse.ArgumentParser()
parser.add_argument('-save_model', type=str)
parser.add_argument('-epochs', type=int)
args = parser.parse_args()

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

from torch.utils.tensorboard import SummaryWriter


def run(init_lr=0.1, mode='rgb', root='', split='data/annotations/charades.json', batch_size=8, save_model='', num_epochs=150):
    writer = SummaryWriter() # tensorboard logging
    
    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    
    print('Getting train dataset...')
    if os.path.exists('./data/train_dataset.pickle'):
        pickle_in = open('./data/train_dataset.pickle', 'rb')
        train_dataset = pickle.load(pickle_in)
    else:
        train_dataset = Dataset(split, 'training', root, mode, test_transforms)
        pickle_out = open('./data/train_dataset.pickle', 'wb')
        pickle.dump(train_dataset, pickle_out)
        pickle_out.close()
    print('Got train dataset.')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    print('Getting validation dataset...')
    if os.path.exists('./data/val_dataset.pickle'):
        pickle_in = open('./data/val_dataset.pickle', 'rb')
        val_dataset = pickle.load(pickle_in)
    else:
        val_dataset = Dataset(split, 'testing', root, mode, test_transforms)
        pickle_out = open('./data/val_dataset.pickle', 'wb')
        pickle.dump(val_dataset, pickle_out)
        pickle_out.close()
    print('Got val dataset.')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)    

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    i3d.replace_logits(157)
    #i3d.load_state_dict(torch.load('/ssd/models/000920.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])


    steps = 0
    # train it
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            loss = 0.0
            optimizer.zero_grad()
            
            # Iterate over data.
            print('About to start dataloader...')
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels, vid = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear') # B x C x T x H x W

                loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)

                ## compute classification loss (with max-pooling along time B x C x T)
                #cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                #tot_cls_loss += cls_loss.data[0]

                loss.backward()

                if phase == 'train':
                    writer.add_scalar('Train loss', loss.data, steps)
                    steps += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print('{} loss: {:.4f}'.format(phase, loss.data))
                        # save model
                        if not os.path.exists(save_model):
                            os.makedirs(save_model)
                        torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
            if phase == 'val':
                writer.add_scalar('Validation loss', loss.data, steps)
                print('{} loss: {:.4f}'.format(phase, loss.data))
    
    writer.close()
    

if __name__ == '__main__':
    run(mode='rgb', root='/vision/group/Charades_RGB/Charades_v1_rgb', save_model='./checkpoints/')
