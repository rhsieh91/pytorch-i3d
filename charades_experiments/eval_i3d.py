# Contributor: piergiaj
# Modified by Samuel Kwong and Richard Hsieh
from sklearn import metrics

import os
import sys
import argparse
import pickle 
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import videotransforms
from collections import OrderedDict

import numpy as np
from pytorch_i3d import InceptionI3d
from charades_dataset_i3d import Charades as Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, help='batch size')
parser.add_argument('--stride', type=int, help='temporal stride for sampling input frames')
parser.add_argument('--num_span_frames', type=int, help='total number of frames to sample per input')
parser.add_argument('--checkpoint_path', type=str, help='path to saved checkpoint (\'\' to train from kinetics baseline)')
args = parser.parse_args()


def run(mode='rgb', root='', split='data/annotations/charades.json', batch_size=8, stride=4, num_span_frames=125):
    
    # setup dataset
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor()
                                         ])
    
    print('Getting validation dataset...')
    val_path = './data/val_dataset_{}_{}.pickle'.format(stride, num_span_frames)
    if os.path.exists(val_path):
        pickle_in = open(val_path, 'rb')
        val_dataset = pickle.load(pickle_in)
    else:
        val_dataset = Dataset(split, 'testing', root, mode, test_transforms, stride, num_span_frames)
        pickle_out = open(val_path, 'wb')
        pickle.dump(val_dataset, pickle_out)
        pickle_out.close()
    print('Got val dataset.')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)    

    
    print('Loading model...')
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        if args.checkpoint_path:
            i3d.replace_logits(157)
            state_dict = torch.load(args.checkpoint_path)['model_state_dict']
            checkpoint = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module'
                checkpoint[name] = v
            i3d.load_state_dict(checkpoint)
        else:
            i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
            i3d.replace_logits(157)
    i3d.cuda()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        i3d = nn.DataParallel(i3d)
    i3d.to(device)
    print('Loaded model.')


    all_preds = [] #torch.zeros((, 157)).cuda()
    all_labels = [] #torch.zeros((, 157)).cuda()
    print('Entering data loading...')
    for i, data in enumerate(val_dataloader):
        # get the inputs
        inputs, labels, vid = data

        t = inputs.shape[2]
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        with torch.no_grad():
            per_frame_logits = i3d(inputs)

        # upsample to input size
        per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear') # B x Classes x T
        
        max_frame_logits = torch.max(per_frame_logits, dim=2)[0] # B x Classes
        labels = torch.max(labels, dim=2)[0] # B x Classes

        # metrics for validation
        pred = (torch.sigmoid(max_frame_logits) >= 0.5).float() # predicted labels for this batch (B x C)
        if i == 0:
            all_preds = np.array(pred.tolist())
            all_labels = np.array(labels.tolist())
        else:
            all_preds = np.append(all_preds, pred.tolist(), axis=0)
            all_labels = np.append(all_labels, labels.tolist(), axis=0)
        #print('Step {}: all_preds.shape={}, all_labels.shape={}'.format(i, all_preds.shape, all_labels.shape))
        #print('Step {}: all_preds={}, all_labels={}'.format(i, all_preds, all_labels))
        if i % 10 == 0:
            all_APs = [metrics.average_precision_score(y_true=all_labels[:, j], y_score=all_preds[:, j]) for j in range(157)]
            mAP = np.nanmean(all_APs)
            print('Step {}'.format(i))
            print('all_APs:')
            print(all_APs)
            print('mAP = {}'.format(mAP))

    # Eval
    all_APs = [metrics.average_precision_score(y_true=all_labels[:, j], y_score=all_preds[:, j]) for j in range(157)]
    mAP = np.nanmean(all_APs)
    print('-' * 50)
    print('Final mAP: {:.4f}'.format(mAP))
    print('-' * 50)
     

if __name__ == '__main__':
    if len(sys.argv) < len(vars(args))+1:
        parser.print_usage()
        parser.print_help()
    else:
        print('Starting...')

        BATCH_SIZE = args.bs
        STRIDE = args.stride # temporal stride for sampling
        NUM_SPAN_FRAMES = args.num_span_frames # total number frames to sample for inputs

        run(root='/vision/group/Charades_RGB/Charades_v1_rgb', batch_size=BATCH_SIZE, stride=STRIDE, num_span_frames=NUM_SPAN_FRAMES)
