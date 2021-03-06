# Contributor: piergiaj
# Modified by Samuel Kwong and Richard Hsieh
from sklearn import metrics

import os
import sys
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
import videotransforms
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

import numpy as np
from pytorch_i3d import InceptionI3d
from charades_dataset import Charades as Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--bs', type=int, help='batch size')
parser.add_argument('--stride', type=int, help='temporal stride for sampling input frames')
parser.add_argument('--num_span_frames', type=int, help='total number of frames to sample per input')
parser.add_argument('--checkpoint_path', type=str, help='path to saved checkpoint (\'\' to train from kinetics baseline)')
args = parser.parse_args()

def save_checkpoint(model, optimizer, loss, save_dir, epoch, steps):
    """Saves checkpoint of model weights during training."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = save_dir + str(epoch).zfill(3) + '.pt'
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'steps': steps 
                },
                save_path)

def run(init_lr=0.01, root='', split_file='data/annotations/charades.json', batch_size=8, save_dir='', stride=4, num_span_frames=32, num_epochs=200):
    writer = SummaryWriter() # tensorboard logging
    
    # setup dataset
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ToTensor()
                                          ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor()
                                         ])
    
    print('Getting train dataset...')
    train_dataset = Dataset(split_file, 'training', root, train_transforms, stride, num_span_frames, is_sife=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print('Getting validation dataset...')
    val_dataset = Dataset(split_file, 'testing', root, test_transforms, stride, num_span_frames, is_sife=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)    

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    
    print('Loading model...')
    # setup the model

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

    optimizer = optim.Adam(i3d.parameters(), lr=init_lr)
    #lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma=0.1)

    steps = 0 if not args.checkpoint_path else torch.load(args.checkpoint_path)['steps']
    start_epoch = 0 if not args.checkpoint_path else torch.load(args.checkpoint_path)['epoch']
    
    # TRAIN
    for epoch in range(start_epoch, num_epochs):
        print('-' * 50)
        print('EPOCH {}/{}'.format(epoch, num_epochs))
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
                print('-'*10, 'TRAINING', '-'*10)
            else:
                i3d.train(False)  # Set model to evaluate mode
                print('-'*10, 'VALIDATION', '-'*10)
            
            # Iterate over data.
            all_preds = []
            all_labels = []
            print('Entering data loading...')
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, labels, vid = data

                t = inputs.shape[2]
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                if phase == 'train':
                    per_frame_logits = i3d(inputs)
                else:
                    with torch.no_grad():
                        per_frame_logits = i3d(inputs)

                # upsample to input size
                per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear') # B x Classes x T
                
                max_frame_logits = torch.max(per_frame_logits, dim=2)[0] # B x Classes
                labels = torch.max(labels, dim=2)[0] # B x Classes

                if phase == 'train':
                    loss = F.binary_cross_entropy_with_logits(max_frame_logits, labels)
                    writer.add_scalar('loss/train', loss, steps)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if steps % 10 == 0:
                        print('Step {} {} loss: {:.4f}'.format(steps, phase, loss))
                    steps += 1
                    
                # metrics for validation
                pred = (torch.sigmoid(max_frame_logits) >= 0.5).float() # predicted labels for this batch (B x C)
                if i == 0:
                    all_preds = np.array(pred.tolist())
                    all_labels = np.array(labels.tolist())
                else:
                    all_preds = np.append(all_preds, pred.tolist(), axis=0)
                    all_labels = np.append(all_labels, labels.tolist(), axis=0)

            # Eval
            all_APs = [metrics.average_precision_score(y_true=all_labels[:, j], y_score=all_preds[:, j]) for j in range(157)]
            mAP = np.nanmean(all_APs)
            if phase == 'train':
                writer.add_scalar('mAP/train', mAP, epoch)
                print('-' * 50)
                print('{} mAP: {:.4f}'.format(phase, mAP))
                print('-' * 50)
                save_checkpoint(i3d, optimizer, loss, save_dir, epoch, steps) # save checkpoint after epoch!
            else:
                writer.add_scalar('mAP/val', mAP, epoch)
                print('{} mAP: {:.4f}'.format(phase, mAP))
        
        #lr_sched.step() # step after epoch
        
    writer.close()
     

if __name__ == '__main__':
    if len(sys.argv) < len(vars(args))+1:
        parser.print_usage()
        parser.print_help()
    else:
        print('Starting...')
        now = datetime.datetime.now()

        LR = args.lr
        BATCH_SIZE = args.bs
        STRIDE = args.stride # temporal stride for sampling
        NUM_SPAN_FRAMES = args.num_span_frames # total number frames to sample for inputs
        NUM_EPOCHS = 200
        SAVE_DIR = './checkpoints-{}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}/'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        with open(SAVE_DIR + 'info.txt', 'w+') as f:
            f.write('MODEL = {}\nLR = {}\nBATCH_SIZE = {}\nSTRIDE = {}\nNUM_SPAN_FRAMES = {}\nEPOCHS = {}'.format('I3D', LR, BATCH_SIZE, STRIDE, NUM_SPAN_FRAMES, NUM_EPOCHS))
        
        run(init_lr=LR, root='/vision/group/Charades_RGB/Charades_v1_rgb', batch_size=BATCH_SIZE,
            save_dir=SAVE_DIR, stride=STRIDE, num_span_frames=NUM_SPAN_FRAMES, num_epochs=NUM_EPOCHS)
