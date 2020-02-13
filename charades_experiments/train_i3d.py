# Contributor: piergiaj
# Modified by Samuel Kwong

import sklearn.metrics as metrics
import os
import sys
import argparse
import pickle 
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

import numpy as np
from pytorch_i3d import InceptionI3d
from charades_dataset_full import Charades as Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--bs', type=int, help='batch size')
parser.add_argument('--stride', type=int, help='temporal stride for sampling input frames')
parser.add_argument('--num_span_frames', type=int, help='total number of frames to sample per input')
args = parser.parse_args()

def save_checkpoint(model, optimizer, loss, save_dir, epoch, steps):
    """Saves checkpoint of model weights during training."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = save_dir + str(epoch).zfill(2) + str(steps).zfill(6) + '.pt'
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'steps': steps 
                },
                save_path)

# From https://github.com/facebookresearch/video-long-term-feature-banks/blob/master/lib/utils/metrics.py
def mean_ap_metric(predicts, targets):
    """Compute mAP, wAP, AUC for Charades."""

    predicts = np.vstack(predicts.cpu().detach().numpy())
    targets = np.vstack(targets.cpu().detach().numpy())

    predict = predicts[:, ~np.all(targets == 0, axis=0)]
    target = targets[:, ~np.all(targets == 0, axis=0)]
    mean_auc = 0
    aps = [0]
    try:
        mean_auc = metrics.roc_auc_score(target, predict)
    except ValueError:
        print(
            'The roc_auc curve requires a sufficient number of classes \
            which are missing in this sample.'
        )
    try:
        aps = metrics.average_precision_score(target, predict, average=None)
    except ValueError:
        print(
            'Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample.'
        )

    mean_ap = np.mean(aps)
    weights = np.sum(target.astype(float), axis=0)
    weights /= np.sum(weights)
    mean_wap = np.sum(np.multiply(aps, weights))
    all_aps = np.zeros((1, targets.shape[1]))
    all_aps[:, ~np.all(targets == 0, axis=0)] = aps

    return mean_auc, mean_ap, mean_wap, all_aps.flatten()

def run(init_lr=0.1, mode='rgb', root='', split='data/annotations/charades.json', batch_size=8, save_dir='', stride=4, num_span_frames=125, num_epochs=150):
    writer = SummaryWriter() # tensorboard logging
    
    # setup dataset
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ToTensor()
                                          ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor()
                                         ])
    
    print('Getting train dataset...')
    train_path = './data/train_dataset_{}_{}.pickle'.format(stride, num_span_frames)
    if os.path.exists(train_path):
        pickle_in = open(train_path, 'rb')
        train_dataset = pickle.load(pickle_in)
    else:
        train_dataset = Dataset(split, 'training', root, mode, train_transforms, stride, num_span_frames)
        pickle_out = open(train_path, 'wb')
        pickle.dump(train_dataset, pickle_out)
        pickle_out.close()
    print('Got train dataset.')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

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

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    
    print('Loading model...')
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        #state_dict = torch.load('checkpoints/000990.pt')#['model_state_dict']
        #checkpoint = OrderedDict()
        #for k, v in state_dict.items():
        #    name = k[7:] # remove 'module'
        #    checkpoint[name] = v
    i3d.replace_logits(157)
    i3d.cuda()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        i3d = nn.DataParallel(i3d)
    i3d.to(device)
    print('Loaded model.')

    optimizer = optim.Adam(i3d.parameters(), lr=init_lr)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [40000, 8000], gamma=0.1)

    steps = 0 # can also load from a step here: torch.load(<checkpoint>)['steps']
    start_epoch = 0 # torch.load(<checkpoint>)['epoch']
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
            num_correct = 0
            num_actions = 0
            print('Entering data loading...')
            for data in dataloaders[phase]:
                # get the inputs
                # note: for SIFE-Net we would also have scene_labels
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
                
                #num_correct += torch.sum((predicted_labels + labels) == 2)
                #num_actions += torch.sum(labels, dim=(0, 1)) 

                ## DEBUGGING
                #print('----------DEBUGGING----------')
                #print('labels:', labels)
                #print('predicted_labels', predicted_labels)
                #print('predicted_labels == labels', predicted_labels == labels)
                #print('num_correct:', num_correct)
                #print('num_actions:', num_actions)
                #print('num predicted actions:', float(torch.sum(predicted_labels, dim=(0,1))))

                # Loss
                if phase == 'train':
                    loss = F.binary_cross_entropy_with_logits(max_frame_logits, labels)
                    writer.add_scalar('loss/train', loss, steps)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print('Step {} {} loss: {:.4f}'.format(steps, phase, loss))
                    steps += 1

            # Accuracy
            predicted_labels = (torch.sigmoid(max_frame_logits) >= 0.5).float() # for eval metric calculation purposes
            mAP = mean_ap_metric(predicted_labels, labels)
            if phase == 'train':
                writer.add_scalar('mAP/train', mAP, epoch)
                print('-' * 50)
                print('{} mAP: {:.4f}'.format(phase, mAP))
                print('-' * 50)
                save_checkpoint(i3d, optimizer, loss, save_dir, epoch, steps) # save checkpoint after epoch!
            else:
                writer.add_scalar('mAP/val', mAP, epoch)
                print('{} mAP: {:.4f}'.format(phase, mAP))
        
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
        NUM_EPOCHS = 150
        SAVE_DIR = './checkpoints-{}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}/'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        with open(SAVE_DIR + 'info.txt', 'w+') as f:
            f.write('LR = {}\nBATCH_SIZE = {}\nSTRIDE = {}\nNUM_SPAN_FRAMES = {}\nEPOCHS = {}'.format(LR, BATCH_SIZE, STRIDE, NUM_SPAN_FRAMES, NUM_EPOCHS))
        
        run(init_lr=LR, root='/vision/group/Charades_RGB/Charades_v1_rgb', batch_size=BATCH_SIZE, save_dir=SAVE_DIR, stride=STRIDE, num_span_frames=NUM_SPAN_FRAMES, num_epochs=NUM_EPOCHS)
