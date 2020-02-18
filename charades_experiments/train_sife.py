# Contributor: piergiaj
# Modified by Samuel Kwong

from sklearn import metrics
import os
import sys
import argparse
import pickle 
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--bs', type=int, help='batch size')
parser.add_argument('--stride', type=int, help='temporal stride for sampling input frames')
parser.add_argument('--num_span_frames', type=int, help='total number of frames to sample per input')
parser.add_argument('--num_features', type=int, help='size of feature space (64 frames=7168, 32 frames=3072, 16 frames=1024)')
args = parser.parse_args()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
from torchvision import datasets, transforms

import numpy as np
from pytorch_i3d import InceptionI3d
from pytorch_sife import SIFE
from charades_dataset_sife import Charades as Dataset

from torch.utils.tensorboard import SummaryWriter

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

def run(init_lr=0.01, mode='rgb', root='', split='data/annotations/charades.json', 
        train_scene_map_pkl='./data/annotations/charades_train_scene_map.pkl',
        test_scene_map_pkl='./data/annotations/charades_test_scene_map.pkl',
        num_features=1024, batch_size=8, save_dir='', stride=4, num_span_frames=32, num_epochs=200):

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
        train_dataset = Dataset(split, train_scene_map_pkl, test_scene_map_pkl, 'training', root, mode, train_transforms, stride, num_span_frames)
        pickle_out = open(train_path, 'wb')
        pickle.dump(train_dataset, pickle_out)
        pickle_out.close()
    print('Got train dataset.')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    print('Getting validation dataset...')
    val_path = './data/val_dataset_{}_{}.pickle'.format(stride, num_span_frames)
    if os.path.exists(val_path):
        pickle_in = open(val_path, 'rb')
        val_dataset = pickle.load(pickle_in)
    else:
        val_dataset = Dataset(split, train_scene_map_pkl, test_scene_map_pkl, 'testing', root, mode, test_transforms, stride, num_span_frames)
        pickle_out = open(val_path, 'wb')
        pickle.dump(val_dataset, pickle_out)
        pickle_out.close()
    print('Got val dataset.')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)    

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    
    print('Loading model...')
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
        sife = SIFE(backbone=i3d, num_features=num_features, num_actions=157, num_scenes=16)
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        sife = SIFE(backbone=i3d, num_features=num_features, num_actions=157, num_scenes=16)
        #state_dict = torch.load('checkpoints/000990.pt')#['model_state_dict']
        #checkpoint = OrderedDict()
        #for k, v in state_dict.items():
        #    name = k[7:] # remove 'module'
        #    checkpoint[name] = v
    sife.cuda()
    sife = nn.DataParallel(sife)
    print('Loaded model.')

    lr = init_lr
    optimizer = optim.Adam(sife.parameters(), lr=lr)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [30], gamma=0.1)

    steps = 0 
    # TRAIN
    for epoch in range(num_epochs):
        print('-' * 50)
        print('EPOCH {}/{}'.format(epoch, num_epochs))
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                sife.train(True)
                print('-'*10, 'TRAINING', '-'*10)
            else:
                sife.train(False)  # Set model to evaluate mode
                print('-'*10, 'VALIDATION', '-'*10)
            
            # Iterate over data.
            all_action_preds = []
            all_scene_preds = []
            print('Entering data loading...')
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs, action_labels, scene_labels, vid = data
  
                t = inputs.shape[2]
                inputs = inputs.cuda()
                action_labels = action_labels.cuda() # B x num_classes x num_frames
                scene_labels = scene_labels.cuda() # B x num_frames
                
                if phase == 'train':
                    per_frame_action_logits, scene_logits = sife(inputs)
                else:
                    with torch.no_grad():
                        per_frame_action_logits, scene_logits = sife(inputs)

                # upsample to input size
                per_frame_action_logits = F.interpolate(per_frame_action_logits, t, mode='linear') # B x Classes x T
                max_frame_action_logits = torch.max(per_frame_action_logits, dim=2)[0] # B x Classes
                action_labels, _ = torch.max(action_labels, dim=2) # B x Classes
                
                _, pred_scene_labels = torch.max(scene_logits, dim=1)
                scene_labels, _ = torch.max(scene_labels, dim=1) # B
                num_correct_scenes += torch.sum(pred_scene_labels == scene_labels)

                # Loss
                if phase == 'train':
                    action_loss = F.binary_cross_entropy_with_logits(max_frame_action_logits, action_labels)
                    scene_loss = F.cross_entropy(scene_logits, scene_labels)
                    loss = action_loss + scene_loss
                    writer.add_scalar('Loss/train_action', action_loss, steps)
                    writer.add_scalar('Loss/train_scene', scene_loss, steps)
                    writer.add_scalar('Loss/train', loss, steps)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if steps % 10 == 0:
                        print('Step {}: action_loss = {}, scene_loss = {}, total_loss = {}'.format(steps, action_loss, scene_loss, loss))
                    steps += 1
                
                # metrics for validation
                pred_action = (F.sigmoid(max_frame_action_logits) >= 0.5).float()
                if i == 0:
                    all_action_preds = np.array(pred_action.tolist())
                    all_action_labels = np.array(action_labels.tolist())
                else:
                    all_action_preds = np.append(all_action_preds, pred_action.tolist(), axis=0)
                    all_action_labels = np.append(all_action_labels, action_labels.tolist(), axis=0)

            # Eval after epoch
            all_action_APs = [metrics.average_precision_score(y_true=all_action_labels[:, j], y_score=all_action_preds[:, j]) for j in range(157)]
            scene_acc = float(num_correct_scenes) / len(dataloaders[phase].dataset)
            mAP_action = np.nanmean(all_action_APs)
            if phase == 'train':
                writer.add_scalar('mAP/train_action', mAP_action, epoch)
                writer.add_scalar('Accuracy/train_scene', scene_acc, epoch)
                print('-' * 50)
                print('{}, action_acc: {:.4f}, scene_acc: {:.4f}'.format(phase, mAP_action, scene_acc))
                print('-' * 50)
                save_checkpoint(sife, optimizer, loss, save_dir, epoch, steps)
            else:
                writer.add_scalar('mAP/val_action', mAP_action, epoch)
                writer.add_scalar('Accuracy/val_scene', scene_acc, epoch)
                print('-' * 50)
                print('{}, action_acc: {:.4f}, scene_acc: {:.4f}'.format(phase, mAP_action, scene_acc))
                print('-' * 50)
        
        lr_sched.step() # step after epoch
    
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
        NUM_FEATURES = args.num_features
        NUM_EPOCHS = 200
        SAVE_DIR = './checkpoints-{}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}/'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        with open(SAVE_DIR + 'info.txt', 'w+') as f:
            f.write('LR = {}\nBATCH_SIZE = {}\nSTRIDE = {}\nNUM_SPAN_FRAMES = {}\nEPOCHS = {}'.format(LR, BATCH_SIZE, STRIDE, NUM_SPAN_FRAMES, NUM_EPOCHS))
        
        run(init_lr=LR, root='/vision/group/Charades_RGB/Charades_v1_rgb', 
            num_features=NUM_FEATURES, batch_size=BATCH_SIZE, save_dir=SAVE_DIR,
            stride=STRIDE, num_span_frames=NUM_SPAN_FRAMES, num_epochs=NUM_EPOCHS)
