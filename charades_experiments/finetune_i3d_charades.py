import os
import sys
import argparse
import datetime
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import videotransforms

from pytorch_i3d import InceptionI3d
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms 
from torch.utils.tensorboard import SummaryWriter
from charades_dataset_full import Charades as Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--bs', type=int, help='batch size')
parser.add_argument('--cs', type=int, help='clip size')
parser.add_argument('--epochs', type=int, help='number of epochs')
args = parser.parse_args()


def train(model, optimizer, dataloaders, num_classes, epochs, 
          save_dir='', use_gpu=False, lr_sched=None):
    # Enable GPU if available
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)
    
    if torch.cuda.device_count() > 1:
        print('Multiple GPUs detected: {}'.format(torch.cuda.device_count()))
        model = nn.DataParallel(model) # wrap model to parallelize multiple GPUs
    model = model.to(device=device) # move model parameters to CPU/GPU
    
    writer = SummaryWriter() # Tensorboard logging
    dataloaders = {'train': train_loader, 'val': test_loader}   
    best_train = -1 # keep track of best val accuracy seen so far
    best_val = -1 # keep track of best val accuracy seen so far
    n_iter = 0

    # Training loop
    for e in range(epochs):    
        print('Epoch {}/{}'.format(e, epochs))
        print('-'*10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
                print('-'*10, 'TRAINING', '-'*10)
            else:
                model.train(False)  # set model to eval mode
                print('-'*10, 'VALIDATION', '-'*10)

            num_correct = 0
            num_samples = 0 

            for data in dataloaders[phase]:
                inputs = data[0] # shape = B x C x T x H x W
                inputs = inputs.to(device=device, dtype=torch.float32) # model expects inputs of float32

                # Forward pass
                if phase == 'train':
                    per_frame_logits = model(inputs)
                else: 
                    with torch.no_grad(): # disable autograd to reduce memory usage for validation phase
                        per_frame_logits = model(inputs)

                # Due to the strides and max-pooling in I3D, it temporally downsamples the video by a factor of 8
                # so we need to upsample (F.interpolate) to get per-frame predictions
                # ALTERNATIVE: Take the average to get per-clip prediction
                per_frame_logits = F.interpolate(per_frame_logits, size=inputs.shape[2], mode='linear') # shape = B x NUM_CLASSES x T

                # Average or max across frames to get a single prediction per clip
                # pred_frame_logits = torch.mean(per_frame_logits, dim=2) # shape = B x NUM_CLASSES
                pred_frame_logits = torch.max(per_frame_logits, dim=2)[0] # shape = B x NUM_CLASSES
                _, pred_class_idx = torch.max(pred_frame_logits, dim=1) # shape = B (values are indices from 0:NUM_CLASSES-1)

                # Ground truth labels
                class_idx = data[1] # shape = B
                class_idx = class_idx.to(device=device)
                num_correct += torch.sum(pred_class_idx == class_idx)
                num_samples += inputs.shape[0]

                # Backward pass only if in 'train' mode
                if phase == 'train':
                    loss = F.cross_entropy(pred_frame_logits, class_idx)
                    writer.add_scalar('Loss/train', loss, n_iter)
                    
                    optimizer.zero_grad()
                    loss.backward() 
                    optimizer.step()

                    if n_iter % 10 == 0:
                        print('{}, loss = {}'.format(phase, loss))
                    n_iter += 1

            # Log train/val accuracy
            accuracy = float(num_correct) / float(num_samples)
            print('num_correct = {}'.format(num_correct))
            print('num_samples = {}'.format(num_samples))
            print('{}, accuracy = {}'.format(phase, accuracy))

            if phase == 'train':
                writer.add_scalar('Accuracy/train', accuracy, e)
                if accuracy > best_train:
                    best_train = accuracy
                    print('BEST TRAINING ACCURACY: {}'.format(best_train))
                    save_checkpoint(model, optimizer, loss, save_dir, e, n_iter)
            else:
                writer.add_scalar('Accuracy/val', accuracy, e)
                if accuracy > best_val:
                    best_val = accuracy
                    print('BEST VALIDATION ACCURACY: {}'.format(best_val))
                    save_checkpoint(model, optimizer, loss, save_dir, e, n_iter)

        if lr_sched is not None:
            lr_sched.step() # decay learning rate

    writer.close()  


def save_checkpoint(model, optimizer, loss, save_dir, epoch, n_iter):
    """Saves checkpoint of model weights during training."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = save_dir + str(epoch).zfill(2) + str(n_iter).zfill(6) + '.pt'
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                },
                save_path)


if __name__ == '__main__':
    print('Starting...')
    now = datetime.datetime.now()
    checkpoints_dirname = './checkpoints-{}-{}-{}-{}-{}-{}/'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    
    if len(sys.argv) < 5:
        parser.print_usage()
        sys.exit(1)

    # Hyperparameters
    USE_GPU = True
    NUM_CLASSES = 157
    ROOT = '/vision/group/Charades_RGB/Charades_v1_rgb'
    LR = args.lr
    BATCH_SIZE = args.bs
    CLIP_SIZE = args.cs
    EPOCHS = args.epochs 
    SAVE_DIR = checkpoints_dirname
    NUM_WORKERS = 2
    SHUFFLE = True
    PIN_MEMORY = True
    
    print('LR =', LR)
    print('BATCH_SIZE =', BATCH_SIZE)
    print('CLIP_SIZE =', CLIP_SIZE)
    print('EPOCHS =', EPOCHS)
    print('SAVE_DIR =', SAVE_DIR)

    # Book-keeping
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    with open(SAVE_DIR + 'info.txt', 'w+') as f:
        f.write('LR = {}\nBATCH_SIZE = {}\nEPOCHS = {}\n'.format(LR, BATCH_SIZE, EPOCHS))

    # Transforms
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                            videotransforms.RandomHorizontalFlip(),
                                            ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # Datasets and Dataloaders
    train_dataset = Dataset(train_split, 'training', root, mode, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=36, pin_memory=True)

    val_dataset = Dataset(train_split, 'testing', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=36, pin_memory=True)    

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    # Load pre-trained I3D model
    i3d = InceptionI3d(400, in_channels=3) # pre-trained model has 400 output classes
    i3d.load_state_dict(torch.load('/vision/u/rhsieh91/pytorch-i3d/models/rgb_imagenet.pt'))
    i3d.replace_logits(NUM_CLASSES) # replace final layer to work with new dataset

    # Set up optimizer and learning rate schedule
    optimizer = optim.Adam(i3d.parameters(), lr=LR) 
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], gamma=0.1) # decay learning rate by gamma at epoch 10 and 20

    # Start training
    train(i3d, optimizer, dataloaders, num_classes=NUM_CLASSES, epochs=EPOCHS, 
          save_dir=SAVE_DIR, use_gpu=USE_GPU, lr_sched=lr_sched)
