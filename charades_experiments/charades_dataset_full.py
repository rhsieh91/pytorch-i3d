# Contributor: piergiaj
# Sampling rate modifications by Samuel Kwong

import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

import cv2
from PIL import Image

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_rgb_frames(image_dir, vid, start, num, stride, transforms):
    frames = []
    for i in range(start, start+num, stride):
        try:
            img = pil_loader(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))
            img = transforms(img)
            img = torch.unsqueeze(img, 0)
        except: # duplicate last frame
            img = frames[-1]
        frames.append(img)
    return frames

def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start+num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
        
        w,h = imgx.shape
        if w < 224 or h < 224:
            d = 224.-min(w,h)
            sc = 1+d/min(w,h)
            imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
            imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
             
        imgx = (imgx/255.)*2 - 1
        imgy = (imgy/255.)*2 - 1
        img = np.asarray([imgx, imgy]).transpose([1,2,0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, stride, num_span_frames, num_classes=157):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    for i, vid in enumerate(data.keys()):
        if (i % 500 == 0):
            print('{}/{}'.format(i, len(data.keys())))

        if data[vid]['subset'] != split:
            continue
        
        if not os.path.exists(os.path.join(root, vid)):
            continue
        
        # Sample clips with temporal stride of 4, having input clips spanning total of 125 frames (~5.2 seconds)
        # Follows approach from Long-Term Feature Banks for Detailed Video Understanding (https://arxiv.org/pdf/1812.05038.pdf)
        
        num_avail_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_span_frames=num_span_frames//2
            
        label = np.zeros((num_classes, num_span_frames), np.float32)

        fps = num_avail_frames/data[vid]['duration'] # we use 24fps as provided by Charades website
        for ann in data[vid]['actions']:
            most_recent = None
            for fr in range(0, num_span_frames*stride, stride): # sample every 4 frames
                if fr < num_avail_frames:
                    if fr/fps > ann[1] and fr/fps < ann[2]:
                        label[ann[0], int(fr/stride)] = 1 # binary classification
                        most_recent = 1
                    else:
                        most_recent = 0
                else: # repeat last frame
                    label[ann[0], int(fr/stride)] = most_recent # binary classification
        dataset.append((vid, label, num_span_frames*stride, stride))
    
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms, stride, num_span_frames):
        
        self.data = make_dataset(split_file, split, root, mode, stride, num_span_frames)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf, stride = self.data[index]

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, 1, nf, stride, self.transforms)
        else:
            imgs = load_flow_frames(self.root, vid, 1, nf, stride)

        inputs = torch.cat(imgs)
        inputs = inputs.permute(1, 0, 2, 3)

        return inputs, torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)
