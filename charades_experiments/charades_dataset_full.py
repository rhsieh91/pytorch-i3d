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


def load_rgb_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start+num):
        try:
            img = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
        except: # duplicate last frame
            img = frames[-1]
        w,h,c = img.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
        img = (img/255.)*2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)

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


def make_dataset(split_file, split, root, mode, num_classes=157):
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
        
        num_span_frames = 125 * 4 # upper boundary to grab from original clip (duplicate last frame if clip is too short)
        num_avail_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_span_frames = num_span_frames//2
            
        label = np.zeros((num_classes,num_span_frames), np.float32)
        last_label = np.zeros((num_classes,num_span_frames), np.float32)

        fps = num_avail_frames/data[vid]['duration'] # we use 24fps as provided by Charades website
        for ann in data[vid]['actions']:
            for fr in range(0,num_span_frames,4): # sample every 4 frames
                if fr < num_avail_frames:
                    if fr/fps > ann[1] and fr/fps < ann[2]:
                        label[ann[0], fr] = 1 # binary classification
                else: # repeat last frame
                    label[ann[0], fr] = 1 # binary classification
        dataset.append((vid, label, num_span_frames))
    
    return dataset


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
            return 0, 0, vid

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, 1, nf)
        else:
            imgs = load_flow_frames(self.root, vid, 1, nf)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)
