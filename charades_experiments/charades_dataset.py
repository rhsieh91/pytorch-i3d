# Contributor: Samuel Kwong

import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate
import numpy as np
import json
import csv
import pickle
import os
import os.path
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_scene_maps():
    with open('./data/annotations/charades_train_scene_map.pkl', 'rb') as f1:
        train_scene_map = pickle.load(f1)
    with open('./data/annotations/charades_test_scene_map.pkl', 'rb') as f2:
        test_scene_map = pickle.load(f2)
    return (train_scene_map, test_scene_map)

def load_sample(split_file, split, root, vid, stride, num_span_frames, transforms, num_actions, scene_maps, is_sife):
    num_avail_frames = len(os.listdir(os.path.join(root, vid)))
    num_frames_needed = num_span_frames * stride
    
    upper_bound = num_avail_frames - num_frames_needed
    if split == 'training':
        offset = 1 if upper_bound <= 0 else np.random.randint(1, upper_bound+1) # temporal augmentation
    else:
        offset = 1

    # frames
    frames = []
    for i in range(offset, offset+num_frames_needed, stride):
        if i < num_avail_frames:
            img = pil_loader(os.path.join(root, vid, vid+'-'+str(i).zfill(6)+'.jpg'))
            img = transforms(img)
            img = torch.unsqueeze(img, 0)
        else: # duplicate last frame
            img = frames[-1]
        frames.append(img)

    # actions label
    actions_label = np.zeros((num_actions, num_span_frames), np.float32)
    with open(split_file, 'r') as f:
        data = json.load(f)
        fps = num_avail_frames/data[vid]['duration'] # we use 24fps as provided by Charades website
        for ann in data[vid]['actions']:
            most_recent = None
            for i,fr in enumerate(range(offset, offset+num_frames_needed, stride)):
                if fr < num_avail_frames:
                    if fr/fps > ann[1] and fr/fps < ann[2]:
                        actions_label[ann[0], i] = 1 # binary classification
                        most_recent = 1
                    else:
                        most_recent = 0
                else: # repeat last frame
                    actions_label[ann[0], i] = most_recent # binary classification
        
    # scene label (if needed)
    scene_label = None
    if is_sife:
        scene_label = np.zeros(num_span_frames, np.int64) # scene label goes from [0, num_class-1]
        train_scene_map, test_scene_map = scene_maps
        if split == 'training':
            scene_label.fill(train_scene_map[vid])
        else:
            scene_label.fill(test_scene_map[vid])
    
    return frames, actions_label, scene_label

def get_vid_names(split_file, split, root):
    with open(split_file, 'r') as f:
        data = json.load(f)
        vid_names = []
        for vid in data.keys():
            if data[vid]['subset'] != split:
                continue
            if not os.path.exists(os.path.join(root, vid)):
                continue
            vid_names.append(vid)
    return vid_names


class Charades(data_utl.Dataset):

    def __init__(self, split_file, split, root, transforms=None, stride=4, num_span_frames=32, is_sife=False):
        
        self.vid_names = get_vid_names(split_file, split, root)
        self.split_file = split_file
        self.split = split
        self.root = root
        self.transforms = transforms
        self.stride = stride
        self.num_span_frames = num_span_frames
        self.num_actions = 157
        self.scene_maps = load_scene_maps() if is_sife else None
        self.is_sife = is_sife

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            list of frames, actions label tensor, scene label tensor (if is_sife=True), and vid name
        """
        vid = self.vid_names[index]
        imgs, actions_label, scene_label = load_sample(self.split_file, self.split, self.root, vid, self.stride, self.num_span_frames, self.transforms,
                                                       self.num_actions, self.scene_maps, self.is_sife)

        inputs = torch.cat(imgs)
        inputs = inputs.permute(1, 0, 2, 3)

        if self.is_sife:
            return inputs, torch.from_numpy(actions_label), torch.from_numpy(scene_label), vid
        else:
            return inputs, torch.from_numpy(actions_label), vid

    def __len__(self):
        return len(self.vid_names)
