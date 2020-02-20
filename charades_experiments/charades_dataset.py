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

def load_sample(split, root, vid, stride, num_span_frames, transforms, num_actions, scene_maps, is_sife):
    num_avail_frames = len(os.listdir(os.path.join(root, vid)))
    num_frames_needed = num_span_frames * stride
    
    upper_bound = num_avail_frames - num_frames_needed
    if split == 'training':
        offset = 0 if upper_bound <= 0 else np.random.randit(0, upper_bound) # temporal augmentation
    else:
        offset = 0

    # frames
    frames = []
    for i in range(offset, offset+num_span_frames, stride):
        if i < num_avail_frames:
            img = pil_loader(os.path.join(root, vid, vid+'-'+str(i).zfill(6)+'.jpg'))
            img = transforms(img)
            img = torch.unsqueeze(img, 0)
        else: # duplicate last frame
            img = frames[-1]
        frames.append(img)

    # actions label
    actions_label = np.zeros((NUM_ACTIONS, num_span_frames), np.float32)
    fps = num_avail_frames/data[vid]['duration'] # we use 24fps as provided by Charades website
    for ann in data[vid]['actions']:
        most_recent = None
        for fr in range(offset, offset+num_frames_needed, stride):
            if fr < num_avail_frames:
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    actions_label[ann[0], int(fr/stride)] = 1 # binary classification
                    most_recent = 1
                else:
                    most_recent = 0
            else: # repeat last frame
                actions_label[ann[0], int(fr/stride)] = most_recent # binary classification
        
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
    with open(split_file, 'r') as f1:
        data = json.load(f1)
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
        imgs, actions_label, scene_label = load_sample(self.split, self.root, vid, self.stride, self.num_span_frames, self.transforms,
                                                       self.num_actions, self.scene_maps, self.is_sife)

        inputs = torch.cat(imgs)
        inputs = inputs.permute(1, 0, 2, 3)

        if self.is_sife:
            return inputs, torch.from_numpy(actions_label), torch.from_numpy(scene_label), vid
        else:
            return inputs, torch.from_numpy(actions_label), vid

    def __len__(self):
        return len(self.vid_names)


if __name__ == '__main__':
    from torchvision import transforms
    import pickle

    split_file = 'data/annotations/charades.json'
    root = ''
    stride = 4
    num_span_frames = 32
    batch_size = 2
    train_scene_map_pkl = './data/annotations/charades_train_scene_map.pkl'
    test_scene_map_pkl = './data/annotations/charades_test_scene_map.pkl'

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
        train_dataset = Charades(split_file, train_scene_map_pkl, test_scene_map_pkl, 'training', root, mode, train_transforms, stride, num_span_frames)
        pickle_out = open(train_path, 'wb')
        pickle.dump(train_dataset, pickle_out)
        pickle_out.close()
    print('Got train dataset.')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)

    for data in train_dataloader:
        inputs, actions, scenes, vid = data
        print('input shape = {}'.format(inputs.shape))
        print('actions shape = {}'.format(actions.shape))
        print('scenes shape = {}'.format(scenes.shape))
        print('scenes = {}'.format(scenes))
        print('vid = {}'.format(vid))

