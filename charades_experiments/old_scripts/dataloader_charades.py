# This code is based off of: https://github.com/gsig/charades-algorithms/blob/master/pytorch/datasets/charadesrgb.py
# Modifications made to Charades.__getitem__() to return video clips instead of single images

""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image
import numpy as np
from glob import glob
import csv
# import cPickle as pickle
import pickle
import os
import pdb

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']

def parse_charades_csv(filename):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'class': x, 'start': float(
                    y), 'end': float(z)} for x, y, z in actions]
            labels[vid] = actions
    return labels


def cls2int(x):
    return int(x[1:])


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def cache(cachefile):
    """ Creates a decorator that caches the result to cachefile """
    def cachedecorator(fn):
        def newf(*args, **kwargs):
            print('cachefile {}'.format(cachefile))
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as f:
                    print("Loading cached result from '%s'" % cachefile)
                    return pickle.load(f)
            res = fn(*args, **kwargs)
            with open(cachefile, 'wb') as f:
                print("Saving result to cache '%s'" % cachefile)
                pickle.dump(res, f)
            return res
        return newf
    return cachedecorator


class Charades(data.Dataset):
    def __init__(self, root, split, labelpath, cachedir, clip_size, is_val, transform=None, target_transform=None):
        self.num_classes = 157
        self.transform = transform
        self.target_transform = target_transform
        self.labels = parse_charades_csv(labelpath)
        self.root = root
        cachename = '{}/{}_{}.pkl'.format(cachedir,
                                          self.__class__.__name__, split)
        self.data = cache(cachename)(self.prepare)(root, self.labels, split)
        self.clip_size = clip_size
        self.is_val = is_val

    def prepare(self, path, labels, split):
        FPS, GAP, testGAP = 24, 4, 25
        datadir = path
        # image_paths, targets, ids = [], [], []
        clips = []

        for i, (vid, label) in enumerate(labels.items()):
            iddir = datadir + '/' + vid
            lines = glob(iddir+'/*.jpg')
            n = len(lines)
            if i % 100 == 0:
                print("{} {}".format(i, iddir))
            if n == 0:
                continue
            if split == 'val_video':
                target = torch.IntTensor(157).zero_()
                for x in label:
                    target[cls2int(x['class'])] = 1
                spacing = np.linspace(0, n-1, testGAP)
                for loc in spacing:
                    impath = '{}/{}-{:06d}.jpg'.format(
                        iddir, vid, int(np.floor(loc))+1)
                    image_paths.append(impath)
                    targets.append(target)
                    ids.append(vid)
            else:
                for x in label:
                    image_paths, targets, ids = [], [], []
                    for ii in range(0, n-1, GAP):
                        if x['start'] < ii/float(FPS) < x['end']:
                            impath = '{}/{}-{:06d}.jpg'.format(
                                iddir, vid, ii+1)
                            image_paths.append(impath)
                            targets.append(cls2int(x['class']))
                            ids.append(vid)
                    clips.append({'image_paths': image_paths, 'targets': targets, 'ids': ids})

        # return {'image_paths': image_paths, 'targets': targets, 'ids': ids}
        return clips
 

    def get_frame_names(self, img_paths):
        frame_names = img_paths
        num_frames = len(frame_names)
        num_frames_necessary = self.clip_size

        # pick frames
        offset = 0
        if num_frames_necessary > num_frames:
            # Pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * \
                (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)
        frame_names = frame_names[offset:offset+num_frames_necessary]

        return frame_names


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_paths = self.data[index]['image_paths']
        frame_names = self.get_frame_names(img_paths)

        clip = []
        for frame in frame_names:
            img = default_loader(frame)
            img = self.transform(img)
            clip.append(torch.unsqueeze(img, 0))

        # format data to torch
        data = torch.cat(clip)
        data = data.permute(1, 0, 2, 3)

        action = self.data[index]['targets'][0] # take the first one since they are all the same
        clip_id = self.data[index]['ids'][0] # take the first one since they are all the same

        return (data, action, clip_id)

    def __len__(self):
        # return len(self.data['image_paths'])
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get(args):
    """ Entry point. Call this function to get all Charades dataloaders """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_file = args.train_file
    val_file = args.val_file
    train_dataset = Charades(
        args.data, 'train', train_file, args.cache,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(args.inputsize),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # missing PCA lighting jitter
            normalize,
        ]))
    val_dataset = Charades(
        args.data, 'val', val_file, args.cache,
        transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            normalize,
        ]))
    valvideo_dataset = Charades(
        args.data, 'val_video', val_file, args.cache,
        transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            normalize,
        ]))
    return train_dataset, val_dataset, valvideo_dataset


if __name__ == '__main__':
    transform = transforms.Compose([transforms.CenterCrop(84),
                                    transforms.ToTensor(),
                                    ])
    import nonechucks as nc
    dataset = Charades(root='/vision/group/Charades_RGB/Charades_v1_rgb',
                       split='train',
                       labelpath='/vision/group/Charades/annotations/Charades_v1_train.csv',
                       cachedir='/vision2/u/rhsieh91/pytorch-i3d/charades_experiments/charades_cache',
                       clip_size=16,
                       is_val=False,
                       transform=transform)
    dataset = nc.SafeDataset(dataset)

    # train_loader_1 = torch.utils.data.DataLoader(dataset,
    #                                            batch_size=8,
    #                                            shuffle=True,
    #                                            num_workers=0,
    #                                            pin_memory=True)
    train_loader_2 = nc.SafeDataLoader(dataset,
                                     batch_size=8,
                                     shuffle=True,
                                     num_workers=0,
                                     pin_memory=True)
    # pdb.set_trace()
    
    for i, a in enumerate(train_loader):
        print(a[0].shape) # data
        print(a[1]) # action
        print(a[2]) # clip id
        if i == 10:
             break
