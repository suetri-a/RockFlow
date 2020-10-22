from pathlib import Path
import os, glob, random

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageFilter
from torchvision import transforms, datasets
from torch.utils.data import Dataset

n_bits = 8


def preprocess(x):
    # Follows:
    # https://github.com/tensorflow/tensor2tensor/blob/e48cf23c505565fd63378286d9722a1632f4bef7/tensor2tensor/models/research/glow.py#L78

    x = x * 255  # undo ToTensor scaling to [0,1]

    n_bins = 2**n_bits
    if n_bits < 8:
      x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5

    return x


def postprocess(x):
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2**n_bits
    return torch.clamp(x, 0, 255).byte()


def get_rock_dataset(args):
    '''
    Data loader for rock datasets
    '''

    if len(args.dataset) == 1:
        ds = RockDataset
    else:
        ds = MultiRockDataset

    ds_train = ds(args, train=True)
    ds_test = ds(args, train=False)
    
    image_shape = (args.patch_size, args.patch_size, ds_train.num_modalities)
    num_classes = ds_train.num_rocks

    return image_shape, num_classes, ds_train, ds_test


class RockDataset(Dataset):


    def __init__(self, args, train = True):
        
        # self.length = 10000
        self.dataset_name = args.dataset[0]
        self.data_dir = os.path.join('data', args.dataset[0])
        self.num_rocks = 1
        self.binary_data = args.binary_data

        subfolders = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir,name))]
        if subfolders:
            self.num_modalities = len(subfolders)
            self.modalities = sorted([s.lower() for s in subfolders])
            imgs = sorted(glob.glob(os.path.join(self.data_dir,subfolders[0],'*.tif')))
        else:
            imgs = sorted(glob.glob(os.path.join(self.data_dir, '*.tif')))
            self.num_modalities = 1
            self.modalities = None

        if train:
            self.length = 10000
            imgs = imgs[:int(0.8*len(imgs))]
        else:
            self.length = 1000
            imgs = imgs[int(0.8*len(imgs)):]

        I = Image.open(imgs[0])
        self.x_dim = I.size[0]
        self.y_dim = I.size[1]
        self.patch_size = args.patch_size
        self.imgs = [os.path.basename(fname) for fname in imgs]


    def __len__(self):

        return self.length


    def __getitem__(self, idx):
        
        img_number = np.random.randint(0, high=len(self.imgs))
        x_coord = np.random.randint(0, high=self.x_dim-self.patch_size)
        y_coord = np.random.randint(0, high=self.y_dim-self.patch_size)

        if self.num_modalities > 1:
            seed = np.random.randint(2147483647)
            I_list = []
            for m in self.modalities:
                random.seed(seed)
                I = Image.open(os.path.join(self.data_dir, m, self.imgs[img_number]))
                I_list.append(self.transformation(I.crop((x_coord, y_coord, x_coord+self.patch_size, y_coord+self.patch_size))))
            patch = torch.cat(I_list, dim=0)

        else:
            I = Image.open(os.path.join(self.data_dir, self.imgs[img_number])).convert(mode='L')
            
            if self.binary_data:
                I = I.filter(ImageFilter.BoxBlur(1))

            patch = self.transformation(I.crop((x_coord, y_coord, x_coord+self.patch_size, y_coord+self.patch_size)))
            if self.binary_data: # clamp values away from boundaries if binarized data
                patch = torch.clamp(patch, -0.35, 0.25)

        return patch, torch.ones((1))

    
    def transformation(self, I):
        '''
        Image transformation function

        '''
        xform = transforms.Compose([transforms.ToTensor(), preprocess])
        return xform(I)


class MultiRockDataset(Dataset):
    
    def __init__(self, args, train = True):
        self.dataset_names = args.dataset
        self.num_rocks = len(self.dataset_names)

        if train:
            self.length = 10000
        else:
            self.length = 1000

        self.datasets = []
        for _ in self.dataset_names:
            self.datasets.append(RockDataset(args, train=train))
            args.dataset.pop(0)

        if any(d.num_modalities>1 for d in self.datasets):
            raise Exception('Multiclass models not supported for multimodality datasets.')

        self.num_modalities = self.datasets[0].num_modalities


    def __len__(self):

        return self.length
    

    def __getitem__(self, idx):
        
        d = np.random.randint(0, high=self.num_rocks)
        patch, _ = self.datasets[d].__getitem__(idx)
        
        return patch, d
