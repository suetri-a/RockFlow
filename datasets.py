from pathlib import Path
import os, glob

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
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


def get_CIFAR10(augment, dataroot, download=True):
    image_shape = (32, 32, 3)
    num_classes = 10

    test_transform = transforms.Compose([transforms.ToTensor(), preprocess])

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])

    train_transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / 'data' / 'CIFAR10'
    train_dataset = datasets.CIFAR10(path, train=True,
                                     transform=train_transform,
                                     target_transform=one_hot_encode,
                                     download=download)

    test_dataset = datasets.CIFAR10(path, train=False,
                                    transform=test_transform,
                                    target_transform=one_hot_encode,
                                    download=download)

    return image_shape, num_classes, train_dataset, test_dataset


def get_SVHN(augment, dataroot, download=True):
    image_shape = (32, 32, 3)
    num_classes = 10

    if augment:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1))]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), preprocess])
    transform = transforms.Compose(transformations)

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), num_classes)

    path = Path(dataroot) / 'data' / 'SVHN'
    train_dataset = datasets.SVHN(path, split='train',
                                  transform=transform,
                                  target_transform=one_hot_encode,
                                  download=download)

    test_dataset = datasets.SVHN(path, split='test',
                                 transform=transform,
                                 target_transform=one_hot_encode,
                                 download=download)

    return image_shape, num_classes, train_dataset, test_dataset


def get_rock_dataset(rock_name, patch_size):
    '''
    Data loader for rock datasets
    '''
    ds_train = RockDataset(rock_name, patch_size=patch_size, train=True)
    ds_test = RockDataset(rock_name, patch_size=patch_size, train=False)

    image_shape = (patch_size, patch_size, 1)
    num_classes = 1

    return image_shape, num_classes, ds_train, ds_test


class RockDataset(Dataset):


    def __init__(self, dataset_name, patch_size=64, train = True):
        
        # self.length = 10000
        self.dataset_name = dataset_name
        self.data_dir = os.path.join('data', dataset_name)
        
        imgs = glob.glob(os.path.join(self.data_dir, '*.tif'))

        if train:
            self.length = 10000
            self.img_num_min = 0
            self.img_num_max = int(0.8*len(imgs))
        else:
            self.length = 1000
            self.img_num_min = int(0.8*len(imgs))+1
            self.img_num_max = len(imgs)

        I = Image.open(imgs[0])
        self.x_dim = I.size[0]
        self.y_dim = I.size[1]
        self.patch_size = patch_size


    def __len__(self):

        return self.length


    def __getitem__(self, idx):
        
        img_number = np.random.randint(self.img_num_min, high=self.img_num_max)

        x_coord = np.random.randint(0, high=self.x_dim-self.patch_size)
        y_coord = np.random.randint(0, high=self.y_dim-self.patch_size)

        I = Image.open(os.path.join(self.data_dir, self.dataset_name + str(img_number).zfill(4) + '.tif'))
        patch = I.crop((x_coord, y_coord, x_coord+self.patch_size, y_coord+self.patch_size))
        
        return self.transformation(patch), torch.ones((1))

    
    def transformation(self, I):
        '''
        Image transformation function

        '''
        xform= transforms.Compose([transforms.ToTensor(), preprocess])
        return xform(I)
