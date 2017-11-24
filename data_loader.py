import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
import torchvision.datasets as dset

from folder import ImageFolder
from datafolder import DataFolder, rotate_ndimage, attributes_loader, ndimage_loader, basic_preprocess, flip_ndimage

def get_loader(root, split, batch_size, scale_size, num_workers=2, shuffle=True, load_attributes=None, flips=False, rotate_angle=0, take_log=False, normalize=False, use_channels=None):
    dataset_name = os.path.basename(root)
    image_root = os.path.join(root, 'splits', split)
    print "Loading data from", image_root, "log scale" if take_log else ""

    if dataset_name in ['CelebA']:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.CenterCrop(160),
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    elif load_attributes is None:
        dataset = ImageFolder(root=image_root, transform=transforms.Compose([
            transforms.Scale(scale_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    else:
        # format input images
        transf_list = []
        if use_channels is not None:
            transf_list+=[transforms.Lambda(lambda img: img[...,use_channels])]
        else:
            use_channels = range(5)
        transf_list += [transforms.Lambda(lambda img: basic_preprocess(img,scale_size, log=take_log, normalize=normalize))]
        if flips:
            # horizontal and vertical flips
            transf_list += [transforms.Lambda(lambda img: flip_ndimage(img,0)),            transforms.Lambda(lambda img: flip_ndimage(img,1))]
        if rotate_angle>0:
            transf_list += [transforms.Lambda(lambda img: rotate_ndimage(img,rotate_angle))]
        transf_list += [transforms.Lambda(lambda img: torch.from_numpy(img.copy().transpose((2,0,1))).float())]
        # transf_list += [transforms.Normalize(np.ones(len(use_channels))*0.0, 
        #                                      np.ones(len(use_channels))*1.0)]
        transform = transforms.Compose(transf_list)

        # format image attributes (labels)
        dataset = DataFolder(root=image_root, transform=transform, 
            image_loader=ndimage_loader, 
            target_loader=lambda path: attributes_loader(path, fields=load_attributes))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers))
    data_loader.shape = [int(num) for num in dataset[0][0].size()]

    return data_loader
