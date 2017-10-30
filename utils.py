from __future__ import print_function

import os
import json
import logging
import numpy as np
from datetime import datetime

import torchvision.utils as vutils
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as cm


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path: # loading model from saved checkpoint
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else: # train new model
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%y-%m-%d_%H:%M:%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def save_image_channels(tensor, filename=None, ncol=8, padding=2,
                        channel_names=None, sample_names=None, take_log=None,
                        normalize=False, range=None, scale_each=False):
    '''
    Saves individual channels of image tensor to file.
    '''
    # extract data into numpy array to manipulate
    tensor = tensor.clone().cpu()
    ncol = tensor.size()[0] if ncol is None else ncol
    t = tensor[:ncol,...]
    N, C, W, H = t.size()
    arr = t.numpy()
        
    # if log scale is requested (better visibility)
    if take_log is not None:
        for i in take_log:
            arr[:,i,...] = np.log(arr[:,i,...]+1e-4)
        arr[np.isnan(arr)] = 0
        arr[np.isneginf(arr)] = 0
        
    # scale to better visualize
    for i in np.arange(C):
        arr[:,i,...] = ((arr[:,i,...] - arr[:,i,...].min()) / (arr[:,i,...].max() - arr[:,i,...].min()) * 255).astype(np.uint8)
     
    # apply water mask
    mask = arr[:,3,...]    
    for i in np.arange(3):
        for j in np.arange(N):
            arr[j,i,...][mask[j]<128] = np.nan
        
    arr_new = arr.reshape((N*C,1,W,H), order='C')
    t1 = torch.from_numpy(arr_new)
    
    # generate grid of images using the make_grid utility in torchvision
    grid = vutils.make_grid(t1, nrow=C, padding=padding,
                            normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.permute(1, 2, 0).numpy().transpose()[0]
    
    # generate plot & save to file
    plt.figure(figsize=(16,6))
    plt.imshow(ndarr, cmap=cm.GnBu)
    if channel_names is not None:
        for i,s in enumerate(channel_names):
            plt.annotate(s, 
                         xy=(0, W/2 + i * (H+padding)), 
                         xytext=(0, W/2 + i * (H+padding)),
                         fontsize=12, color="black", weight="bold")
    
    if sample_names is not None:
        for i,s in enumerate(sample_names):
            plt.annotate(s, 
                         xy=(W/2 + i * (H+padding), 0), 
                         xytext=(W/2 + i * (H+padding), 0),
                         fontsize=12, color="blue", weight="bold")
    plt.axis("off")
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')
