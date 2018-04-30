# adapted after https://github.com/yxlao/pytorch-reverse-gan/blob/master/dcgan_reverse.py

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable

def reverse_z(netG, x, nz=128, z_distribution="normal", cuda=False, clip='disabled', lr=0.001, niter=1000, loss_type='L2', apply_transform=None):
    """
    Estimate z_approx given G and G(z).
    Args:
        netG: nn.Module, generator network.
        g_z: Variable, G(z).
        opt: argparse.Namespace, network and training options.
        z: Variable, the ground truth z, ref only here, not used in recovery.
        clip: Although clip could come from of `opt.clip`, here we keep it
              to be more explicit.
    Returns:
        Variable, z_approx, the estimated z value.
    """
    # sanity check
    assert clip in ['disabled', 'standard', 'stochastic']

    xt = torch.from_numpy(x).float()
    if apply_transform is not None:
            xt = apply_transform(xt)
    xv = Variable(xt)
    xv = xv.detach()

    # loss metrics
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    # init tensor
    if z_distribution == 'uniform':
        z_approx = torch.FloatTensor(1, nz, 1, 1).uniform_(-1, 1)
    elif z_distribution == 'normal':
        z_approx = torch.FloatTensor(1, nz, 1, 1).normal_(0, 1)
    else:
        raise ValueError()

    # transfer to gpu
    if cuda:
        mse_loss.cuda()
        l1_loss.cuda()
        z_approx = z_approx.cuda()
        xv = xv.cuda()

    # convert to variable
    z_approx = Variable(z_approx)
    z_approx.requires_grad = True

    # optimizer
    optimizer_approx = optim.Adam([z_approx], lr=lr, betas=(0.5, 0.999))

    # train
    loss_g_z_hist = []
    loss_min = 1000
    z_approx_min = z_approx
    iter_disp = niter / 10
    for i in range(niter):
        g_z_approx = netG(z_approx)
        mse_g_z = mse_loss(g_z_approx, xv)
        l1_g_z = l1_loss(g_z_approx, xv)
        if loss_type == 'L2':
            loss_g_z = mse_g_z
        elif loss_type == 'L1':
            loss_g_z = l1_g_z
        else:
            loss_g_z = mse_g_z + l1_g_z

        if i % iter_disp == 0:
            print("[Iter {}/{}] loss_g_z: {}"
                  .format(i, niter, mse_g_z.data[0]))
        loss_g_z_hist += [loss_g_z.data[0]]
        if mse_g_z.data[0] < loss_min:
            loss_min = loss_g_z.data[0]
            z_approx_min = z_approx

        # bprop
        optimizer_approx.zero_grad()
        loss_g_z.backward()
        optimizer_approx.step()

        # clipping
        if clip == 'standard':
            z_approx.data[z_approx.data > 1] = 1
            z_approx.data[z_approx.data < -1] = -1
        if clip == 'stochastic':
            z_approx.data[z_approx.data > 1] = random.uniform(-1, 1)
            z_approx.data[z_approx.data < -1] = random.uniform(-1, 1)

    return z_approx_min, loss_g_z_hist