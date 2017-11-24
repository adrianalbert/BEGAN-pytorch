from __future__ import print_function

import os
import StringIO
import scipy.misc
import numpy as np
from glob import glob
from tqdm import trange
from itertools import chain
from collections import deque

import torch
from torch import nn
import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable

from utils import save_image_channels
from models import *
from data_loader import get_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def next(loader):
    return loader.next()

class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader

        self.num_gpu = config.num_gpu
        self.dataset = config.dataset

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.build_model()

        if self.num_gpu == 1:
            self.G.cuda()
            self.D.cuda()
        elif self.num_gpu > 1:
            self.D = nn.DataParallel(self.D.cuda(),device_ids=range(self.num_gpu))
            self.G = nn.DataParallel(self.G.cuda(),device_ids=range(self.num_gpu))

        if self.load_path:
            self.load_model()

        self.use_tensorboard = config.use_tensorboard
        if self.use_tensorboard:
            import tensorflow as tf
            self.summary_writer = tf.summary.FileWriter(self.model_dir)

            def inject_summary(summary_writer, tag, value, step):
                if hasattr(value, '__len__'):
                    for idx, img in enumerate(value):
                        summary = tf.Summary()
                        sio = StringIO.StringIO()
                        scipy.misc.toimage(img).save(sio, format="png")
                        image_summary = tf.Summary.Image(encoded_image_string=sio.getvalue())
                        summary.value.add(tag="{}/{}".format(tag, idx), image=image_summary)
                        summary_writer.add_summary(summary, global_step=step)
                else:
                    summary= tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                    summary_writer.add_summary(summary, global_step=step)

            self.inject_summary = inject_summary

    def build_model(self):
        channel, height, width = self.data_loader.shape
        assert height == width, "height and width should be same"

        repeat_num = int(np.log2(height)) - 2
        self.D = DiscriminatorCNN(
                channel, self.z_num, repeat_num, self.conv_hidden_num, self.num_gpu)
        self.G = GeneratorCNN(
                self.z_num, self.D.conv2_input_dim, channel, repeat_num, self.conv_hidden_num, self.num_gpu)

        self.G.apply(weights_init)
        self.D.apply(weights_init)

    def train(self):
        # l1 = L1Loss()
        # l1 = nn.L1Loss()
        # note that the above options rely on the backend torch.nn implementation of the L1 loss. That implementation does not allow gradients to flow through the second argument (the target) of the loss function. However this redefinition does. See also: https://discuss.pytorch.org/t/nn-criterions-dont-compute-the-gradient-w-r-t-targets/3693
        def l1(input, target):
            return torch.mean(torch.sum(torch.abs(input - target), 1))

        rand_seed = 4321
        if self.num_gpu > 0:
            torch.cuda.manual_seed_all(rand_seed)
            torch.cuda.manual_seed(rand_seed)
        else:
            torch.manual_seed_all(rand_seed)
            torch.manual_seed(rand_seed)

        z_D = torch.FloatTensor(self.batch_size, self.z_num)
        z_G = torch.FloatTensor(self.batch_size, self.z_num)
        z_fixed = Variable(torch.FloatTensor(self.batch_size, self.z_num).normal_(0, 1), volatile=True)

        if self.num_gpu > 0:
            z_fixed = z_fixed.cuda()
            z_G = Variable(z_G.cuda(), requires_grad=True)
            z_D = Variable(z_D.cuda(), requires_grad=True)
        else:
            z_G = Variable(z_G, requires_grad=True)
            z_D = Variable(z_D, requires_grad=True)

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        def get_optimizer(lr):
            return optimizer(self.G.parameters(), lr=lr, betas=(self.beta1, self.beta2)), \
                   optimizer(self.D.parameters(), lr=lr, betas=(self.beta1, self.beta2))

        g_optim, d_optim = get_optimizer(self.lr)

        data_loader = iter(self.data_loader)
        sample = next(data_loader)
        if len(sample) == 3:
            x_fixed,attr_fixed,names_fixed = sample
        else:
            x_fixed, _ = sample
        x_fixed = self._get_variable(x_fixed)

        self.take_log_display = not self.config.take_log #self.config.use_channels if self.config.take_log is None or not self.config.take_log else False
        if self.config.save_image_channels:
            save_image_channels(x_fixed.data, 
                filename='{}/x_fixed.png'.format(self.model_dir), ncol=10, padding=10, take_log=self.take_log_display, scale_each=True, normalize=self.config.normalize, channel_names=self.config.src_names)
        else:
            vutils.save_image(x_fixed.data, 
                filename='{}/x_fixed.png'.format(self.model_dir), nrow=2, padding=10)

        k_t = 0.0
        k_t_hist = []
        k_t_thresh = 50
        alpha = 0.5
        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            try:
                sample = next(data_loader)
                if len(sample) == 3:
                    x,x_attr,x_name = sample
                else:
                    x, _ = sample
            except StopIteration:
                data_loader = iter(self.data_loader)
                sample = next(data_loader)
                if len(sample) == 3:
                    x,x_attr,x_name = sample
                else:
                    x, _ = sample

            # ground truth (real) data
            x = self._get_variable(x)
            batch_size = x.size(0)

            # Update Discriminator D

            z_D.data.normal_(0, 1)
            sample_z_D = self.G(z_D)
            AE_x = self.D(x)
            AE_G_d = self.D(sample_z_D.detach())

            d_loss_real = l1(AE_x, x)
            d_loss_fake = l1(AE_G_d, sample_z_D.detach())

            d_loss = d_loss_real - k_t * d_loss_fake
            self.D.zero_grad()
            self.G.zero_grad()
            d_loss.backward()
            d_optim.step()

            # Update Generator G

            z_G.data.normal_(0, 1)
            sample_z_G = self.G(z_G)
            AE_G_g = self.D(sample_z_G)
            g_loss = l1(AE_G_g, sample_z_G) 

            # loss = d_loss + g_loss
            self.D.zero_grad()
            self.G.zero_grad()
            g_loss.backward()
            g_optim.step()

            g_d_balance = (self.gamma*d_loss_real-d_loss_fake).data[0]
            k_t += self.lambda_k * g_d_balance
            k_t = alpha * k_t + (1-alpha) * np.mean(k_t_hist) if len(k_t_hist)>k_t_thresh else k_t
            k_t = max(min(1, k_t), 0)
            k_t_hist.append(k_t)
            k_t_hist = k_t_hist[-k_t_thresh:]

            measure = d_loss_real.data[0] + abs(g_d_balance)
            measure_history.append(measure)

            z_G_grad_norm = np.mean(np.abs((z_G.grad.data.cpu().numpy())))
    
            if step % self.log_step == 0:
                print("[{}/{}] Loss_D: {:.4f} L_x: {:.4f} Loss_G: {:.4f} "
                      "measure: {:.4f}, k_t: {:.4f}, lr: {:.7f}". \
                      format(step, self.max_step, d_loss.data[0], d_loss_real.data[0],
                             g_loss.data[0], measure, k_t, self.lr))
                x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

                if self.use_tensorboard:
                    info = {
                        'loss/loss_D': d_loss.data[0],
                        'loss/loss_D_fake': d_loss_fake.data[0],
                        'loss/L_x': d_loss_real.data[0],
                        'loss/Loss_G': g_loss.data[0],
                        'grad/z_G_norm': z_G_grad_norm,
                        'misc/measure': measure,
                        'misc/k_t': k_t,
                        'misc/lr': self.lr,
                        'misc/balance': g_d_balance
                    }
                    for tag, value in info.items():
                        self.inject_summary(self.summary_writer, tag, value, step)

                    # self.inject_summary(
                    #         self.summary_writer, "AE_G", AE_G_g.data[:,:3,...].cpu().numpy(), step)
                    # self.inject_summary(
                    #         self.summary_writer, "AE_x", AE_x.data[:,:3,...].cpu().numpy(), step)
                    # self.inject_summary(
                    #         self.summary_writer, "z_G", sample_z_G.data[:,:3,...].cpu().numpy(), step)

                    self.summary_writer.flush()

            if step % self.save_step == self.save_step - 1:
                self.save_model(step)

            if step % self.lr_update_step == self.lr_update_step - 1:
                cur_measure = np.mean(measure_history)
                if cur_measure > prev_measure * 0.9999:
                    self.lr *= 0.5
                    g_optim, d_optim = get_optimizer(self.lr)
                prev_measure = cur_measure

    def generate(self, inputs, path, idx=None):
        path = '{}/{}_G.png'.format(path, idx)
        x = self.G(inputs)
        if self.config.save_image_channels:
            save_image_channels(x.data, filename=path, ncol=10, padding=10, normalize=self.config.normalize, take_log=self.take_log_display, scale_each=True, channel_names=self.config.src_names)
        else:
            vutils.save_image(x.data, filename=path, nrow=2, padding=10)
        print("[*] Generated samples saved: {}".format(path))
        return x

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        x_path = '{}/{}_D.png'.format(path, idx)
        x = self.D(inputs)  
        if self.config.save_image_channels:
            save_image_channels(x.data, filename=x_path, ncol=10, padding=10, normalize=self.config.normalize, take_log=self.take_log_display, scale_each=True, channel_names=self.config.src_names)
        else:
            vutils.save_image(x.data, filename=x_path, nrow=2, padding=10)

        print("[*] Direct AE samples saved: {}".format(x_path))

        if x_fake is not None:
            x_fake_path = '{}/{}_D_fake.png'.format(path, idx)
            x = self.D(x_fake)
            if self.config.save_image_channels:
                save_image_channels(x.data, filename=x_fake_path, ncol=10, padding=10, normalize=self.config.normalize, take_log=self.take_log_display, scale_each=True, channel_names=self.config.src_names)
            else:
                vutils.save_image(x.data, filename=x_fake_path, nrow=2, padding=10)

            print("[*] Synth. AE samples saved: {}".format(x_fake_path))

    def test(self):
        data_loader = iter(self.data_loader)
        x_fixed = self._get_variable(next(data_loader))
        if self.config.save_image_channels:
            save_image_channels(x_fixed.data, filename='{}/x_fixed_test.png'.format(self.model_dir), ncol=10, padding=10, normalize=self.config.normalize,take_log=take_log, scale_each=True, channel_names=self.config.src_names)
        else:
            vutils.save_image(x_fixed.data, 
                filename='{}/x_fixed_test.png'.format(self.model_dir), nrow=2, padding=10)

        self.autoencode(x_fixed, self.model_dir, idx="test", x_fake=None)

    def save_model(self, step):
        print("[*] Save models to {}...".format(self.model_dir))

        torch.save(self.G.state_dict(), '{}/G_{}.pth'.format(self.model_dir, step+1))
        torch.save(self.D.state_dict(), '{}/D_{}.pth'.format(self.model_dir, step+1))

    def load_model(self):
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'G_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return
        idxes = [int(os.path.basename(path).split('.')[0].split('_')[-1]) for path in paths]
        self.start_step = max(idxes)

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else: 
            map_location = None

        G_filename = '{}/G_{}.pth'.format(self.load_path, self.start_step)
        self.G.load_state_dict(
            torch.load(G_filename, map_location=map_location))
        print("[*] G network loaded: {}".format(G_filename))

        D_filename = '{}/D_{}.pth'.format(self.load_path, self.start_step)
        self.D.load_state_dict(
            torch.load(D_filename, map_location=map_location))
        print("[*] D network loaded: {}".format(D_filename))

    def _get_variable(self, inputs, require_grads=True):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out
