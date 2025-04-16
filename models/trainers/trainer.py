import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import numpy as np
import collections
import torch_optimizer as optim
from torch.optim import AdamW
import torch.nn.utils as torch_utils
from torchvision.transforms import transforms

from models.trainers.lion import Lion
from utils import *
from data_loader.utils import *
from models.utils import *
from models.baseModel import baseModel

from data_loader.DP_datasets_lr_aug import datasets

import models.trainers.lr_scheduler as lr_scheduler

from data_loader.data_sampler import DistIterSampler

from ptflops import get_model_complexity_info
import importlib
from shutil import copy2
import os
from models import create_network
from models import loss
from .metrics import cal_metrics
import pyiqa


class Model(baseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.ws = torch.distributed.get_world_size() if config.dist else 1
        self.device = config.device

        ### NETWORKS ###
        ## main network
        if self.rank <= 0 : print(toGreen('Loading Model...'))
        self.network = DeblurNet(config).cuda()
        self.blur = transforms.RandomChoice([transforms.GaussianBlur(3, sigma=10.),
                                               transforms.GaussianBlur(5, sigma=10.),
                                               transforms.GaussianBlur(5, sigma=10.),
                                               transforms.GaussianBlur(7, sigma=10.)],
                                              p=[0.25, 0.25, 0.25, 0.25])

        ## LPIPS network
        if self.is_train:
            if self.config.is_amp:
                self.scaler = torch.cuda.amp.GradScaler()

            if self.rank <= 0: print(toRed('\tinitializing LPIPS'))
            if config.cuda:
                if config.dist:
                    ## download pretrined checkpoint from 0th process
                    if self.rank <= 0:
                        self.LPIPS = pyiqa.archs.lpips_arch.LPIPS(net='alex').to(torch.device('cuda'))
                    dist.barrier()
                    if self.rank > 0:
                        self.LPIPS = pyiqa.archs.lpips_arch.LPIPS(net='alex').to(torch.device('cuda'))
                else:
                    self.LPIPS = pyiqa.archs.lpips_arch.LPIPS(net='alex').to(torch.device('cuda'))
            else:
                self.LPIPS = pyiqa.archs.lpips_arch.LPIPS(net='alex').to(torch.device('cuda'))

            for param in self.LPIPS.parameters():
                param.requires_grad_(False)

        ### INIT for training ###
        if self.is_train:
            self.itr_global = {'train': 0, 'valid': 0}
            self.itr_inc = {'train': 1, 'valid': 1} # iteration increase factor (for video processing, it might be higher than 1)
            self.network.init()
            self._set_optim()
            self._set_lr_scheduler()
            self._set_dataloader()

            if config.is_verbose:
                for name, param in self.network.named_parameters():
                    if self.rank <= 0: print(name, ', ', param.requires_grad)

        ### PROFILE ###
        if self.rank <= 0:
            with torch.no_grad():
                get_model_complexity_info(self.network.Network, (1, self.config.inch, self.config.val_height, self.config.val_width), input_constructor = self.network.input_constructor, as_strings=False,print_per_layer_stat=config.is_verbose)
                Macs,params = get_model_complexity_info(self.network.Network, (1, self.config.inch, self.config.height, self.config.width), input_constructor = self.network.input_constructor, as_strings=False,print_per_layer_stat=config.is_verbose)

        ### DDP ###
        if config.cuda:
            if config.dist:
                if self.rank <= 0: print(toGreen('Building Dist Parallel Model...'))
                self.network = DDP(self.network, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), broadcast_buffers=True, find_unused_parameters=False)
            else:
                self.network = DP(self.network).to(torch.device('cuda'))

            if self.rank <= 0:
                print(toGreen('Computing model complexity...'))
                print(toRed('\t{:<30}  {:<8} B'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 )))
                print(toRed('\t{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2, '\n')))
                if self.is_train:
                    with open(config.LOG_DIR.offset + '/cost.txt', 'w') as f:
                        f.write('{:<30}  {:<8} B\n'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 ))
                        f.write('{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2))
                        f.close()

    def get_itr_per_epoch(self, state):
        if state == 'train':
            return len(self.data_loader_train) * self.itr_inc[state]
        else:
            return len(self.data_loader_eval) * self.itr_inc[state]

    def _set_loss(self, lr = None):
        if self.rank <= 0: print(toGreen('Building Loss...'))
        self.MSE = torch.nn.MSELoss().cuda()
        self.MAE = torch.nn.L1Loss().cuda()
        self.CSE = torch.nn.CrossEntropyLoss(reduction='none').cuda()
        self.MSE_sum = torch.nn.MSELoss(reduction = 'sum').cuda()
        if self.config.CL.CL:
            self.CL = eval(f'loss.{self.config.CL.CLfunc}(self.config).cuda()')

    def _set_optim(self, lr = None):
        if self.rank <= 0: print(toGreen('Building Optim...'))
        self._set_loss()
        lr = self.config.lr_init if lr is None else lr

        if self.config.optimizer == 'Lion':
            self.optimizer = Lion([{'params': self.network.parameters(), 'lr': self.config.lr_init, 'initial_lr': self.config.lr_init}],
                                  weight_decay=self.config.weight_decay, lr=lr, betas=(self.config.beta1, self.config.beta2))
        elif self.config.optimizer == 'AdamW':
            self.optimizer = AdamW([{'params': self.network.parameters(), 'lr': self.config.lr_init, 'initial_lr': self.config.lr_init}],
                                   eps=1e-8, weight_decay=self.config.weight_decay, lr=lr, betas=(self.config.beta1, self.config.beta2))
        else:
            raise

        self.optimizers.append(self.optimizer)

    def _set_lr_scheduler(self):
        if self.config.LRS == 'CA':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.T_max,
                                                               eta_min=self.config.eta_min, last_epoch=-1, verbose=False))
        elif self.config.LRS == 'LD':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LR_decay(
                        optimizer, decay_period = self.config.decay_period,
                        decay_rate = self.config.decay_rate))

    def _set_dataloader(self, h=None, w=None):
        if self.rank <= 0: print(toGreen('Loading Data Loader...'))

        self.sampler_train = None
        self.sampler_eval = None

        self.dataset_train = datasets(self.config, is_train = True)
        self.dataset_eval = datasets(self.config, is_train = False)

        if h is not None and w is not None:  # progressive training
            self.dataset_train.h = h
            self.dataset_train.w = w

        if self.config.dist == True:
            self.sampler_train = DistIterSampler(self.dataset_train, self.ws, self.rank)
            self.sampler_eval = DistIterSampler(self.dataset_eval, self.ws, self.rank, is_train=False)
        else:
            self.sampler_train = None
            self.sampler_eval = None

        if self.is_train:
            self.data_loader_train = self._create_dataloader(self.dataset_train, sampler = self.sampler_train, is_train = True)
            self.data_loader_eval = self._create_dataloader(self.dataset_eval, sampler = self.sampler_eval, is_train = False)

    def _update(self, errs, warmup_itr = -1):
        lr = None
        self.optimizer.zero_grad()
        errs['total'].backward()

        torch_utils.clip_grad_norm_(self.network.parameters(), self.config.gc)
        self.optimizer.step()
        lr = self._update_learning_rate(self.itr_global['train'], warmup_itr)

        return lr

    def _update_amp(self, errs, warmup_itr = -1):
        lr = None
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(errs['total']).backward()

        self.scaler.unscale_(self.optimizer)
        torch_utils.clip_grad_norm_(self.network.parameters(), self.config.gc)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        lr = self._update_learning_rate(self.itr_global['train'], warmup_itr)

        return lr

    ########################### Edit from here for training/testing scheme ###############################
    def _set_results(self, inputs, outs, errs, lr, norm=1):

        # save visuals (inputs)
        if self.config.save_sample:
            self.results['vis'] = collections.OrderedDict()
            if 'dual' not in self.config.mode:
                self.results['vis']['C'] = inputs['c']
            else:
                self.results['vis']['L'] = inputs['l']
                self.results['vis']['R'] = inputs['r']

            self.results['vis']['result'] = outs['result']
            self.results['vis']['GT'] = inputs['gt']

            if 'D' in self.config.mode or 'IFAN' in self.config.mode:
                self.results['vis']['f_R'] = F.interpolate(inputs['r'], scale_factor=1/8, mode='area')
                self.results['vis']['f_R_w'] = outs['f_R_w']
                self.results['vis']['f_R_gt'] = F.interpolate(inputs['l'], scale_factor=1/8, mode='area')
            if 'R' in self.config.mode or 'IFAN' in self.config.mode:
                self.results['vis']['S'] = F.interpolate(inputs['gt'], scale_factor=1/8, mode='area')
                self.results['vis']['SB'] = outs['SB']
                self.results['vis']['B'] = F.interpolate(inputs['c'], scale_factor=1/8, mode='area')

        ## essential ##
        self.results['errs'] = errs
        self.results['norm'] = norm
        self.results['lr'] = lr

    def _get_results(self, C, GT, is_train, inputs):
        if GT is None:
            GT = C
            C = self.blur(GT)

        if self.config.is_amp:
            with torch.cuda.amp.autocast():
                outs = self.network(C)
        else:
            outs = self.network(C)

        ## loss
        if self.config.is_train:
            errs = collections.OrderedDict()
            if 'result' in outs.keys():
                errs['total'] = torch.tensor(0.)
                # deblur loss
                errs['image'] = self.MSE(outs['result'], GT)
                errs['total'] = errs['total'] + errs['image']

            if is_train:
                if self.config.LPIPS:
                    # explodes when you use DDP
                    dist = self.LPIPS.forward(outs['result'] * 2. - 1., GT * 2. - 1.) #imge range [-1, 1]
                    with torch.no_grad():
                        errs['LPIPS'] = torch.mean(dist) # flow log
                    errs['LPIPS_MSE'] = 1e-1 * self.MSE(torch.zeros_like(dist).cuda(), dist)
                    errs['total'] = errs['total'] + errs['LPIPS_MSE']
                if self.config.CL.CL:
                    errs['CLloss'] = self.CL(self.network, C, GT, outs['result'], inputs=inputs)
                    errs['total'] = errs['total'] + errs['CLloss']
            else:
                errs['psnr'] = get_psnr2(outs['result'], GT)
                if self.config.data == 'mycataractblur' or self.config.data == 'WNLO':
                    mt = cal_metrics(outs['result'], choose_metrics=['sd', 'vo'])
                    errs['sd'] = torch.tensor(mt['sd'])
                    errs['vo'] = torch.tensor(mt['vo'])
                if self.config.LPIPS:
                    dist = self.LPIPS.forward(outs['result'] * 2. - 1., GT * 2. - 1.) #imge range [-1, 1]
                    errs['LPIPS'] = torch.mean(dist) # flow log
                if self.config.SSIM:
                    mt = pyiqa.archs.ssim_arch.ssim(outs['result'], GT)[0]
                    errs['ssim'] = torch.tensor(mt)

            return errs, outs
        else:
            return outs

    def iteration(self, inputs, epoch, max_epoch, is_train):
        # init for logging
        state = 'train' if is_train else 'valid'
        self.itr_global[state] += self.itr_inc[state]

        if not isinstance(inputs['c'], list):
            C = inputs['c'].cuda()
        else:
            C = list(map(lambda _: _.cuda(), inputs['c']))

        if not isinstance(inputs['gt'], list):
            if len(inputs['gt'].shape) > 1:
                GT = inputs['gt'].cuda()
            else:
                GT = None  # self supervise
        else:
            GT = list(map(lambda _: _.cuda(), inputs['gt']))

        if 'c_ex' in inputs.keys():
            inputs['c_ex'] = inputs['c_ex'].cuda()
            inputs['gt_ex'] = inputs['gt_ex'].cuda()

        """check"""
        # img = (C[0].detach().cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
        # tar = (GT[0].detach().cpu().numpy().transpose(1, 2, 0)*255).astype(np.uint8)
        # cv2.imshow("img", img)
        # cv2.imshow("tar", tar)
        # cv2.waitKey()
        """"""

        # run network / get results and losses / update network, get learning rate
        errs, outs = self._get_results(C, GT, is_train, inputs)
        if self.config.is_amp:
            lr = self._update_amp(errs, self.config.warmup_itr) if is_train else None
        else:
            lr = self._update(errs, self.config.warmup_itr) if is_train else None


        # set results for the log
        outs_ = collections.OrderedDict()
        for k, v in outs.items():
            try:
                outs_[k] = v.clone().detach()
            except AttributeError:
                continue
        errs_ = collections.OrderedDict()
        for k, v in errs.items():
            errs_[k] = v.clone().detach()
        norm, _, _, _ = outs_['result'].shape
        self._set_results(inputs, outs_, errs_, lr, norm)

class DeblurNet(nn.Module):
    def __init__(self, config):
        super(DeblurNet, self).__init__()
        self.rank = torch.distributed.get_rank() if config.dist else -1

        self.config = config
        self.device = config.device
        self.Network = create_network(config)
        if self.rank <= 0: print(toRed('\tinitializing deblurring network'))

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = self.config.wi)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def init(self):
        self.Network.apply(self.weights_init)

    def input_constructor(self, res):
        b, c, h, w = res[:]

        C = torch.FloatTensor(np.random.randn(b, c, h, w)).cuda()

        return {'x':C}

    #####################################################
    def forward(self, C, R=None, L=None, GT=None, is_train=False):
        is_train = is_train or self.config.save_sample and self.config.is_train

        outs = self.Network(C)

        return outs

