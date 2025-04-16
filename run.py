import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import json

import time
import numpy
import os
import sys
import collections
import numpy as np
import gc
import math
import random

from models import create_model
from utils import *
from ckpt_manager import CKPT_Manager
import datetime


class Trainer():
    def __init__(self, config, rank = -1):
        self.rank = rank
        if config.dist:
            self.pg = dist.new_group(range(dist.get_world_size()))

        self.config = config
        self.device = config.device
        if self.rank <= 0: self.summary = SummaryWriter(config.LOG_DIR.log_scalar)

        ## model
        self.model = create_model(config)
        # if self.rank <= 0 and config.is_verbose:
        #     self.model.print()

        ## checkpoint manager
        self.ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.cuda, config.max_ckpt_num, is_descending = True)

        ## training vars
        if not config.evaluate:
            self.states = ['train', 'valid']
        else:
            self.states = ['valid']
        # self.states = ['valid', 'train']
        self.max_epoch = int(math.ceil(config.total_itr / self.model.get_itr_per_epoch('train')))
        self.epoch_range = np.arange(1, self.max_epoch + 1)
        self.err_epoch = {'train': {}, 'valid': {}}
        self.norm = torch.tensor(0).cuda()
        self.lr = 0

        if self.config.resume is not None or self.config.resume_abs is not None:
            if self.rank <= 0: print(toGreen('Resume Trianing...'))
            if self.rank <= 0: print(toRed('\tResuming {}..'.format(self.config.resume if self.config.resume is not None else self.config.resume_abs)))
            resume_state = self.ckpt_manager.resume(self.model.get_network(), self.config.resume, self.config.resume_abs, self.rank)
            if self.config.resume is not None and not self.config.evaluate:
                self.epoch_range = np.arange(resume_state['epoch'] + 1, self.max_epoch + 1)
                self.model.resume_training(resume_state)

        if self.config.ifprogressive:
            self.progressive_epoch = list(map(lambda x: int(x / self.model.get_itr_per_epoch('train')) + 1, self.config.progressive_list))
            print(f"Progressive training: HW and batchsize will change to {self.config.progressive, self.config.progressive_bs} at epoch {self.progressive_epoch}.")

    def train(self):
        torch.backends.cudnn.benchmark = True
        if self.rank <= 0 : print(toYellow('\n\n=========== TRAINING START ============'))
        for epoch in self.epoch_range:
            # if self.rank <= 0 and epoch == 1:
            #     if self.config.resume is None:
            #         self.ckpt_manager.save(self.model.get_network(), self.model.get_training_state(0), 0, score = [1e-8, 1e8])
            is_log = epoch == 1 or epoch % self.config.write_ckpt_every_epoch == 0 or epoch > self.max_epoch - 10
            # is_log = True
            if self.config.resume is not None and epoch == int(self.config.resume) + 1:
                is_log = True

            if self.config.ifprogressive:
                if epoch in self.progressive_epoch:
                    self.config.val_freq = self.config.progressive_val_freq[self.progressive_epoch.index(epoch)]
                    self.config.batch_size = self.config.progressive_bs[self.progressive_epoch.index(epoch)]
                    new_h, new_w = self.config.progressive[self.progressive_epoch.index(epoch)]
                    self.model._set_dataloader(new_h, new_w)
                    print(f"Training HW and batchsize changed to {(new_h, new_w), self.config.batch_size} at epoch {epoch}")

            for state in self.states:
                epoch_time = time.time()
                if state == 'train':
                    self.model.train()
                    self.iteration(epoch, state, is_log)
                elif is_log:
                    if self.config.evaluate:
                        self.model.eval()
                        with torch.no_grad():
                            self.iteration(epoch, state, is_log)
                    else:
                        if isinstance(self.config.val_freq, list):
                            if epoch in self.config.val_freq:
                                self.model.eval()
                                with torch.no_grad():
                                    self.iteration(epoch, state, is_log)
                            else:
                                _ = epoch
                                while _ not in self.config.val_freq and _ < self.config.val_freq[-1]:
                                    _ += 1
                                print(f"Next ckpt will be saved at {_}")
                        else:
                            if epoch % self.config.val_freq == 0:
                                self.model.eval()
                                with torch.no_grad():
                                    self.iteration(epoch, state, is_log)
                    if self.config.ifprogressive:
                        print(f"Progressive training: HW and batchsize will change to {self.config.progressive, self.config.progressive_bs} at epoch {self.progressive_epoch}.")

                if state == 'valid' or state == 'train' : # add "or state == 'train" if you want to save train logs
                    if is_log:
                        if config.dist: dist.all_reduce(self.norm, op=dist.ReduceOp.SUM, group=self.pg, async_op=False)
                        for k, v in self.err_epoch[state].items():
                            if config.dist:  dist.all_reduce(self.err_epoch[state][k], op=dist.ReduceOp.SUM, group=self.pg, async_op=False)
                            self.err_epoch[state][k] = (self.err_epoch[state][k] / self.norm).item()

                            if self.rank <= 0:
                                self.summary.add_scalar('loss/epoch_{}_{}'.format(state, k), self.err_epoch[state][k], epoch)
                                self.summary.add_scalar('loss/itr_{}_{}'.format(state, k), self.err_epoch[state][k], self.model.itr_global['train'])

                        if self.rank <= 0:
                            torch.cuda.synchronize()
                            if state == 'train':
                                print_logs(state.upper() + ' TOTAL', self.config.mode, epoch, self.max_epoch, epoch_time, iter = self.model.itr_global[state], iter_total = self.config.total_itr, errs = self.err_epoch[state], lr = self.lr, is_overwrite = False)
                            else:
                                print_logs(state.upper() + ' TOTAL', self.config.mode, epoch, self.max_epoch, epoch_time, errs = self.err_epoch[state], lr = self.lr, is_overwrite = False)
                                print('\n')

                            if state == 'valid':
                                is_saved = False
                                try:
                                    if math.isnan(self.err_epoch['valid']['psnr']) == False:
                                        self.ckpt_manager.save(self.model.get_network(), self.model.get_training_state(epoch), epoch, score=[self.err_epoch['valid']['psnr']])  # self.err_epoch['valid']['LPIPS']
                                    is_saved = True
                                except Exception as ex:
                                    print(f'Save ckpt failed at {epoch} ({ex})')
                                    is_saved = False

                        self.err_epoch[state] = {}
                        if config.dist:
                            dist.barrier()

            if self.rank <= 0:
                if epoch % self.config.refresh_image_log_every_epoch['train'] == 0:
                    remove_file_end_with(self.config.LOG_DIR.sample, '*.jpg')
                    remove_file_end_with(self.config.LOG_DIR.sample, '*.png')
                if epoch % self.config.refresh_image_log_every_epoch['valid'] == 0:
                    remove_file_end_with(self.config.LOG_DIR.sample_val, '*.jpg')
                    remove_file_end_with(self.config.LOG_DIR.sample_val, '*.png')

            gc.collect()

    def iteration(self, epoch, state, is_log):


        is_train = True if state == 'train' else False
        data_loader = self.model.data_loader_train if is_train else self.model.data_loader_eval
        if config.dist:
            if is_train: self.model.sampler_train.set_epoch(epoch)


        itr = 0
        self.norm = torch.tensor(0).cuda()
        for inputs in data_loader:
            lr = None
            itr_time = time.time()

            self.model.iteration(inputs, epoch, self.max_epoch, is_train)
            itr += 1

            if is_log:
                ## To sum up all errs[key] across the GPU, and to average it by norm
                if isinstance(inputs['gt'], list):
                    bs = inputs['gt'][0].size()[0]
                else:
                    bs = inputs['gt'].size()[0]
                errs = self.model.results['errs']
                norm = self.model.results['norm']
                for k, v in errs.items():
                    v = v * norm
                    if itr == 1:
                        self.err_epoch[state][k] = v
                    else:
                        if k in self.err_epoch[state].keys():
                            self.err_epoch[state][k] += v
                        else:
                            self.err_epoch[state][k] = v
                self.norm = self.norm + norm
                ##

                if self.rank <= 0:
                    if config.save_sample:
                        # saves image patches for logging
                        vis = self.model.results['vis']
                        sample_dir = self.config.LOG_DIR.sample if is_train else self.config.LOG_DIR.sample_val
                        if itr == 1 or self.model.itr_global[state] % config.write_log_every_itr[state] == 0:
                            try:
                                i = 1
                                for key, val in vis.items():
                                    if val.dim() == 5:
                                        for j in range(val.size()[1]):
                                            vutils.save_image(val[:, j, :, :, :], '{}/E{:02}_I{:06}_{:02}_{}_{:03}.jpg'.format(sample_dir, epoch, self.model.itr_global[state], i, key, j), nrow=3, padding = 0, normalize = False)
                                    else:
                                        vutils.save_image(val, '{}/E{:02}_I{:06}_{:02}_{}.jpg'.format(sample_dir, epoch, self.model.itr_global[state], i, key), nrow=3, padding = 0, normalize = False)
                                    i += 1
                            except Exception as ex:
                                print('\n\n\n\nsaving error: ', ex, '\n\n\n\n')

                    self.lr = self.model.results['lr']
                    torch.cuda.synchronize()

                    errs_itr = collections.OrderedDict()
                    for k, v in errs.items():
                        errs_itr[k] = v / norm
                    print_logs(state.upper(), self.config.mode, epoch, self.max_epoch, itr_time, itr * self.model.itr_inc[state], self.model.get_itr_per_epoch(state), errs = errs_itr, lr = self.lr, is_overwrite = itr > 1)

##########################################################
def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

if __name__ == '__main__':
    evaluate = False
    resume_abs = None

    # set if resume
    mode = None
    resume = None  # 'resume epoch'
    #################
    height = 256
    width = 256

    import importlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--is_train', type = str, default = 'True', help = 'is train or not')
    parser.add_argument('--config', type = str, default = 'config_MPT', help = 'config name')
    parser.add_argument('--mode', type = str, default = mode, help = 'mode name')
    parser.add_argument('--project', type = str, default = 'MPT', help = 'project name')
    args, _ = parser.parse_known_args()

    args.is_train = True if args.is_train == 'True' else False
    if args.is_train:
        parser.add_argument('-trainer', '--trainer', type = str, default = 'trainer', help = 'model name')
        parser.add_argument('-net', '--network', type = str, default = 'MPT', help = 'network name')
        parser.add_argument('-data', '--data', type = str, default = '3dhistech', help = 'dataset')
        args, _ = parser.parse_known_args()

        config_lib = importlib.import_module('configs.{}'.format(args.config))
        config = config_lib.get_config(args.project, args.mode, args.config, args.data)
        args.mode = config.mode
        config.is_train = True
        # config.height = height if height is not None else config.height
        # config.width = width if width is not None else config.width

        if evaluate:
            config.evaluate = True
            config.LPIPS = True
            config.SSIM = True
            config.resume_abs = resume_abs

        ## DEFAULT
        parser.add_argument('-r', '--resume', type = str, default = resume, help = 'name of state or ckpt (names are the same)')
        parser.add_argument('-ra', '--resume_abs', type = str, default = config.resume_abs, help = 'absolute path of state or ckpt')
        parser.add_argument('-dl', '--delete_log', action = 'store_true', default = False, help = 'whether to delete log')
        parser.add_argument('-lr', '--lr_init', type = float, default = config.lr_init, help = 'leraning rate')
        parser.add_argument('-b', '--batch_size', type = int, default = config.batch_size, help = 'number of batch')
        parser.add_argument('-th', '--thread_num', type = int, default = config.thread_num, help = 'number of thread')
        parser.add_argument('-dist', '--dist', action = 'store_true', default = config.dist, help = 'whether to distributed pytorch')
        parser.add_argument('-cpu', '--cpu', action = 'store_true', default = config.dist, help = 'whether to use cpu')
        parser.add_argument('-vs', '--is_verbose', action = 'store_true', default = False, help = 'whether to delete log')
        parser.add_argument('-ss', '--save_sample', action = 'store_true', default = False, help = 'whether to save_sample')
        parser.add_argument("--local_rank", type=int)

        ## CUSTOM
        parser.add_argument('-wi', '--weights_init', type = float, default = config.wi, help = 'weights_init')
        parser.add_argument('-proc', '--proc', type = str, default = 'proc', help = 'dummy process name for killing')
        parser.add_argument('-gc', '--gc', type = float, default = config.gc, help = 'gradient clipping')

        args, _ = parser.parse_known_args()

        ## default
        config.trainer = args.trainer
        config.network = args.network
        config.data = args.data

        config.resume = args.resume
        config.resume_abs = args.resume_abs
        config.delete_log = False if config.resume is not None else args.delete_log
        config.lr_init = args.lr_init
        config.batch_size = args.batch_size
        config.thread_num = args.thread_num
        config.dist = args.dist
        config.cuda = not args.cpu
        if config.cuda == True:
            config.device = 'cuda'
        else:
            config.device = 'cpu'
        config.is_verbose = args.is_verbose
        config.save_sample = args.save_sample

        # CUSTOM
        config.wi = args.weights_init
        config.gc = args.gc


        if config.dist:
            init_dist()
            rank = dist.get_rank()
        else:
            rank = -1

        if rank <= 0:
            handle_directory(config, config.delete_log)
            print(toGreen('Laoding Config...'))
            # config_lib.print_config(config)
            config_lib.log_config(config.LOG_DIR.config, config)
            print(toRed('\tProject : {}'.format(config.project)))
            print(toRed('\tMode : {}'.format(config.mode)))
            print(toRed('\tConfig: {}'.format(config.config)))
            print(toRed('\tNetwork: {}'.format(config.network)))
            print(toRed('\tTrainer: {}'.format(config.trainer)))

        if config.dist:
            dist.barrier()

        ## random seed
        seed = config.manual_seed
        if seed is None:
            seed = 0
        if rank <= 0 and config.is_verbose: print('Random seed: {}'.format(seed))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        trainer = Trainer(config, rank)
        if config.dist:
            dist.barrier()
        trainer.train()

    else:
        from eval import *
        from configs.config import get_config, set_eval_path
        from easydict import EasyDict as edict
        print(toGreen('Laoding Config for evaluation'))
        parser.add_argument('-net', '--network', type = str, default = 'MPT', help = 'network name')
        parser.add_argument('-data', '--data', type = str, default = '3dhistech', help = 'dataset')
        args, _ = parser.parse_known_args()

        config_lib = importlib.import_module('configs.{}'.format(args.config))
        config = config_lib.get_config(args.project, args.mode, args.config, args.data)

        config.is_train = False
        ## EVAL
        parser.add_argument('-data_offset', '--data_offset', type = str, default = config.data_offset, help = 'root path of the dataset')
        parser.add_argument('-output_offset', '--output_offset', type = str, default = config.output_offset, help = 'root path of the outputs')
        parser.add_argument('-ckpt_name', '--ckpt_name', type=str, default = None, help='ckpt name')
        parser.add_argument('-dist', '--dist', action = 'store_true', default = False, help = 'whether to use distributed pytorch')
        parser.add_argument('-cpu', '--cpu', action = 'store_true', default = config.dist, help = 'whether to use cpu')
        parser.add_argument('-eval_mode', '--eval_mode', type=str, default = 'quan', help = 'evaluation mode. qual(qualitative)|quan(quantitative)')
        args, _ = parser.parse_known_args()

        config.network = args.network
        config.EVAL.ckpt_name = args.ckpt_name

        config.dist = args.dist
        config.cuda = not args.cpu
        if config.cuda == True:
            config.device = 'cuda'
        else:
            config.device = 'cpu'

        config.EVAL.eval_mode = args.eval_mode
        config.EVAL.data = args.data

        config.data_offset = args.data_offset
        config.EVAL.LOG_DIR.save = os.path.join(args.output_offset)
        config = set_eval_path(config, config.EVAL.data)

        print(toRed('\tProject : {}'.format(config.project)))
        print(toRed('\tMode : {}'.format(config.mode)))
        print(toRed('\tConfig: {}'.format(config.config)))
        print(toRed('\tNetwork: {}'.format(config.network)))
        print(toRed('\tTrainer: {}'.format(config.trainer)))

        eval(config)
