import os
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torch.nn.functional as F

from data_loader.utils import *

class datasets(data.Dataset):
    def __init__(self, config, is_train):
        super(datasets, self).__init__()
        self.config = config
        self.is_train = is_train
        self.h = config.height
        self.w = config.width
        self.val_h = config.val_height
        self.val_w = config.val_width
        self.norm_val = config.norm_val
        self.max_sig = config.max_sig

        if is_train:
            self.c_folder_path_list, self.c_file_path_list, _ = load_file_list(config, config.c_path, config.input_path, is_flatten = True)
            self.gt_folder_path_list, self.gt_file_path_list, _ = load_file_list(config, config.c_path, config.gt_path, is_flatten = True)
            self.is_augment = True
            if config.CL.extradata and config.CL.CL:
                if config.CL.data == 'LFDOF':
                    self.CL_c_folder_path_list, self.CL_c_file_path_list, _ = load_file_list_LFDOF(config.CL.c_path, config.CL.input_path)
                    self.CL_gt_folder_path_list, self.CL_gt_file_path_list, _ = load_file_list_LFDOF(config.CL.c_path, config.CL.gt_path)
                elif config.CL.data == 'mycataractblur' or config.CL.data == 'WNLO':
                    self.CL_c_folder_path_list, self.CL_c_file_path_list, _ = load_file_list(config, config.CL.c_path, config.CL.input_path, is_flatten=True)
        else:
            self.c_folder_path_list, self.c_file_path_list, _ = load_file_list(config, config.VAL.c_path, config.VAL.input_path, is_flatten = True)
            self.gt_folder_path_list, self.gt_file_path_list, _ = load_file_list(config, config.VAL.c_path, config.VAL.gt_path, is_flatten = True)
            self.is_augment = False

        if self.gt_file_path_list is not None:
            datalist_validate(config, self.gt_file_path_list, self.c_file_path_list)
        self.len = len(self.c_file_path_list)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len

        c_frame = read_frame(self.config, self.c_file_path_list[index], self.norm_val)

        if self.is_train and self.config.CL.extradata and self.config.CL.CL:
            if self.config.CL.data == 'LFDOF':
                c_frame_ex, gt_frame_ex = read_frame_LFDOF(self.CL_c_file_path_list, self.CL_gt_file_path_list)
            elif self.config.CL.data == 'mycataractblur' or self.config.CL.data == 'WNLO':
                c_frame_ex, gt_frame_ex = read_frame_LFDOF(self.CL_c_file_path_list, self.CL_c_file_path_list)
        else:
            c_frame_ex, gt_frame_ex = None, None

        if self.config.inch == 1:
            c_frame = c_frame[:, :, :, None]

        if self.is_augment:
            # Noise
            if random.uniform(0, 1) <= 0.05:
                row,col,ch = c_frame[0].shape
                mean = 0.0
                sigma = random.uniform(0.001, self.max_sig)
                gauss = np.random.normal(mean,sigma,(row,col,ch))
                gauss = gauss.reshape(row,col,ch)
                c_frame = np.expand_dims(np.clip(c_frame[0] + gauss, 0.0, 1.0), axis = 0)

            # Grayscale
            if self.config.data != 'BBBC':
                if random.uniform(0, 1) <= 0.3:
                    c_frame = np.expand_dims(color_to_gray(c_frame[0]), axis = 0)

            # Scaling
            if random.uniform(0, 1) <= 0.5:
                row,col,ch = c_frame[0].shape
                scale = random.uniform(max(min(max(self.h/row + 1e-2, self.w/col + 1e-2), 1.0), 0.7), 1.0)
                c_frame = np.expand_dims(cv2.resize(c_frame[0], dsize=(int(col*scale), int(row*scale)), interpolation=cv2.INTER_AREA), axis = 0)
                if self.config.inch == 1:
                    c_frame = c_frame[:, :, :, None]

        if c_frame.shape[1] < self.h or c_frame.shape[2] < self.w:
            c_frame = cv2.resize(c_frame[0], (self.h, self.w))[None, :]

        cropped_frames = c_frame

        if self.is_train:
            cropped_frames = crop_multi(cropped_frames, self.h, self.w, is_random = True)
            if c_frame_ex is not None:
                cropped_frames_ex = crop_multi(np.concatenate([c_frame_ex, gt_frame_ex], axis = 3), self.h, self.w, is_random=True)
        else:
            cropped_frames = crop_multi(cropped_frames, self.val_h, self.val_w, is_random = False)

        c_patches = cropped_frames[:, :, :, :self.config.inch]
        shape = c_patches.shape
        h = shape[1]
        w = shape[2]
        c_patches = c_patches.reshape((h, w, -1, self.config.inch))
        c_patches = np.transpose(c_patches, (2, 0, 1, 3))

        gt_patches = None

        if c_frame_ex is not None:
            c_patches_ex = cropped_frames_ex[:, :, :, :3]
            shape = c_patches_ex.shape
            h = shape[1]
            w = shape[2]
            c_patches_ex = c_patches_ex.reshape((h, w, -1, 3))
            c_patches_ex = np.transpose(c_patches_ex, (2, 0, 1, 3))
            gt_patches_ex = cropped_frames_ex[:, :, :, 3:]
            gt_patches_ex = gt_patches_ex.reshape((h, w, -1, 3))
            gt_patches_ex = np.transpose(gt_patches_ex, (2, 0, 1, 3))

        if c_frame_ex is None:
            return {'c': torch.FloatTensor(np.transpose(c_patches, (0, 3, 1, 2)))[0],
                    'gt': torch.FloatTensor(np.transpose(gt_patches, (0, 3, 1, 2)))[0] if gt_patches is not None else False}
        else:
            return {'c': torch.FloatTensor(np.transpose(c_patches, (0, 3, 1, 2)))[0],
                    'gt': torch.FloatTensor(np.transpose(gt_patches, (0, 3, 1, 2)))[0] if gt_patches is not None else False,
                    'c_ex': torch.FloatTensor(np.transpose(c_patches_ex, (0, 3, 1, 2)))[0],
                    'gt_ex': torch.FloatTensor(np.transpose(gt_patches_ex, (0, 3, 1, 2)))[0]}


