import torch
import torchvision.utils as vutils
import torch.nn.functional as F

import os
import sys
import datetime
import gc
from pathlib import Path

import numpy as np
import cv2
import math
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity
import collections

from models.MPT import MPT
from utils import *
from data_loader.utils import load_file_list, read_frame, refine_image
from ckpt_manager import CKPT_Manager

from models import create_model
import models.archs.LPIPS as LPIPS


def ssim(img1, img2, PIXEL_MAX = 1.0):
    return structural_similarity(img1, img2, data_range=PIXEL_MAX, channel_axis=2)


def psnr(img1, img2, PIXEL_MAX = 1.0):
    mse_ = np.mean( (img1 - img2) ** 2 )
    return 10 * math.log10(PIXEL_MAX / mse_)


def init(config, mode = 'deblur'):
    date = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')

    ckpt_name = config.EVAL.ckpt_name

    network_data = torch.load(ckpt_name)
    model = MPT().cuda()
    try:
        model.load_state_dict(network_data)
    except:
        f = lambda x: x.split('Network.', 1)[-1] if x.startswith('Network.') else x
        network_data = {f(key): value for key, value in network_data.items()}
        model.load_state_dict(network_data)

    save_path_root = config.EVAL.LOG_DIR.save

    save_path_root_deblur = './output'
    save_path_root_deblur_score = save_path_root_deblur
    Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)
    # torch.save(network.state_dict(), os.path.join(save_path_root_deblur, ckpt_name))
    save_path_root_deblur = os.path.join(save_path_root_deblur, config.EVAL.data, date)

    gt_file_path_list = None

    _, input_c_file_path_list, _ = load_file_list(config, config.EVAL.c_path, config.EVAL.input_path, is_flatten=True)
    if config.EVAL.gt_path is not None:
        _, gt_file_path_list, _ = load_file_list(config, config.EVAL.c_path, config.EVAL.gt_path, is_flatten=True)

    return model, save_path_root_deblur, save_path_root_deblur_score, ckpt_name, input_c_file_path_list, gt_file_path_list

def eval_quan_qual(config):
    mode = 'quanti_quali'
    network, save_path_root_deblur, save_path_root_deblur_score, ckpt_name,\
    input_c_file_path_list, gt_file_path_list = init(config, mode)

    ##
    time_norm = 0
    total_itr_time = 0

    PSNR = 0
    SSIM  = 0
    LPIPs = 0
    PSNR_mean = 0.
    SSIM_mean = 0.
    LPIPS_mean = 0.
    LPIPSN = LPIPS.PerceptualLoss(model='net-lin',net='alex', use_gpu=config.cuda).to(config.device)
    ##

    print(toYellow('\n\n=========== EVALUATION START ============'))
    for i, frame_name in enumerate(input_c_file_path_list):
        refine_val = config.refine_val

        rotate = None

        # Read image
        C = refine_image(read_frame(config, input_c_file_path_list[i], config.norm_val, rotate), refine_val)
        C = torch.FloatTensor(C.transpose(0, 3, 1, 2).copy()).to(config.device)

        if gt_file_path_list is not None:
            GT = refine_image(read_frame(config, gt_file_path_list[i], config.norm_val, rotate), refine_val)
            GT = torch.FloatTensor(GT.transpose(0, 3, 1, 2).copy()).to(config.device)

        # Run network
        time_norm = time_norm + 1
        with torch.no_grad():
            torch.cuda.synchronize()
            init_time = time.time()
            if config.is_amp:
                with torch.cuda.amp.autocast():
                    out = network(C)
            else:
                out = network(C)
            torch.cuda.synchronize()
            itr_time = time.time() - init_time
            total_itr_time = total_itr_time + itr_time

            output = out['result']

        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0) #[0, 1]

        ## QRun networkuantitative evaluation
        if gt_file_path_list is not None:
            GT_cpu = GT.cpu().numpy()[0].transpose(1, 2, 0) #[0, 1]
            PSNR = psnr(output_cpu, GT_cpu)
            SSIM = ssim(output_cpu, GT_cpu)
            with torch.no_grad():
                LPIPs = LPIPSN.forward(output * 2. - 1., GT * 2. - 1.).item() #[-1, 1]

        ## Qualitative evaluation
        frame_name = os.path.basename(frame_name)
        frame_name, _ = os.path.splitext(frame_name)

        ## If you want to save the deblurred image, uncomment the following lines
        # for iformat in ['jpg']:
        #     Path(os.path.join(save_path_root_deblur, 'input', iformat)).mkdir(parents=True, exist_ok=True)
        #     Path(os.path.join(save_path_root_deblur, 'output', iformat)).mkdir(parents=True, exist_ok=True)
        #
        #     save_file_path_deblur_input = os.path.join(save_path_root_deblur, 'input', iformat, '{:02d}.{}'.format(i+1, iformat))
        #     save_file_path_deblur = os.path.join(save_path_root_deblur, 'output', iformat, '{:02d}.{}'.format(i+1, iformat))
        #
        #     vutils.save_image(C, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False)
        #     vutils.save_image(output, '{}'.format(save_file_path_deblur), nrow=1, padding = 0, normalize = False)

        # Log
        print('[EVAL {} on {}][{:02}/{}] {} PSNR: {:.5f}, SSIM: {:.5f}, LPIPS: {:.5f} ({:.5f}sec)'.format(config.mode, config.EVAL.data, i + 1, len(input_c_file_path_list), frame_name, PSNR, SSIM, LPIPs, itr_time))
        with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'w' if i == 0 else 'a') as file:
            file.write('[EVAL {}][{:02}/{}] {} PSNR: {:.5f}, SSIM: {:.5f}, LPIPS: {:.5f} ({:.5f}sec)\n'.format(config.mode, i + 1, len(input_c_file_path_list), frame_name, PSNR, SSIM, LPIPs, itr_time))
            file.close()

        PSNR_mean += PSNR
        SSIM_mean += SSIM
        LPIPS_mean += LPIPs

        gc.collect()

    total_itr_time = total_itr_time / time_norm

    PSNR_mean = PSNR_mean / len(input_c_file_path_list)
    SSIM_mean = SSIM_mean / len(input_c_file_path_list)
    LPIPS_mean = LPIPS_mean / len(input_c_file_path_list)

    sys.stdout.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} LPIPS: {:.5f} ({:.5f}sec)'.format(ckpt_name, config.EVAL.data, PSNR_mean, SSIM_mean, LPIPS_mean, total_itr_time))
    with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'a') as file:
        file.write('\n[TOTAL {}] PSNR: {:.5f} SSIM: {:.5f} LPIPS: {:.5f} ({:.5f}sec)'.format(ckpt_name, PSNR_mean, SSIM_mean, LPIPS_mean, total_itr_time))
        file.close()

def eval(config):
    eval_quan_qual(config)

