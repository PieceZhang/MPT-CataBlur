import datetime

from easydict import EasyDict as edict
import json
import os
import collections
import numpy as np

def get_config(project = '', mode = '', config_ = '', data=''):
    ## GLOBAL
    config = edict()
    config.project = project
    config.mode = mode
    config.config = config_
    config.data = data
    config.is_train = False
    config.thread_num = 8
    config.resume = None  # 'resume epoch' DO NOT CHANGE
    config.resume_abs = None  # 'resume abs name' DO NOT CHANGE
    config.manual_seed = 0
    config.is_verbose = False
    config.save_sample = False
    config.is_amp = True

    config.evaluate = False

    config.cuda = True
    if config.cuda == True:
        config.device = 'cuda'
    else:
        config.device = 'cpu'
    config.dist = False

    ##################################### TRAIN #####################################
    config.trainer = ''
    config.network = ''

    config.in_bit = 8
    config.norm_val = (2**config.in_bit - 1)

    config.batch_size = 16
    if config.dist:
        config.batch_size = int(config.batch_size / 2)  # if use 2 GPUs
    config.batch_size_test = 1

    config.val_freq = 1
    # config.val_freq = [4, 12, 24, 28, 32, 36, 38, 39, 40]
    # config.val_freq = [5, 10, 15, 20, 25, 50, 75, 100, 120, 140, 160, 180, *[_ for _ in range(200, 300, 5)], *[_ for _ in range(300, 320, 2)], *[_ for _ in range(320, 400)]]
    # config.val_freq = [3, 6, 9, 15, 20, 50, 75, 100, *[_ for _ in range(100, 120, 4)], *[_ for _ in range(120, 150, 2)]]
    # config.val_freq = [*[_ for _ in range(0, 2000, 50)], *[_ for _ in range(2000, 6000, 15)]]

    # optimizer
    config.optimizer = 'AdamW'
    if config.optimizer == 'Lion':
        # https://github.com/google/automl/tree/master/lion
        config.lr_init = 2e-5  # 3-10x smaller than that of adamW
        config.weight_decay = 0.1
        config.beta1 = 0.95
        config.beta2 = 0.98
    elif config.optimizer == 'AdamW':
        config.lr_init = 1e-4
        config.weight_decay = 0.01
        config.beta1 = 0.9
        config.beta2 = 0.99
    else:
        raise

    config.LRS = 'CA'  # LD / CA
    config.eta_min = config.lr_init * 0.2
    config.warmup_itr = -1

    config.netparams = {'scale_factor': [2, 4, 8], 'auxoutloss_weight': 2e-5, 'auxloss_weight': 2e-5}

    config.SSIM = False
    # LPIPS loss
    config.LPIPS = False
    # CL loss
    config.CL = edict()
    config.CL.CL = False
    config.CL.CLfunc = 'FRACLexMB'
    config.CL.weight = 5e-4
    config.CL.extradata = False
    if config.CL.extradata and config.CL.CL:
        config.CL.data_offset = '../Datasets'

        config.CL.data = 'LFDOF'
        config.CL.c_path = os.path.join(config.CL.data_offset, 'LFDOF_Dataset/train_data')
        config.CL.input_path = 'input'
        config.CL.gt_path = 'ground_truth'

        # config.CL.data = 'mycataractblur'
        # config.CL.c_path = os.path.join(config.CL.data_offset, 'CataBlur')
        # config.CL.input_path = 'blur'
        # config.CL.gt_path = 'blur'

        # config.CL.data = 'WNLO'
        # config.CL.c_path = os.path.join(config.CL.data_offset, 'WNLO')
        # config.CL.input_path = 'blur'
        # config.CL.gt_path = 'blur'

    # data dir
    if config.mode == '' or config.mode is None:
        config.mode = config.data + '_' + datetime.datetime.now().strftime('%m-%d')  # %Y-%m-%d_%H:%M:%S
        if config.CL.CL:
            config.mode = f'{config.CL.CLfunc}_' + config.mode
    print(f'Use {config.data} dataset.')
    if config.data == 'mycataractblur':
        config.inch = 3
        config.data_offset = '../Datasets'
        config.c_path = os.path.join(config.data_offset, 'CataBlur')
        config.input_path = 'Frames'
        config.gt_path = 'Frames'
        config.VAL = edict()
        config.VAL.c_path = os.path.join(config.data_offset, 'CataBlur')
        config.VAL.input_path = 'Frames'
        config.VAL.gt_path = 'Frames'
        config.height = 256
        config.width = 256
        config.val_height = 640  # 720
        config.val_width = 1280  # 1280
        config.total_itr = 120000
        config.T_max = config.total_itr
    elif config.data == 'WNLO':
        config.inch = 3
        config.data_offset = '../Datasets'
        config.c_path = os.path.join(config.data_offset, 'WNLO')
        config.input_path = 'blur'
        config.gt_path = 'blur'
        config.VAL = edict()
        config.VAL.c_path = os.path.join(config.data_offset, 'WNLO')
        config.VAL.input_path = 'blur'
        config.VAL.gt_path = 'blur'
        config.height = 256
        config.width = 256
        config.val_height = 256  # 720
        config.val_width = 256  # 1280
        config.total_itr = 100000
        config.T_max = config.total_itr
    elif config.data == 'cadisv2targeted':
        config.inch = 3
        config.data_offset = '../Datasets'
        config.c_path = os.path.join(config.data_offset, 'CaDISv2_targeteddeblur/train')
        config.input_path = 'blur'
        config.gt_path = 'GT'
        config.VAL = edict()
        config.VAL.c_path = os.path.join(config.data_offset, 'CaDISv2_targeteddeblur/test')
        config.VAL.input_path = config.input_path
        config.VAL.gt_path = config.gt_path
        config.height = 256
        config.width = 256
        config.val_height = 512  # 540
        config.val_width = 896  # 960
        config.total_itr = 100000
        config.T_max = config.total_itr
    elif config.data == '3dhistech':
        config.inch = 3
        config.data_offset = '../Datasets'
        config.c_path = os.path.join(config.data_offset, '3DHistech/train')
        config.input_path = 'blur'
        config.gt_path = 'GT'
        config.VAL = edict()
        config.VAL.c_path = os.path.join(config.data_offset, '3DHistech/val')
        config.VAL.input_path = config.input_path
        config.VAL.gt_path = config.gt_path
        config.height = 128
        config.width = 128
        config.val_height = 256  # 256
        config.val_width = 256  # 256
        config.total_itr = 250000
        config.T_max = config.total_itr
    elif config.data == 'BBBC':
        config.inch = 1
        config.BBBCw = 'w2'  # w1 or w2
        """"""
        config.mode += config.BBBCw
        """"""
        config.data_offset = '../Datasets'
        config.c_path = os.path.join(config.data_offset, 'BBBC006_Dataset/train')
        config.input_path = 'blur'
        config.gt_path = 'GT'
        config.VAL = edict()
        config.VAL.c_path = os.path.join(config.data_offset, 'BBBC006_Dataset/test')
        config.VAL.input_path = config.input_path
        config.VAL.gt_path = config.gt_path
        config.height = 256
        config.width = 256
        config.val_height = 512  # 520
        config.val_width = 640  # 696
        config.total_itr = 80000
        config.T_max = config.total_itr
    elif config.data == 'LFDOF':
        config.inch = 3
        config.data_offset = '../Datasets'
        config.c_path = os.path.join(config.data_offset, 'LFDOF_Dataset/train_data')
        config.input_path = 'input'
        config.gt_path = 'ground_truth'
        config.VAL = edict()
        config.VAL.c_path = os.path.join(config.data_offset, 'LFDOF_Dataset/test_data')
        config.VAL.input_path = config.input_path
        config.VAL.gt_path = config.gt_path
        config.height = 256
        config.width = 256
        config.val_height = 640  # 540
        config.val_width = 896  # 960
        config.total_itr = 200000
        config.T_max = config.total_itr
    elif config.data == 'DPDD':
        config.inch = 3
        config.data_offset = '../Datasets'
        config.c_path = os.path.join(config.data_offset, 'DPDD/train_c')
        config.input_path = 'source'
        config.gt_path = 'target'
        config.VAL = edict()
        config.VAL.c_path = os.path.join(config.data_offset, 'DPDD/val_c')
        config.VAL.input_path = config.input_path
        config.VAL.gt_path = config.gt_path
        config.height = 256
        config.width = 256
        config.val_height = 1024  # 1120
        config.val_width = 1536  # 1680
        config.total_itr = 120000
        config.T_max = config.total_itr
    else:
        raise

    config.ifprogressive = False

    # logs
    # config.save_freq = 2000
    config.max_ckpt_num = 5
    config.write_ckpt_every_epoch = 1
    config.refresh_image_log_every_epoch = {'train':20, 'valid':20}
    config.write_log_every_itr = {'train':200, 'valid': 100}

    # log dirs
    config.LOG_DIR = edict()
    log_offset = './logs'
    log_offset = os.path.join(log_offset, config.project)
    log_offset = os.path.join(log_offset, '{}'.format(config.mode))
    config.LOG_DIR.offset = log_offset
    config.LOG_DIR.ckpt = os.path.join(config.LOG_DIR.offset, 'checkpoint', 'train', 'epoch')
    config.LOG_DIR.ckpt_ckpt = os.path.join(config.LOG_DIR.offset, 'checkpoint', 'train', 'epoch', 'ckpt')
    config.LOG_DIR.ckpt_state = os.path.join(config.LOG_DIR.offset, 'checkpoint', 'train', 'epoch', 'state')
    config.LOG_DIR.log_scalar = os.path.join(config.LOG_DIR.offset, 'log', 'train', 'scalar')
    config.LOG_DIR.log_image = os.path.join(config.LOG_DIR.offset, 'log', 'train', 'image', 'train')
    config.LOG_DIR.sample = os.path.join(config.LOG_DIR.offset, 'sample', 'train')
    config.LOG_DIR.sample_val = os.path.join(config.LOG_DIR.offset, 'sample', 'valid')
    config.LOG_DIR.config = os.path.join(config.LOG_DIR.offset, 'config')

    ##################################### EVAL ######################################
    config.EVAL = edict()
    config.EVAL.eval_mode = 'quan'
    config.EVAL.data = 'DPDD' # DPDD/PixelDP/RealDOF/CUHK

    config.EVAL.load_ckpt_by_score = True
    config.EVAL.ckpt_name = None
    config.EVAL.ckpt_epoch = None
    config.EVAL.ckpt_abs_name = None
    config.EVAL.low_res = False
    config.EVAL.ckpt_load_path = None

    # data dir
    config.EVAL.c_path = None
    config.EVAL.l_path = None
    config.EVAL.r_path = None

    config.EVAL.input_path = None
    config.EVAL.gt_path = None

    # log dir
    config.EVAL.LOG_DIR = edict()
    config.output_offset = os.path.join(config.LOG_DIR.offset, 'result')
    config.EVAL.LOG_DIR.save = config.output_offset

    return config

def set_eval_path(config, data):
    if data == 'DPDD':
        config.EVAL.c_path = os.path.join(config.data_offset, 'DPDD/test_c')
        config.EVAL.input_path = 'blur'
        config.EVAL.gt_path = 'GT'
    elif data == 'LFDOF':
        config.EVAL.c_path = os.path.join(config.data_offset, 'LFDOF_Dataset/test_data')
        config.EVAL.input_path = 'input'
        config.EVAL.gt_path = 'ground_truth'
    elif data == 'BBBC':
        config.EVAL.c_path = os.path.join(config.data_offset, 'BBBC006_Dataset/test')
        config.EVAL.input_path = 'blur'
        config.EVAL.gt_path = 'GT'
    elif data == '3dhistech':
        config.EVAL.c_path = os.path.join(config.data_offset, '3DHistech/test')
        config.EVAL.input_path = 'blur'
        config.EVAL.gt_path = 'GT'
    elif data == 'cadisv2targeted':
        config.EVAL.c_path = os.path.join(config.data_offset, 'CaDISv2_targeteddeblur/test')
        config.EVAL.input_path = 'blur'
        config.EVAL.gt_path = 'GT'
    elif data == 'random':
        config.EVAL.c_path = os.path.join(config.data_offset, 'random')
        config.EVAL.input_path = 'source'
        config.EVAL.gt_path = 'target'
    elif data == 'WNLO':
        config.EVAL.c_path = os.path.join(config.data_offset, 'WNLO')
    elif data == 'mycataractblur':
        config.EVAL.c_path = os.path.join(config.data_offset, 'CataBlur')

    return config

def log_config(path, cfg):
    with open(path + '/config.txt', 'w') as f:
        f.write(json.dumps(cfg, indent=4))
        f.close()

def print_config(cfg):
    print(json.dumps(cfg, indent=4))

