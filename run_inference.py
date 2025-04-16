import argparse
import os
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2
import numpy as np
from matplotlib import pyplot
from configs import config as configllib
from ckpt_manager import CKPT_Manager
from models.MPT import MPT
import random
import numbers

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch Deblur inference on a folder of images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR', help='path to images folder')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--output', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp', 'tiff'], nargs='*', type=str, help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        if isinstance(array, np.ndarray):
            array = np.transpose(array, (2, 0, 1))
            tensor = torch.from_numpy(array)
            return tensor.float()
        elif isinstance(array, tuple):
            tensor = [torch.from_numpy(array[i].transpose(2, 0, 1)).float() for i in range(len(array))]
            return tensor
        else:
            raise

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target=None):
        h, w, _ = inputs.shape
        th, tw = self.size
        if w == tw and h == th:
            if target is not None:
                return inputs, target
            else:
                return inputs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = inputs[y1: y1 + th, x1: x1 + tw]
        if target is not None:
            target = target[y1: y1 + th, x1: x1 + tw]
        if target is not None:
            return inputs, target
        else:
            return inputs

@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()
    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir/'Result'
    else:
        save_path = Path(args.output)
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()

    input_transform = transforms.Compose([
        ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
    ])

    imgs = []
    for ext in args.img_exts:
        test_files = data_dir.files('*.{}'.format(ext))
        for file in test_files:
            imgs.append(file)

    print('{} samples found'.format(len(imgs)))
    # create model
    network_data = torch.load(args.pretrained)
    model = MPT().cuda()

    try:
        model.load_state_dict(network_data)
    except:
        f = lambda x: x.split('module.Network.', 1)[-1] if x.startswith('module.Network.') else x
        network_data = {f(key): value for key, value in network_data.items()}
        model.load_state_dict(network_data)

    model.eval()
    cudnn.benchmark = True

    for img_file in tqdm(imgs):

        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:1024, :1280, :]
        img = input_transform(img)
        input_var = img.unsqueeze(0)

        input_var = input_var.to(device)
        # compute output
        output = model(input_var)

        result_rgb = tensor2rgb(output['result'][0])
        result_deblur = result_rgb.clip(0, 1)
        deblur_save = (result_deblur * 255).round().astype(np.uint8).transpose(1, 2, 0)
        deblur_save = cv2.cvtColor(deblur_save, cv2.COLOR_RGB2BGR)

        # cv2.imshow('deblur', deblur_save)
        # cv2.waitKey()
        cv2.imwrite(save_path/'{}'.format(img_file.basename()), deblur_save)

def tensor2rgb(img_tensor):
    map_np = img_tensor.detach().cpu().numpy()
    _, h, w = map_np.shape

    return map_np


if __name__ == '__main__':
    main()
