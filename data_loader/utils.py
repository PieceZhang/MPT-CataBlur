import os
import numpy as np
import cv2
from pathlib import Path
import glob
from pathlib import Path
import random


def read_frame(config, path, norm_val = None, rotate = None):
    if norm_val == (2**16-1):
        frame = cv2.imread(path, -1)
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
        frame = frame / norm_val
        frame = frame[...,::-1]
    else:
        if config.data == 'BBBC':
            frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            frame -= frame.min()
            frame = frame / frame.max()
        else:
            frame = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            frame = frame / 255.
        if rotate is not None:
            frame = cv2.rotate(frame, rotate)
    return np.expand_dims(frame, axis = 0)


def read_frame_LFDOF(clist, gtlist):
    index = random.randint(0, len(clist)-1)
    c = cv2.cvtColor(cv2.imread(clist[index], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    c = c / 255.
    c = np.expand_dims(c, axis=0)
    gt = cv2.cvtColor(cv2.imread(gtlist[index], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gt = gt / 255.
    gt = np.expand_dims(gt, axis=0)
    return c, gt


def crop_multi(x, hrg, wrg, is_random=False, row_index=0, col_index=1):

    h, w = x[0].shape[row_index], x[0].shape[col_index]

    if (h < hrg) or (w < wrg):  # (h <= hrg) or (w <= wrg)
        raise AssertionError(f"The size of cropping ({hrg, wrg}) should smaller than the original image ({h, w})")

    if is_random:
        if h != hrg:
            h_offset = int(np.random.uniform(0, h - hrg) - 1)
        else:
            h_offset = 0
        if w != wrg:
            w_offset = int(np.random.uniform(0, w - wrg) - 1)
        else:
            w_offset = 0
        results = []
        for data in x:
            results.append(data[int(h_offset):int(hrg + h_offset), int(w_offset):int(wrg + w_offset)])
        return np.asarray(results)
    else:
        # central crop
        h_offset = (h - hrg) / 2
        w_offset = (w - wrg) / 2
        results = []
        for data in x:
            results.append(data[int(h_offset):int(h - h_offset), int(w_offset):int(w - w_offset)])
        return np.asarray(results)

def color_to_gray(img):
    c_linear = 0.2126*img[:, :, 0] + 0.7152*img[:, :, 1] + 0.07228*img[:, :, 2]
    c_linear_temp = c_linear.copy()

    c_linear_temp[np.where(c_linear <= 0.0031308)] = 12.92 * c_linear[np.where(c_linear <= 0.0031308)]
    c_linear_temp[np.where(c_linear > 0.0031308)] = 1.055 * np.power(c_linear[np.where(c_linear > 0.0031308)], 1.0/2.4) - 0.055

    img[:, :, 0] = c_linear_temp
    img[:, :, 1] = c_linear_temp
    img[:, :, 2] = c_linear_temp

    return img

def refine_image(img, val = 16):
    shape = img.shape
    if len(shape) == 4:
        _, h, w, _ = shape[:]
        return img[:, 0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 3:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val, :]
    elif len(shape) == 2:
        h, w = shape[:2]
        return img[0 : h - h % val, 0 : w - w % val]

def refine_image_pt(image, val = 16):
    size = image.size()
    h = size[2]
    w = size[3]
    refine_h = h - h % val
    refine_w = w - w % val

    return image[:, :, :refine_h, :refine_w]

def load_file_list(config, root_path, child_path = None, is_flatten = False, dataname=None):
    if dataname is None:
        dataname = config.data

    if dataname == 'LFDOF':
        folder_paths = os.path.join(root_path, child_path)
        images = []
        if child_path == 'input':
            path = Path(folder_paths)
            for img in sorted(path.rglob('*.png')):
                images.append(str(img))
        elif child_path == 'ground_truth':
            path = Path(os.path.join(root_path, 'input'))
            for img in sorted(path.rglob('*.png')):
                images.append(os.path.join(root_path, 'ground_truth', img.name[:8] + '.png'))
        return np.array([folder_paths]), np.array(images, dtype=object), images.__len__()
    elif dataname == 'BBBC':
        folder_paths = os.path.join(root_path, child_path)
        images = []
        if child_path == 'blur':
            path = Path(folder_paths)
            for img in sorted(path.rglob('*.tif')):
                if config.BBBCw == 'w1w2':
                    images.append(str(img))
                elif img.parts[-1].split('_')[-2] == config.BBBCw:
                    images.append(str(img))
        elif child_path == 'GT':
            path = Path(os.path.join(root_path, child_path))
            for img in sorted(path.rglob('*.tif')):
                if config.BBBCw == 'w1w2':
                    images.append(str(img))  # z stack len = 3
                    images.append(str(img))
                    images.append(str(img))
                elif img.parts[-1].split('_')[-2] == config.BBBCw:
                    images.append(str(img))  # z stack len = 3
                    images.append(str(img))
                    images.append(str(img))
        return np.array([folder_paths]), np.array(images, dtype=object), images.__len__()
    elif dataname == 'cadisv2targeted':
        folder_paths = os.path.join(root_path, child_path)
        images = []
        if child_path == 'blur':
            path = Path(folder_paths)
            for img in sorted(path.rglob('*.png')):
                images.append(str(img))
        elif child_path == 'GT':
            path = Path(os.path.join(root_path, child_path))
            for img in sorted(path.rglob('*.png')):
                images.append(str(img))
                images.append(str(img))
        return np.array([folder_paths]), np.array(images, dtype=object), images.__len__()
    elif dataname == 'mycataractblur':
        if 'LFDOF' in root_path:
            return load_file_list_LFDOF(root_path, child_path)
        else:
            folder_paths = os.path.join(root_path, child_path)
            images = []
            path = Path(folder_paths)
            for img in sorted(path.rglob('*.jpg')):
                images.append(str(img))
            return np.array([folder_paths]), np.array(images, dtype=object), images.__len__()
    elif dataname == 'WNLO':
        if 'LFDOF' in root_path:
            return load_file_list_LFDOF(root_path, child_path)
        else:
            folder_paths = os.path.join(root_path, child_path)
            images = []
            path = Path(folder_paths)
            for img in sorted(path.rglob('*.png')):
                images.append(str(img))
            return np.array([folder_paths]), np.array(images, dtype=object), images.__len__()
    elif dataname == 'cataract101':
        folder_paths = os.path.join(root_path, child_path)
        images = []
        if child_path == 'blur':
            path = Path(folder_paths)
            for img in sorted(path.rglob('*.jpg')):
                if config.w == 'all':
                    images.append(str(img))
                elif img.parts[-1].split('_')[-1][1:2] == config.w:
                    images.append(str(img))
        elif child_path == 'GT':
            path = Path(os.path.join(root_path, child_path))
            for img in sorted(path.rglob('*.jpg')):
                if config.w == 'all':
                    images.append(str(img))
                    images.append(str(img))
                    images.append(str(img))
                else:
                    images.append(str(img))
        return np.array([folder_paths]), np.array(images, dtype=object), images.__len__()
    else:
        folder_paths = []
        filenames_pure = []
        filenames_structured = []
        num_files = 0
        for root, dirnames, filenames in os.walk(root_path):
            # print('Root:', root, ', dirnames:', dirnames, ', filenames', filenames)
            if len(dirnames) != 0:
                if dirnames[0][0] == '@':
                    del(dirnames[0])

            if len(dirnames) == 0:
                if root == '.':
                    continue
                if child_path is not None and child_path != Path(root).name:
                    continue
                folder_paths.append(root)
                filenames_pure = []
                for i in np.arange(len(filenames)):
                    if filenames[i][0] != '.' and filenames[i] != 'Thumbs.db':
                        filenames_pure.append(os.path.join(root, filenames[i]))
                # print('filenames_pure:', filenames_pure)
                filenames_structured.append(np.array(sorted(filenames_pure), dtype='str'))
                num_files += len(filenames_pure)

        folder_paths = np.array(folder_paths)
        filenames_structured = np.array(filenames_structured, dtype=object)

        sort_idx = np.argsort(folder_paths)
        folder_paths = folder_paths[sort_idx]
        filenames_structured = filenames_structured[sort_idx]

        if is_flatten:
            if len(filenames_structured) > 1:
                filenames_structured = np.concatenate(filenames_structured).ravel()
            else:
                filenames_structured = filenames_structured.flatten()

        return folder_paths, filenames_structured, num_files


def load_file_list_LFDOF(root_path, child_path = None):
    folder_paths = os.path.join(root_path, child_path)
    images = []
    if child_path == 'input':
        path = Path(folder_paths)
        for img in sorted(path.rglob('*.png')):
            images.append(str(img))
    elif child_path == 'ground_truth':
        path = Path(os.path.join(root_path, 'input'))
        for img in sorted(path.rglob('*.png')):
            images.append(os.path.join(root_path, 'ground_truth', img.name[:8] + '.png'))
    return np.array([folder_paths]), np.array(images, dtype=object), images.__len__()


def get_base_name(path):
    return os.path.basename(path.split('.')[0])

def get_folder_name(path):
    path = os.path.dirname(path)
    return path.split(os.sep)[-1]

def datalist_validate(config, GT, blur):
    if config.data == 'BBBC':
        for gt, b in zip(GT.tolist(), blur.tolist()):
            gtl = gt.split('_')
            bl = b.split('_')
            assert gtl[2] == bl[2] and gtl[3] == bl[3] and gtl[4] == bl[4], f'Error in dataset list: {gt}, {b}'
    elif config.data == 'LFDOF':
        for gt, b in zip(GT.tolist(), blur.tolist()):
            gtl = os.path.split(gt)[-1]
            bl = os.path.split(b)[-1]
            assert gtl[:8] == bl[:8], f'Error in dataset list: {gt}, {b}'
    elif config.data == 'cataract101':
        for gt, b in zip(GT.tolist(), blur.tolist()):
            gtl = os.path.split(gt)[-1]
            bl = os.path.split(b)[-1]
            assert gtl[:14] == bl[:14], f'Error in dataset list: {gt}, {b}'
            if config.w != 'all':
                assert bl[-5] == config.w, f'Error in dataset list: {gt}, {b}'
    elif config.data == '3dhistech':
        for gt, b in zip(GT.tolist(), blur.tolist()):
            gtl = os.path.split(gt)[-1]
            bl = os.path.split(b)[-1]
            assert gtl == bl, f'Error in dataset list: {gt}, {b}'
    elif config.data == 'cadisv2targeted':
        for gt, b in zip(GT.tolist(), blur.tolist()):
            gtl = gt.split('.')
            bl = b.split('.')
            assert gtl[-2][-19:] in bl[-2], f'Error in dataset list: {gt}, {b}'
    else:
        pass
