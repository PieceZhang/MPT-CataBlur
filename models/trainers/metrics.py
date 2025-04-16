import numpy as np
import cv2
import skimage
import skimage.metrics
import math
from sklearn import metrics

from models.trainers.psnr_ssim import calculate_ssim


class Metrics():
    def __init__(self):

        pass

    def brenner(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 2):
            for y in range(0, shape[1]):
                out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
        return out

    def Laplacian(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(img, cv2.CV_64F).var()

    def SMD(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = np.shape(img)
        out = 0
        for x in range(1, shape[0] - 1):
            for y in range(0, shape[1]):
                out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
                out += math.fabs(int(img[x, y] - int(img[x + 1, y])))
        return out

    def SMD2(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1] - 1):
                out += math.fabs(int(img[x, y]) - int(img[x + 1, y])) * math.fabs(int(img[x, y] - int(img[x, y + 1])))
        return out

    def energy(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = np.shape(img)
        out = 0
        for x in range(0, shape[0] - 1):
            for y in range(0, shape[1] - 1):
                out += ((int(img[x + 1, y]) - int(img[x, y])) ** 2) * ((int(img[x, y + 1] - int(img[x, y]))) ** 2)
        return out

    def vollath(self, img, img1=None, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s = img.shape
        u = np.mean(img)
        img1 = np.zeros(s, )
        img1[0: s[0] - 1, :] = img[1: s[0], :]
        out = np.sum(np.multiply(img, img1)) - s[0] * s[1] * (u ** 2)
        return np.sqrt(out / (s[1] * (s[0] - 1))) if out >= 0 else 0

    def spatial_frequency(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.int32(img)
        rf, cf = 0, 0
        k = 0
        for i in range(0, len(img)):
            for j in range(1, len(img[i])):
                rf += math.pow(img[i, j] - img[i, j - 1], 2)
                k += 1
        rf /= k

        k = 0
        for i in range(1, len(img)):
            for j in range(0, len(img[i])):
                cf += math.pow(img[i, j] - img[i - 1, j], 2)
                k += 1
        cf /= k

        return math.sqrt(rf + cf)

    def correlation_coe(self, ref, dist, multi_channel=False):
        if ref is None or dist is None:
            return 0
        if multi_channel:
            ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
            dist = cv2.cvtColor(dist, cv2.COLOR_RGB2GRAY)
        s = np.shape(ref)
        a_avg = np.sum(ref) / (s[0] * s[1])
        b_avg = np.sum(dist) / (s[0] * s[1])

        ta = ref - a_avg
        tb = dist - b_avg

        cov_ab = np.sum(ta * tb)

        sq = np.sqrt(np.sum(ta * ta) * np.sum(tb * tb))
        corr_factor = cov_ab / sq
        return corr_factor

    def standard_deviation(self, img, img1=None, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        s = img.shape
        img = img - np.mean(img)
        sd = np.sqrt(np.sum(np.multiply(img, img)) / (s[0] * s[1]))
        return sd

    def average_gradient(self, img, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.int32(img)
        ag = 0
        for i in range(1, len(img)):
            for j in range(1, len(img[i])):
                dx = img[i, j] - img[i - 1, j]
                dy = img[i, j] - img[i, j - 1]
                ds = np.sqrt((pow(dx, 2) + pow(dy, 2)) / 2)
                ag += ds
        return ag / ((len(img) - 1) * (len(img[0]) - 1))

    def nmi(self , img_a, img_b, multi_channel=False):
        if img_a is None or img_b is None:
            return 0
        if multi_channel:
            img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
            img_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

        img_a = img_a.flatten()
        img_b = img_b.flatten()
        return metrics.normalized_mutual_info_score(img_a , img_b , average_method = 'arithmetic')

    def entropy(self, img, img1=None, multi_channel=False):
        if multi_channel:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        temp = np.zeros((256,), dtype=np.float32)
        k = 0
        for i in range(len(img)):
            for j in range(len(img[i])):
                val = img[i, j]
                temp[val] = temp[img[i, j]] + 1
                k = k + 1

        for i in range(len(temp)):
            temp[i] = temp[i] / k

        en = 0
        for i in range(len(temp)):
            if temp[i] != 0:
                en = en - temp[i] * (math.log(temp[i]) / math.log(2.0))

        return en

    def ssim_m(self, ref, dist, multi_channel=True):
        if ref is None or dist is None:
            return 0
        # return skimage.metrics.structural_similarity(ref, dist, channel_axis=2)
        return calculate_ssim(ref, dist, 0)

    def psnr_m(self, ref, dist, multi_channel=True):
        '''
        :param ref:
        :param dist:
        :param multi_channel:
        :return:
        '''
        if ref is None or dist is None:
            return 0
        return skimage.metrics.peak_signal_noise_ratio(ref, dist, )


me = Metrics()

metrics_dict = {
    'ssim': me.ssim_m,
    'psnr': me.psnr_m,
    'cc':   me.correlation_coe,
    'nmi':  me.nmi,
    'sd':   me.standard_deviation,
    'vo':   me.vollath,
}

def cal_metrics(img_source, img_target=None, choose_metrics=['ssim', 'psnr', 'sd', 'vo']):
    '''
    :param img_source: BGR img
    :param img_target:
    :param choose_metrics:
    :return: dict
    '''
    assert (img_source is not None) or (img_target is not None)

    img_source = img_source.permute(0, 2, 3, 1).detach().cpu().numpy()
    img_source = np.array_split(img_source, img_source.shape[0])

    if img_target is not None:
        img_target = img_target.permute(0, 2, 3, 1).detach().cpu().numpy()
        img_target = np.array_split(img_target, img_target.shape[0])

    re_dict = dict()
    for cm in choose_metrics:
        re_dict[cm] = 0

    for i, source in enumerate(img_source):
        source = source.squeeze(0) * 255
        img_source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)  # COLOR_BGR2GRAY
        if img_target is not None:
            target = img_target[i].squeeze(0) * 255
            img_target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)  # COLOR_BGR2GRAY
        else:
            target = None
            img_target_gray = None
        for cm in choose_metrics:
            if cm == 'ssim':
                re_dict[cm] += metrics_dict[cm](source.astype(np.uint8), target.astype(np.uint8))
            else:
                re_dict[cm] += metrics_dict[cm](img_source_gray, img_target_gray)

    for cm in choose_metrics:
        re_dict[cm] /= len(img_source)

    return re_dict

