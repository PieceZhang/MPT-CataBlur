import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision
from pytorch_wavelets import DWTForward
from torchvision import transforms


class FCL(nn.Module):
    """FreqContrastiveLoss"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.DWT = DWTForward(J=1, wave='haar', mode='reflect')
        self.cl_loss_type = 'l1'
        self.loss = torch.nn.L1Loss()
        self.reblur = torchvision.transforms.GaussianBlur(15, sigma=20.)

    def forward(self, network, blur, GT, out, **kwargs):
        rebl = self.reblur(blur)
        p_list = [GT]
        n_list = [blur, rebl]

        with torch.no_grad():
            pdwt_list = []
            for p in p_list:
                pdwt = self.DWT(p)
                pdwt_list.append(pdwt)  # both high freq & low freq component
            ndwt_list = []
            for n in n_list:
                ndwt = self.DWT(n)
                ndwt_list.append(ndwt[1][0])  # high freq component
            adwt = self.DWT(out)  # both high freq & low freq component
            pos_loss = self.cl_pos(adwt, pdwt_list)
            neg_loss = self.cl_neg(adwt, ndwt_list)
        loss = self.cl_loss(pos_loss, neg_loss) * self.config.CL.weight
        return loss

    def cl_pos(self, a, p_list):
        pos_loss = 0
        for p in p_list:
            pos_loss += self.loss(a[0], p[0])  # low freq
            pos_loss += self.loss(a[1][0], p[1][0])  # high freq
        pos_loss /= len(p_list)
        return pos_loss

    def cl_neg(self, a, n_list):
        neg_loss = 0
        for n in n_list:
            neg_loss += self.loss(a[1][0], n)
        neg_loss /= len(n_list)
        return neg_loss

    def cl_loss(self, pos_loss, neg_loss):
        # minimize posloss, maximize negloss

        if self.cl_loss_type in ['l2', 'cosine']:
            cl_loss = pos_loss - neg_loss

        elif self.cl_loss_type == 'l1':
            cl_loss = pos_loss / (neg_loss + 3e-7)
        else:
            raise TypeError(f'{self.args.cl_loss_type} not fount in cl_loss')

        return cl_loss


class FACL(FCL):
    """FreqAugContrastiveLoss"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.DWT = DWTForward(J=1, wave='haar', mode='reflect')
        self.cl_loss_type = 'l1'
        self.loss = torch.nn.L1Loss()
        # self.reblur = torchvision.transforms.GaussianBlur(15, sigma=20.)
        self.aug = transforms.Compose([
            transforms.RandomChoice([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                                     transforms.ColorJitter(0.6, 0.6, 0.6, 0.1)],
                                    p=[0.25, 0.25, 0.25, 0.25]),
            transforms.RandomGrayscale(p=0.2)
        ])  # only use aug in CL? or use aug in trainer too?
        self.aug_len = 10

    def forward(self, network, C, GT, out, **kwargs):
        # rebl = self.reblur(C)
        p_list = [GT]
        n_list = [C, *[self.aug(C) for _ in range(self.aug_len)]]

        with torch.no_grad():
            pdwt_list = []
            for p in p_list:
                pdwt = self.DWT(p)
                pdwt_list.append(pdwt)  # both high freq & low freq component
            ndwt_list = []
            for n in n_list:
                ndwt = self.DWT(n)
                ndwt_list.append(ndwt[1][0])  # high freq component
            adwt = self.DWT(out)  # both high freq & low freq component
            pos_loss = self.cl_pos(adwt, pdwt_list)
            neg_loss = self.cl_neg(adwt, ndwt_list)
        loss = self.cl_loss(pos_loss, neg_loss) * self.config.CL.weight
        return loss
    

class FRCL(FCL):
    """FreqResidualContrastiveLoss"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.DWT = DWTForward(J=1, wave='haar', mode='reflect')
        self.cl_loss_type = 'l1'
        self.loss = torch.nn.L1Loss()
        self.reblur = torchvision.transforms.GaussianBlur(15, sigma=20.)

    def forward(self, network, C, GT, anchor, **kwargs):
        rebl = self.reblur(C)
        p_list = [GT]
        n_list = [C, rebl]

        with torch.no_grad():
            pdwt_list = []
            for p in p_list:
                pdwt = self.DWT(p)
                pdwt_list.append(pdwt)  # both high freq & low freq component
            ndwt_list = []
            for n in n_list:
                ndwt = self.DWT(n)
                ndwt_list.append(ndwt[1][0])  # high freq component
            adwt = self.DWT(anchor)  # both high freq & low freq component
            pos_loss = self.cl_pos(adwt, pdwt_list)
            neg_loss = self.cl_neg(adwt, ndwt_list)
            negres_loss = self.cl_res(adwt, ndwt_list)
        loss = self.cl_loss(pos_loss, neg_loss + negres_loss) * self.config.CL.weight
        return loss

    def cl_res(self, a, n_list):  # not use positive sample, only push away from neg
        res_c_reb = n_list[0] - n_list[1]  # c - reb
        res_a_reb = a[1][0] - n_list[1]  # a - reb
        negres_loss = self.loss(res_c_reb, res_a_reb)
        return negres_loss


class FRACL(FCL):
    """FreqResidualAugContrastiveLoss"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.DWT = DWTForward(J=1, wave='haar', mode='reflect')
        self.cl_loss_type = 'l1'
        self.loss = torch.nn.L1Loss()
        self.reblur = transforms.GaussianBlur(15, sigma=20.)
        self.aug = transforms.Compose([
            transforms.RandomChoice([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                                     transforms.ColorJitter(0.6, 0.6, 0.6, 0.1)],
                                    p=[0.25, 0.25, 0.25, 0.25]),
            transforms.RandomGrayscale(p=0.2)
        ])  # only use aug in CL? or use aug in trainer too?
        self.aug_len = 5

    def forward(self, network, C, GT, anchor, **kwargs):
        rebl = self.reblur(C)
        p_list = [GT]
        n_list = [C, rebl, *[self.aug(rebl) for _ in range(self.aug_len)]]

        with torch.no_grad():
            pdwt_list = []
            for p in p_list:
                pdwt = self.DWT(p)
                pdwt_list.append(pdwt)  # both high freq & low freq component
            ndwt_list = []
            for n in n_list:
                ndwt = self.DWT(n)
                ndwt_list.append(ndwt[1][0])  # high freq component
            adwt = self.DWT(anchor)  # both high freq & low freq component
            pos_loss = self.cl_pos(adwt, pdwt_list)
            neg_loss = self.cl_neg(adwt, ndwt_list)
            negres_loss = self.cl_res(adwt, ndwt_list)
        loss = self.cl_loss(pos_loss, neg_loss + negres_loss) * self.config.CL.weight
        return loss

    def cl_res(self, a, n_list):  # not use positive sample, only push away from neg
        """
        n_list: [C, rebl, rebl_aug1, rebl_aug2, ...]
        """
        c = n_list[0]
        negres_loss = 0
        res_c_reb = []
        for n in n_list[1:]:
            res_c_reb.append(c - n)  # c - reb
        res_c_reb = torch.mean(torch.stack(res_c_reb, dim=0), dim=0).squeeze(0)
        for n in n_list[1:]:
            res_a_reb = a[1][0] - n  # a - reb
            negres_loss += self.loss(res_c_reb, res_a_reb)
        negres_loss /= len(n_list[1:])
        return negres_loss


class FRACLexMB(FCL):
    """FreqResidualAugContrastiveLoss with extra data GT
        My Blur"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.DWT = DWTForward(J=1, wave='haar', mode='reflect')
        self.cl_loss_type = 'l1'
        self.loss = torch.nn.L1Loss()
        self.reblurrdm = transforms.RandomChoice([transforms.GaussianBlur(3, sigma=10.),
                                               transforms.GaussianBlur(5, sigma=10.),
                                               transforms.GaussianBlur(5, sigma=10.),
                                               transforms.GaussianBlur(7, sigma=10.)],
                                              p=[0.25, 0.25, 0.25, 0.25])
        self.aug = transforms.Compose([
            transforms.RandomChoice([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                                     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                                     transforms.ColorJitter(0.6, 0.6, 0.6, 0.1)],
                                    p=[0.25, 0.25, 0.25, 0.25]),
            transforms.RandomGrayscale(p=0.2)
        ])  # only use aug in CL? or use aug in trainer too?
        self.aug_len = 3

    def forward(self, network, blur, GT, anchor, **kwargs):
        # rebl = self.reblur(blur)
        p_list = [GT]  # no augmentation to GT
        n_list = [blur, *[self.aug(blur) for _ in range(self.aug_len)]]

        inputs = kwargs['inputs']
        if 'c_ex' in inputs.keys():  # use extra data for CL_res
            blur_ex = inputs['c_ex']
            rebl_ex = self.reblurrdm(blur_ex)
            nex_et_list = [blur_ex, rebl_ex, *[self.aug(rebl_ex) for _ in range(self.aug_len)]]  # blur_ex, rebl_ex
        else:  # no extra data
            raise

        with torch.no_grad():  # no grad
            pdwt_list = []
            for p in p_list:
                pdwt = self.DWT(p)
                pdwt_list.append(pdwt)  # both high freq & low freq component
            ndwt_list = []
            for n in n_list:
                ndwt = self.DWT(n)
                ndwt_list.append(ndwt[1][0])  # high freq component
            adwt = self.DWT(anchor)  # both high freq & low freq component
            pos_loss = self.cl_pos(adwt, pdwt_list)
            neg_loss = self.cl_neg(adwt, ndwt_list)

            # extended freq res CL
            nexdwt_list = []
            for nex in nex_et_list:
                ndwt = self.DWT(nex)
                nexdwt_list.append(ndwt[1][0])  # high freq component
            pex = network(blur_ex)['result']
            pex_dwt = self.DWT(pex)
            negres_loss = self.cl_res_et(pex_dwt, nexdwt_list)

        loss = self.cl_loss(pos_loss, neg_loss + negres_loss) * self.config.CL.weight
        return loss

    def cl_res_et(self, a, n_list):
        """
        high freq residual
        not use positive sample, only push away from neg
        n_list: [blur, rebl, rebl_aug1, rebl_aug2, ...]
        """
        blur = n_list[0]
        negres_loss = 0
        res_b_reb = []
        for n in n_list[1:]:
            res_b_reb.append(n - blur)  # reb - blur
        res_b_reb = torch.mean(torch.stack(res_b_reb, dim=0), dim=0).squeeze(0)
        for n in n_list[1:]:
            res_a_reb = a[1][0] - n  # a(reb) - reb
            negres_loss += torch.norm(res_a_reb, p=1) / torch.norm(res_b_reb, p=1) * 0.1
        negres_loss /= len(n_list[1:])
        return negres_loss