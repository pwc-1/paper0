import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import ops
import numpy as np
import os
from . import BGN
import time
import cv2 as cv
import mindspore.numpy as msnp


class Lap_Pyramid(nn.Cell):
    def __init__(self):
        super(Lap_Pyramid, self).__init__()
        self.de_conv_1 = nn.Conv2d(3, 3, 3,pad_mode='pad',padding=1, stride=2)
        self.de_conv_2 = nn.Conv2d(3, 3, 3,pad_mode='pad',padding=1,stride=2)
        self.de_conv_3 = nn.Conv2d(3, 3, 3,pad_mode='pad',padding=1,stride=2)
        self.re_cov_1 = nn.Conv2d(3, 3, 3,pad_mode='pad', padding=1, stride=1)
        self.re_cov_2 = nn.Conv2d(3, 3, 3,pad_mode='pad', padding=1, stride=1)
        self.re_cov_3 = nn.Conv2d(3, 3, 3,pad_mode='pad', padding=1, stride=1)

    def de_cov(self, x):
        seq = []
        level_1 = self.de_conv_1(x)
        level_2 = self.de_conv_2(level_1)
        level_3 = self.de_conv_3(level_2)
        seq_1 = x - ops.interpolate(self.re_cov_1(level_1), size=(x.shape[2], x.shape[3]),mode='bilinear')
        seq_2 = level_1 - ops.interpolate(self.re_cov_2(level_2), size=(level_1.shape[2], level_1.shape[3]), mode='bilinear')
        seq_3 = level_2 - ops.interpolate(self.re_cov_3(level_3), size=(level_2.shape[2], level_2.shape[3]), mode='bilinear')
        seq.append(level_3)
        seq.append(seq_3)
        seq.append(seq_2)
        seq.append(seq_1)
        return seq
    def pyramid_recons(self, pyr):
        rec_1 = ops.interpolate(self.re_cov_3(pyr[0]), size=(pyr[1].shape[2], pyr[1].shape[3]), mode='bilinear')
        image = rec_1 + pyr[1]
        rec_2 = ops.interpolate(self.re_cov_2(image), size=(pyr[2].shape[2], pyr[2].shape[3]), mode='bilinear')
        image = rec_2 + pyr[2]
        rec_3 = ops.interpolate(self.re_cov_1(image), size=(pyr[3].shape[2], pyr[3].shape[3]), mode='bilinear')
        image = rec_3 + pyr[3]
        return image
     
    
class Trans_high(nn.Cell):
    def __init__(self, num_high=3):
        super(Trans_high, self).__init__()

        self.model = nn.SequentialCell(*[nn.Conv2d(9, 9, 1,1), nn.LeakyReLU(), nn.Conv2d(9, 3, 1, 1)])
        #self.model = nn.Sequential(*[nn.Conv2d(9, 9, 3,1), nn.LeakyReLU(), nn.Conv2d(9, 3, 3, 1)])

        self.trans_mask_block_1 = nn.SequentialCell(*[nn.Conv2d(3, 3, 1,1), nn.LeakyReLU(), nn.Conv2d(3, 3, 1,1)])
        self.trans_mask_block_2 = nn.SequentialCell(*[nn.Conv2d(3, 3, 1,1), nn.LeakyReLU(), nn.Conv2d(3, 3, 1,1)])

    def construct(self, x, pyr_original, fake_low):

        pyr_result = []
        pyr_result.append(fake_low)

        mask = self.model(x)

        result_highfreq_1 = pyr_original[1] * mask
        pyr_result.append(result_highfreq_1)

        
        mask_1 = ops.interpolate(mask, size=(pyr_original[2].shape[2], pyr_original[2].shape[3]),mode='bilinear')
        mask_1 = self.trans_mask_block_1(mask_1)
        result_highfreq_2 = pyr_original[2] * mask_1
        pyr_result.append(result_highfreq_2)

        mask_2 = ops.interpolate(mask, size=(pyr_original[3].shape[2], pyr_original[3].shape[3]),mode='bilinear')
        mask_2 = self.trans_mask_block_1(mask_2)
        result_highfreq_3 = pyr_original[3] * mask_2
        pyr_result.append(result_highfreq_3)
        return pyr_result
    

class MSLT(nn.Cell):
    def __init__(self, nrb_high=1, num_high=3):
        super(MSLT, self).__init__()

        self.lap_pyramid = Lap_Pyramid()
        trans_low = BGN.B_transformer()
        trans_high = Trans_high() 
        self.trans_low = trans_low
        self.trans_high = trans_high

    def construct(self, real_A_full):
        pyr_A = self.lap_pyramid.de_cov(real_A_full)
        fake_B_low = self.trans_low(pyr_A[0])
        real_A_up = ops.interpolate(pyr_A[0], size=(pyr_A[1].shape[2], pyr_A[1].shape[3]),mode='bilinear')
        fake_B_up = ops.interpolate(fake_B_low, size=(pyr_A[1].shape[2], pyr_A[1].shape[3]),mode='bilinear')
        high_with_low = ops.concat([pyr_A[1], real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low)
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)

        return fake_B_full
