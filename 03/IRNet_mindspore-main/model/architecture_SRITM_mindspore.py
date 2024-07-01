# import torch.nn as nn
from . import block_mindspore as B
from mindspore import nn
from mindspore import ops
# import torch
import logging
import numpy as np


class SRITM_IRNet_5(nn.Cell):
    def __init__(self, in_nc=3, nf=64, num_modules=5, out_nc=3, upscale=4):
        super(SRITM_IRNet_5, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=1)

        # IRBs
        self.IRB1 = B.IRB_add(in_channels=nf)
        self.IRB2 = B.IRB_add(in_channels=nf)
        self.IRB3 = B.IRB_add(in_channels=nf)
        self.IRB4 = B.IRB_add(in_channels=nf)
        self.IRB5 = B.IRB_add(in_channels=nf)


        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)

        self.conv3_layer = B.conv_layer(nf, nf, kernel_size=3)  # change out channels
        
        
        self.upsampler1 = B.pixelshuffle_block(nf, nf, upscale_factor=2)
        self.upsampler2 = B.pixelshuffle_block(nf, out_nc, upscale_factor=2)


    def construct(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IRB1(out_fea)
        out_B2 = self.IRB2(out_B1)
        out_B3 = self.IRB3(out_B2)
        out_B4 = self.IRB4(out_B3)
        out_B5 = self.IRB5(out_B4)
        
        out_B = self.c(ops.cat([out_B1, out_B2, out_B3, out_B4, out_B5], axis=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.conv3_layer(out_lr)
        
        output = self.upsampler1(output)
        output = self.upsampler2(output)
        
        return output