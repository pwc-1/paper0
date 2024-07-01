import math
import pandas as pd
import numpy as np
import PIL.Image as Image
import mindspore.dataset.vision as V
import mindspore.dataset.transforms as T
from mindspore import Parameter
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Initializer
from mindspore.ops import composite as C
from einops import rearrange

mindspore.set_context(device_target='Ascend', device_id=0)


class ContextBlock2d(nn.Cell):
    def __init__(self, inplanes, planes, pool='att',
                 fusions=['channel_add']):  # pool='att', fusions=['channel_add'], ratio=8
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1, pad_mode='pad', has_bias=True)
            self.softmax = nn.Softmax(axis=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.SequentialCell(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1, pad_mode='pad', has_bias=True),
                nn.LayerNorm([self.planes, 1, 1], epsilon=1e-05, begin_norm_axis=1, begin_params_axis=1),
                nn.ReLU(),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1, pad_mode='pad', has_bias=True)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.SequentialCell(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1, pad_mode='pad', has_bias=True),
                nn.LayerNorm([self.planes, 1, 1], epsilon=1e-05, begin_norm_axis=1, begin_params_axis=1),
                nn.ReLU(),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1, pad_mode='pad', has_bias=True)
            )
        else:
            self.channel_mul_conv = None
        # self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            # torch.kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        # if self.channel_add_conv is not None:
        #     last_zero_init(self.channel_add_conv)
        # if self.channel_mul_conv is not None:
        #     last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.shape
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = ops.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def construct(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]

            channel_mul_term = ops.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]

            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class CAWA(nn.Cell):
    def __init__(self, num_classes, aux=False, backbone=None, jpu=False, pretrained_base=False, zt='val', **kwargs):
        super(CAWA, self).__init__()
        self.aux = aux
        self.encoder = Encoder(32, 48, 64, (64, 96, 128), 128)
        self.feature_fusion = FeatureFusionModule(64, 64, 128)
        self.classifier = Classifer(128, num_classes)
        self.zt = zt

        if self.zt == 'train':
            self.classifier4 = Classifer1(128, num_classes)
            self.class5 = nn.SequentialCell(
                nn.Conv2d(128, 256, 3, padding=1, pad_mode='pad', has_bias=True),
                nn.BatchNorm2d(256, momentum=0.9),
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, padding=1, pad_mode='pad', has_bias=True),
                nn.AdaptiveAvgPool2d((2, 2)),
                ops.flatten(),
                nn.Dense(512 * 4, num_classes - 1),  # ignore the background class
            )

    def construct(self, x):
        size = x.shape[2:]
        x8, x16, x32, x_en, x4 = self.encoder(x)
        x, x8_aux, x32_k, x16_fusion_k, x8_k = self.feature_fusion(x8, x_en, x16, x4)
        x = self.classifier(x, size)
        result = {}
        outputs = []
        outputs.append(x)

        if self.zt == 'train':
            auxout_se = self.class5(x32)  # semantic auxiliary supervision
            auxout_se = ops.sigmoid(auxout_se)
            result['class'] = auxout_se
            auxout = self.classifier4(x8_aux, size)

            result['edg'] = auxout
        result['out'] = x

        return result


class _ConvBNReLU(nn.Cell):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.ReLU()
        )

    def construct(self, x):
        return self.conv(x)


class _DSConv(nn.Cell):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, padding=1, group=dw_channels, pad_mode='pad',
                      has_bias=False),
            nn.BatchNorm2d(dw_channels, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(dw_channels, out_channels, 1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.ReLU()
        )

    def construct(self, x):
        return self.conv(x)


class _DWConv(nn.Cell):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(dw_channels, out_channels, 3, stride, padding=1, group=dw_channels, pad_mode='pad',
                      has_bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.ReLU()
        )

    def construct(self, x):
        return self.conv(x)


class _GroupConv(nn.Cell):
    def __init__(self, dw_channels, out_channels, stride=1, groupss=8, **kwargs):
        super(_GroupConv, self).__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(dw_channels, out_channels, 3, stride, padding=1, group=groupss, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.ReLU()
        )

    def construct(self, x):
        return self.conv(x)


class GCU(nn.Cell):
    """Global Context Upsampling module"""

    def __init__(self, in_ch1=128, in_ch2=128, in_ch3=128):
        super(GCU, self).__init__()

        self.gcblock1 = ContextBlock2d(in_ch1, in_ch1, pool='avg')
        self.group_conv1 = _GroupConv(in_ch1, in_ch1, 1, 2)
        self.group_conv2 = _GroupConv(in_ch2, in_ch2, 1, 2)

        self.gcblock3 = ContextBlock2d(in_ch3, in_ch3)

    def construct(self, x32, x16, x8):
        x32 = self.gcblock1(x32)

        x16_32 = ops.interpolate(x32, x16.shape[2:], mode='bilinear', align_corners=True)
        x16_32 = x16 + x16_32
        x16_fusion = self.group_conv1(x16_32)

        x8_16 = ops.interpolate(x16_fusion, x8.shape[2:], mode='bilinear', align_corners=True)
        # x8_fusion = torch.mul(x8 , x8_16)
        x8_fusion = x8 + x8_16
        x8gp = self.group_conv2(x8_fusion)
        x8gc = self.gcblock3(x8gp)
        return x8gc, x32, x16_fusion, x8gp


class Encoder(nn.Cell):
    """Global feature extractor module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(2, 2, 2), **kwargs):
        super(Encoder, self).__init__()
        self.conv = Conv(3, dw_channels1, 3, 2)

        self.dsconv3 = CAWAConv(in_channels)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, in_channels, 2)

        self.dsconv4 = CAWAConv(64)
        self.conv5 = _DSConv(64, 96)
        self.conv6 = _DSConv(96, 128)

        self.ppm = PPContextModule(out_channels, out_channels // 4, out_channels, [1, 2, 4])

    def construct(self, x):
        x = self.conv(x)
        x4 = self.dsconv1(x)

        x8 = self.dsconv2(x4)

        x16 = self.dsconv3(x8)

        x32 = self.dsconv4(x16)
        x32 = self.conv5(x32)
        x32_2 = self.conv6(x32)

        out = self.ppm(x32_2)
        return x8, x16, x32_2, out, x4


class FeatureFusionModule(nn.Cell):
    """Feature fusion module"""

    def __init__(self, x8_in_ch=64, x16_in_ch=64, out_channels=128, **kwargs):
        super(FeatureFusionModule, self).__init__()

        self.conv_8 = nn.SequentialCell(
            nn.Conv2d(x8_in_ch, out_channels, 1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.ReLU()
        )

        self.conv_16 = nn.SequentialCell(
            nn.Conv2d(x16_in_ch, out_channels, 1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.9))

        self.gcu = GCU(out_channels, out_channels, out_channels)

        self.dw = _DSConv(48, 128, 2)

        self.fusion = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128, 1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.Sigmoid()
        )
        self.conv5 = Conv(256, 128)

    def construct(self, x8, x32, x16, x4):
        x8 = self.conv_8(x8)
        x16 = self.conv_16(x16)
        x4 = self.dw(x4)
        x4 = ops.cat([x4, x8], 1)
        x4 = self.conv5(x4)
        x4 = self.fusion(x4) * x4
        out, x32_k, x16_fusion_k, x8_k = self.gcu(x32, x16, x4)

        return out, x4, x32_k, x16_fusion_k, x8_k


class Classifer(nn.Cell):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()

        self.dsconv1 = Conv(dw_channels, dw_channels)

        self.conv1 = nn.Conv2d(dw_channels, dw_channels, stride, pad_mode='pad', has_bias=True)

        self.conv = nn.SequentialCell(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1, pad_mode='pad', has_bias=True)
        )

    def construct(self, x, size):
        x = self.dsconv1(x)
        x = self.conv1(x)
        x = self.conv(x)

        x = ops.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


class Classifer1(nn.Cell):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer1, self).__init__()

        self.conv = nn.SequentialCell(
            nn.Conv2d(128, 32, 3, padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(32, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1, pad_mode='pad', has_bias=True)
        )

    def construct(self, x, size):
        x = self.conv(x)

        x = ops.interpolate(x, size, mode='bilinear', align_corners=True)
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Cell):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.ReLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=autopad(k, p, d), group=g, dilation=d, pad_mode='pad',
                              has_bias=False)
        self.bn = nn.BatchNorm2d(c2, momentum=0.9)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Cell) else nn.Identity()

    def construct(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class CAWAConv(nn.Cell):

    def __init__(self, c, g=4):
        super().__init__()

        self.softmax = nn.Softmax(axis=-1)
        self.attention = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, pad_mode="pad", has_bias=True),
            nn.BatchNorm2d(c, momentum=0.9)
        )
        self.GWC = _DWConv(c, c)
        self.ds_conv = Conv(c, c * 4, k=3, s=2, g=4)

    def construct(self, x):
        xk = x
        m = 2
        n = 2
        h, w = x.shape[-2], x.shape[-1]
        pad_h = (h % m)
        pad_w = (w % n)
        if pad_h > 0 or pad_w > 0:
            xk = ops.pad(xk, (0, pad_w, 0, pad_h))
        # 2 64 102 180
        tp0 = self.attention(xk) * xk
        A, B, C, D = tp0.shape
        att = tp0.reshape(A, B, C // 2, D // 2, 4)

        # att = rearrange(self.attention(xk)*xk, 'B C (m H) (n W) -> B C H W (m n)', m=2, n=2)
        att = self.softmax(att)

        tp1 = self.ds_conv(x)
        A, B, C, D = tp1.shape
        x = tp1.reshape(A, B // 4, C, D, 4)

        # x = rearrange(self.ds_conv(x), 'B (m C) H W -> B C H W m', m=4)
        x = ops.sum(x * att, -1)
        return x


class PPContextModule(nn.Cell):
    """
    Simple Context module.

    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=None):
        super().__init__()

        self.stages = nn.CellList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])

        self.conv_out = Conv(
            inter_channels,
            out_channels,
            k=3,
            p=1
        )

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        # prior = nn.AdaptiveAvgPool2d(output_size=size)
        # conv = Conv(
        #     in_channels, out_channels, k=1)
        conv = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(output_size=size),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='pad', has_bias=True),
            nn.BatchNorm2d(out_channels, momentum=0.9)
        )
        # return nn.Sequential(prior, conv)
        return conv

    def construct(self, input):
        out = None
        input_shape = input.shape[2:]

        for stage in self.stages:
            x = stage(input)
            x = ops.interpolate(
                x,
                input_shape,
                mode='bilinear',
                align_corners=self.align_corners)
            if out is None:
                out = x
            else:
                out += x

        out = self.conv_out(out)
        return out


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            V.ToTensor(),
            V.Normalize(mean=mean, std=std, is_hwc=False),
        ])

    def __call__(self, img):
        return self.transforms(img)


if __name__ == '__main__':
    param_dict = mindspore.load_checkpoint(cawa_path + '/CACW_MT.ckpt')
    pd.DataFrame(param_dict.keys()).to_csv('param_ga_torch.csv')
    # print(param_dict.keys())
    img = Image.open(cawa_path + "/3667.jpg").convert('RGB')
    fk = CAWA(6)
    mindspore.load_param_into_net(fk, param_dict)
    s = SegmentationPresetEval(512)
    img = s(img)
    fk.set_train(False)
    img = np.array(img)
    img = img.reshape((1, 3, 376, 491))

    img = mindspore.tensor(img)
    print(img.shape)
    # fk.load_state_dict()
    target = fk(img)
    # confmat = utils.ConfusionMatrix(6)

    print(target.keys())

    print(target['out'])
