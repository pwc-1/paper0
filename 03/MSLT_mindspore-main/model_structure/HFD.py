import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import ops
import os
import numpy as np
import time

def mean_channels(inp):
    assert(inp.ndim == 4)
    spatial_sum = inp.sum(axis=3, keepdims=True).sum(axis=2, keepdims=True)
    return spatial_sum / (inp.shape[2] * inp.shape[3])

def stdv_channels(inp):
    assert(inp.ndim == 4)
    F_mean = mean_channels(inp)
    F_variance = (inp - F_mean).pow(2).sum(axis=3, keepdims=True).sum(axis=2, keepdims=True) / (inp.shape[2] * inp.shape[3])
    return F_variance.pow(0.5)

class CFD(nn.Cell):
    def __init__(self,inchannel,outchannel):
        super(CFD, self).__init__()

        self.contrast = stdv_channels
        # self.avg_pool = ops.AdaptiveAvgPool2D(1)
        self.conv_du = nn.Sigmoid()


    def construct(self, x):
        y = ops.mean(x, axis=[2, 3], keep_dims=True) + self.contrast(x)
        y = self.conv_du(y)
        return x*y, x - (x*y)


# 3layers with control
class HFD(nn.Cell):
    def __init__(self, in_nc=3, out_nc=3, base_nf=40):
        super(HFD, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc
      
        self.cond_scale1 = nn.Dense(base_nf, base_nf,has_bias=True, bias_init=True)
        self.cond_scale2 = nn.Dense(base_nf, base_nf,has_bias=True,  bias_init=True)
        self.cond_scale3 = nn.Dense(base_nf, base_nf,has_bias=True, bias_init=True)

        self.conv0 = nn.Conv2d(in_nc, base_nf, 1, 1,has_bias=True, bias_init=True)
        self.conv00 = nn.Conv2d(base_nf, out_nc, 1, 1,has_bias=True, bias_init=True)

        self.conv1 = nn.Conv2d(base_nf, base_nf, 1, 1,has_bias=True, bias_init=True)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1,has_bias=True, bias_init=True)

        self.d = nn.Conv2d(base_nf, base_nf, 1, 1,has_bias=True, bias_init=True)
        self.distil = CFD(base_nf, base_nf)

        self.act = nn.ReLU()


    def construct(self, x):
        #cond = self.cond_net(x)

        #scale1 = self.cond_scale1(cond)
        #shift1 = self.cond_shift1(cond)

        #scale2 = self.cond_scale2(cond)
        #shift2 = self.cond_shift2(cond)

        #scale3 = self.cond_scale3(cond)
        #shift3 = self.cond_shift3(cond)
        out = self.conv0(x)
        dist1, rem1 = self.distil(out)
        distilled_c1 = self.act(self.d(dist1))


        rem1 = self.conv1(rem1)
        cond1 = self.conv2(rem1)
        cond1 = ops.mean(cond1, axis=[2, 3], keep_dims=False)
        scale1 = self.cond_scale1(cond1)
        rem1 = rem1 * scale1.view(-1, self.base_nf, 1, 1) + rem1
        r_c1 = self.act(rem1)
        
        dist2,rem2 = self.distil(r_c1)
        distilled_c2 = self.act(self.d(dist2))

        rem2 = self.conv1(rem2)
        cond2 = self.conv2(rem2)
        cond2 = ops.mean(cond2, axis=[2, 3], keep_dims=False)
        scale2 = self.cond_scale1(cond2)
        rem2 = rem2 * scale2.view(-1, self.base_nf, 1, 1) + rem2
        r_c2 = self.act(rem2)

        dist3,rem3 = self.distil(r_c2)
        distilled_c3 = self.act(self.d(dist3))

        rem3 = self.conv1(rem3)
        cond3 = self.conv2(rem3)
        cond3 = ops.mean(cond3, axis=[2, 3], keep_dims=False)
        scale3 = self.cond_scale1(cond3)
        rem3 = rem3 * scale3.view(-1, self.base_nf, 1, 1) + rem3
        r_c4 = self.act(rem3)

        out = distilled_c1 + distilled_c2 + distilled_c3 + r_c4  
        out = self.conv00(out)

        return out
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = CSRNet().cuda()
    #a = torch.randn(1, 3, 3840, 2160).cuda()
    a = torch.randn(1, 3, 1024, 1024).cuda()
    # a = torch.randn(1, 3, 512, 512).cuda()
    flops, params = thop.profile(model,inputs=(a,))
    print("flops","params",flops,params)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(20):
        enhanced_image = model(a)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            torch.cuda.synchronize()
            starter.record()
            enhanced_image = model(a)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)
