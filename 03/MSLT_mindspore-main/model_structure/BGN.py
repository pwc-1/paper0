import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import ops
from . import HFD

# 3layers with control
class MLPNet(nn.Cell):
    def __init__(self, in_nc=3, out_nc=3, base_nf=64):
        super(MLPNet, self).__init__()

        self.base_nf = base_nf
        self.out_nc = out_nc

        # self.cond_net = Condition(in_nc=in_nc, nf=cond_nf)

        self.cond_scale1 = nn.Dense(base_nf, base_nf,has_bias=True, bias_init=True)
        self.conv1 = nn.Conv2d(in_nc, base_nf, 1, 1,has_bias=True, bias_init=True)
        self.conv2 = nn.Conv2d(base_nf, base_nf, 1, 1,has_bias=True, bias_init=True)
        self.conv3 = nn.Conv2d(base_nf, out_nc, 1, 1,has_bias=True, bias_init=True)
        self.act = nn.ReLU()

    def construct(self, x):
        out = self.conv1(x)
        cond1 = self.conv2(out)
        cond1 = ops.mean(cond1, axis=[2, 3], keep_dims=False)
        scale1 = self.cond_scale1(cond1)
        out = out * scale1.view(-1, self.base_nf, 1, 1) + out
        out = self.act(self.conv3(out))
        return out
    

class Slice(nn.Cell):
    def __init__(self):
        super(Slice, self).__init__()

    def construct(self, bilateral_grid, guidemap):

        # N, _, H, W = guidemap.shape
        # meshgrid = ops.Meshgrid(indexing='ij')
        # hg, wg = meshgrid((mindspore.numpy.arange(0, H), mindspore.numpy.arange(0, W)))  # [0,511] HxW
        # hg = hg.float().tile((N, 1, 1)).unsqueeze(3) / (H - 1)  # norm to [0,1] NxHxWx1
        # wg = wg.float().tile((N, 1, 1)).unsqueeze(3) / (W - 1)  # norm to [0,1] NxHxWx1
        # hg, wg = hg * 2 - 1, wg * 2 - 1
        # op = ops.Concat(axis=3)
        # guidemap = guidemap.permute(0, 2, 3, 1).float()#.contiguous()
        # guidemap_guide = op((wg, hg, guidemap)).unsqueeze(1)  # Nx1xHxWx3
        # coeff = ops.grid_sample(bilateral_grid.float(), guidemap_guide, align_corners=True)
        # return coeff.squeeze(2)
        N, _, H, W = guidemap.shape
        meshgrid = ops.Meshgrid(indexing='ij')
        hg, wg = meshgrid((mindspore.numpy.arange(0, H), mindspore.numpy.arange(0, W)))  # [0,511] HxW
        if N>1:
            hg = hg.float().tile((N, 1, 1))
            wg = wg.float().tile((N, 1, 1))
        else:
            hg = hg.float().unsqueeze(0)
            wg = wg.float().unsqueeze(0)
        hg = hg.unsqueeze(3) / (H - 1)  # norm to [0,1] NxHxWx1
        wg = wg.unsqueeze(3) / (W - 1)  # norm to [0,1] NxHxWx1
        hg, wg = hg * 2 - 1, wg * 2 - 1
        op = ops.Concat(axis=3)
        guidemap = guidemap.permute(0, 2, 3, 1).float()#.contiguous()
        guidemap_guide = op((wg, hg, guidemap)).unsqueeze(1)  # Nx1xHxWx3
        coeff = ops.grid_sample(bilateral_grid.float(), guidemap_guide, align_corners=True)
        return coeff.squeeze(2)


class ApplyCoeffs(nn.Cell):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def construct(self, coeff, full_res_input):
        R = (full_res_input * coeff[:, 0:3, :, :]).sum(axis=1,keepdims=True) + coeff[:, 3:4, :, :]
        G = (full_res_input * coeff[:, 4:7, :, :]).sum(axis=1,keepdims=True) + coeff[:, 7:8, :, :]
        B = (full_res_input * coeff[:, 8:11, :, :]).sum(axis=1,keepdims=True)+ coeff[:, 11:12, :, :]
        op = ops.Concat(axis=1)
        result = op((R, G, B))

        return result

class B_transformer(nn.Cell):
    def __init__(self):
        super(B_transformer, self).__init__()

        self.guide = MLPNet(in_nc=3, out_nc=1, base_nf=8)

        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()

        #self.u_net = FDADN.FDADN(in_nc=3, nf=64, out_nc=8)
        self.fenet = HFD.HFD(in_nc=3, out_nc=8, base_nf=40)
        # self.u_net_mini = FDADN(in_nc=3, nf=64, out_nc=3)
        # self.u_net_mini = UNet_mini(n_channels=3)
        self.smooth = nn.PReLU()
        self.fitune = MLPNet(in_nc=3, out_nc=3, base_nf=8)

        self.p = nn.PReLU()

        self.point = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1,pad_mode='pad', padding=0)

    def construct(self, x):
        # x_u= F.interpolate(x, (48, 48), mode='bicubic', align_corners=True)
        
        x_r = ops.interpolate(x, size=(48, 48),mode='bicubic')
        coeff = self.fenet(x_r).reshape(-1, 12, 6, 16, 16)
        
        guidance = self.guide(x)
        slice_coeffs = self.slice(coeff, guidance)

        # x_u = self.u_net_mini(x_u)
        # x_u = F.interpolate(x_u, (x.shape[2], x.shape[3]), mode='bicubic', align_corners=True
        output = self.apply_coeffs(slice_coeffs, self.p(self.point(x))) 

        output = self.fitune(output)

        return output

if __name__=='__main__':
    for i in range(1000):
        bt = B_transformer().cuda()
        data = torch.zeros(1, 3, 1024, 1024).cuda()
        x = bt(data)
        torch.cuda.synchronize()
        print(bt(data).shape)
