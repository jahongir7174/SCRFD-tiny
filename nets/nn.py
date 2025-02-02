import torch

from utils import util


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.01)
        self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class DPUnit(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, torch.nn.Identity())
        self.conv2 = Conv(out_ch, out_ch, torch.nn.ReLU(True), k=3, p=1, g=out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Backbone(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(filters[0], filters[1], torch.nn.ReLU(True), k=3, s=2, p=1))
        self.p1.append(DPUnit(filters[1], filters[1]))
        # p2/4
        self.p2.append(torch.nn.MaxPool2d(kernel_size=2))
        self.p2.append(DPUnit(filters[1], filters[2]))
        self.p2.append(DPUnit(filters[2], filters[2]))
        # p3/8
        self.p3.append(torch.nn.MaxPool2d(kernel_size=2))
        self.p3.append(DPUnit(filters[2], filters[3]))
        self.p3.append(DPUnit(filters[3], filters[3]))
        self.p3.append(DPUnit(filters[3], filters[3]))
        # p4/16
        self.p4.append(torch.nn.MaxPool2d(kernel_size=2))
        self.p4.append(DPUnit(filters[3], filters[4]))
        self.p4.append(DPUnit(filters[4], filters[4]))
        # p5/32
        self.p5.append(torch.nn.MaxPool2d(kernel_size=2))
        self.p5.append(DPUnit(filters[4], filters[5]))
        self.p5.append(DPUnit(filters[5], filters[5]))
        self.p5.append(DPUnit(filters[5], filters[5]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return [p3, p4, p5]


class Neck(torch.nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.conv1 = DPUnit(filters[5], filters[4])
        self.conv2 = DPUnit(filters[4], filters[3])
        self.conv3 = DPUnit(filters[3], filters[3])

    def forward(self, x):
        p3, p4, p5 = x
        p5 = self.conv1(p5)
        p4 = self.conv2(p4 + self.up(p5))
        p3 = self.conv3(p3 + self.up(p4))
        return [p3, p4, p5]


class Head(torch.nn.Module):
    def __init__(self, params, filters, nc=1, nk=5):
        super().__init__()
        self.nc = nc
        self.nk = nk
        self.anchor_generator = util.AnchorGenerator(params['face_anchors']['strides'],
                                                     params['face_anchors']['ratios'],
                                                     params['face_anchors']['scales'],
                                                     params['face_anchors']['sizes'])
        # usually the numbers of anchors for each level are the same except SSD detectors
        self.na = self.anchor_generator.anchors[0].size(0)

        self.m = torch.nn.ModuleList()
        self.cls = torch.nn.ModuleList()
        self.box = torch.nn.ModuleList()
        self.kpt = torch.nn.ModuleList()

        for i in range(len(self.anchor_generator.strides)):
            self.m.append(DPUnit(filters[i], filters[i]))

            self.box.append(torch.nn.Conv2d(filters[i], 4 * self.na, kernel_size=1))
            self.cls.append(torch.nn.Conv2d(filters[i], self.nc * self.na, kernel_size=1))
            self.kpt.append(torch.nn.Conv2d(filters[i], self.nk * 2 * self.na, kernel_size=1))

    def forward(self, x):
        n = x[0].shape[0]
        x = [m(i) for i, m in zip(x, self.m)]

        cls = [m(i) for i, m in zip(x, self.cls)]
        box = [m(i) for i, m in zip(x, self.box)]
        kpt = [m(i) for i, m in zip(x, self.kpt)]
        if torch.onnx.is_in_onnx_export():
            cls = [i.permute(0, 2, 3, 1).reshape(n, -1, self.nc).sigmoid() for i in cls]
            box = [i.permute(0, 2, 3, 1).reshape(n, -1, 4) for i in box]
            kpt = [i.permute(0, 2, 3, 1).reshape(n, -1, 10) for i in kpt]
        return cls, box, kpt


class Detector(torch.nn.Module):
    def __init__(self, params, filters):
        super().__init__()
        self.backbone = Backbone(filters)
        self.neck = Neck(filters)
        self.head = Head(params, (filters[3], filters[3], filters[4]))

        img_dummy = torch.zeros(1, filters[0], 256, 256)
        self.head.strides = [256 / x.shape[-2] for x in self.forward(img_dummy)[0]]
        self.strides = self.head.strides

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def version_n(params):
    return Detector(params, filters=(3, 16, 64, 64, 64, 64))
