

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

# -----------------------------------------#
# -----------------------------------------#
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)#2 128 77 1 压缩了宽
        x_w = self.pool_w(x).permute(0, 1, 3, 2)#2 128 77 1 压缩了高

        y = torch.cat([x_h, x_w], dim=2)#2 128 154 1
        y = self.conv1(y)#2 8 154 1
        y = self.bn1(y)
        y = self.act(y)#2 8 154 1

        x_h, x_w = torch.split(y, [h, w], dim=2)#2 8 77 1 ；2 8 77 1
        x_w = x_w.permute(0, 1, 3, 2)#2 8 1 77

        a_h = self.conv_h(x_h).sigmoid()#2 128 77 1
        a_w = self.conv_w(x_w).sigmoid()#2 128 1 77

        out = identity * a_w * a_h

        return out
class myPagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
        super(myPagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,
                          kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):  #
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)

        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))#通道要对的上相当于*后为128 又返回为512
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        x = (1 - sim_map) * x_k + sim_map * y_q  #由于做的是坐标注意力，当都相似时使用包含更多语义信息的分支，当不相似时使用局部分支

        return x
class RelationAwareFusion_coordatt_pag(nn.Module):
    def __init__(self, raf_channel,channels, conv_cfg, norm_cfg, act_cfg, ext=2, r=16):
        super(RelationAwareFusion_coordatt_pag, self).__init__()
        self.r = r

        self.conv_bn_relu1 = ConvModule(raf_channel * ext, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_bn_relu2 = ConvModule(raf_channel, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.context_head = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)
        self.smooth = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                 act_cfg=None)

        self.coordatt1=CoordAtt( channels)
        self.coordatt2=CoordAtt( channels)


        self.pag=myPagFM(raf_channel,channels)
    def forward(self, sp_feat, co_feat):
        # **_att: B x C x 1 x 1
        s_feat1=self.conv_bn_relu1(sp_feat)#调整通道数
        c_feat1=self.conv_bn_relu2(co_feat)

        s_feat2=self.coordatt1(s_feat1)#big
        c_feat2=self.coordatt2(c_feat1)

        # c_feat2 = F.interpolate(c_feat2, s_feat1.size()[2:], mode='bilinear', align_corners=False)#上采样
        # c_feat2 = self.context_head(c_feat2)#插值了用个1*1润色一下
        #
        # out = self.smooth(s_feat2 + c_feat2)#一个1*1卷积融合通道信息
        out=self.pag(s_feat2,c_feat2)
        return s_feat2, c_feat2, out#返回两个尺度一样的图和一个融合图
