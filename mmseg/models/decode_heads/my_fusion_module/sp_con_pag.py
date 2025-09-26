
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Sequence
from mmcv.cnn import ConvModule

import torch
import torch.nn as nn




import math

class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg):
        super(ChannelAtt, self).__init__()
        self.conv_bn_relu = ConvModule(in_channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x, fre=False):
        """Forward function."""
        feat = self.conv_bn_relu(x)
        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        return feat, atten


#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#

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

        x = (1 - sim_map) * x_k + sim_map * y_q  #

        return x
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
        x_h = self.pool_h(x)  # 2 128 77 1 压缩了宽
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 2 128 77 1 压缩了高

        y = torch.cat([x_h, x_w], dim=2)  # 2 128 154 1
        y = self.conv1(y)  # 2 8 154 1
        y = self.bn1(y)
        y = self.act(y)  # 2 8 154 1

        x_h, x_w = torch.split(y, [h, w], dim=2)  # 2 8 77 1 ；2 8 77 1
        x_w = x_w.permute(0, 1, 3, 2)  # 2 8 1 77

        a_h = self.conv_h(x_h).sigmoid()  # 2 128 77 1
        a_w = self.conv_w(x_w).sigmoid()  # 2 128 1 77

        # feat = identity
        # att=torch.sum(a_w*a_h, dim=1)
        out = identity * a_w * a_h

        return out


#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
class RelationAwareFusion_coordatt_pag_raf(nn.Module):
    def __init__(self,raf_channel, channels, conv_cfg, norm_cfg, act_cfg, ext=2, r=16):
        super(RelationAwareFusion_coordatt_pag_raf, self).__init__()
        self.in_channel_cnn=int(raf_channel * ext)
        self.conv_bn_relu1 = ConvModule(self.in_channel_cnn, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_bn_relu2 = ConvModule(raf_channel, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                        norm_cfg=norm_cfg, act_cfg=act_cfg)


        self.coordatt1 = CoordAtt(channels)
        self.coordatt2 = CoordAtt(channels)

        self.pag = myPagFM(channels, channels)

        self.r = r
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        self.g_sptial_channel = nn.Parameter(torch.zeros(1))

        self.spatial_mlp = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.spatial_att = ChannelAtt(self.in_channel_cnn, channels, conv_cfg, norm_cfg, act_cfg)
        self.context_mlp = nn.Sequential(*[nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels)])
        self.context_att = ChannelAtt(channels, channels, conv_cfg, norm_cfg, act_cfg)

        self.context_head = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)
        self.smooth = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                 act_cfg=None)

        self.smooth2 = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                  act_cfg=None)

    def forward(self, sp_feat, co_feat):
        # **_att: B x C x 1 x 1
        # ----------------------------------------------#
        # 空间
        sptial_s_feat1 = self.conv_bn_relu1(sp_feat)  # 调整通道数
        sptial_c_feat1 = self.conv_bn_relu2(co_feat)

        sptial_s_feat2 = self.coordatt1(sptial_s_feat1)
        sptial_c_feat2 = self.coordatt2(sptial_c_feat1)

        sptail_out = self.pag(sptial_s_feat2, sptial_c_feat2)
        # ----------------------------------------------#
        # 通道
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        b, c, h, w = s_att.size()
        s_att_split = s_att.view(b, self.r, c // self.r)  # 2 16 8
        c_att_split = c_att.view(b, self.r, c // self.r)  # 2 16 8#多头通道注意力 计算量仅为原来的1/8
        chl_affinity = torch.bmm(s_att_split, c_att_split.permute(0, 2, 1))  # 2 16 16 通道向量相乘，通道关联性，通道余弦相似度  在这获得了通道的相似度
        chl_affinity = chl_affinity.view(b, -1)  #
        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity))  # 2 128 激励
        co_mlp_out = F.relu(self.context_mlp(chl_affinity))
        re_s_att = torch.sigmoid(s_att + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_c_att = torch.sigmoid(c_att + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1))
        c_feat = torch.mul(c_feat, re_c_att)  # 2 128 39 39
        s_feat = torch.mul(s_feat, re_s_att)  # 2 128 77 77 重新加权

        c_feat = F.interpolate(c_feat, s_feat.size()[2:], mode='bilinear', align_corners=False)  # 上采样
        c_feat = self.context_head(c_feat)  #
        channel_out = self.smooth(s_feat + c_feat)  # 一个1*1卷积融合通道信息
        # -----------------------------------------#
        # 融合
        final_out=self.smooth2 (sptail_out+channel_out)

        return channel_out, sptail_out,  final_out#

