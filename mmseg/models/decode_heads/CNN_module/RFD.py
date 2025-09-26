
import numpy as np
import torch
import torch.nn as nn


# original size to 4x downsampling layer
class SRFD(nn.Module):
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        out_c14 = int(out_channels / 4)  # out_channels / 4
        out_c12 = int(out_channels / 2)  # out_channels / 2

        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)

        # original size to 2x downsampling layer
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)
        self.cut_c = Cut(out_c14, out_c12)
        self.fusion1 = nn.Conv2d(out_channels, out_c12, kernel_size=1, stride=1)

        # 2x to 4x downsampling layer
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.cut_r = Cut(out_c12, out_channels)
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]

    # original size to 2x downsampling layer
        c = x                   # c = [B, C/4, H, W]
        # CutD
        c = self.cut_c(c)       # c = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]
        # ConvD
        x = self.conv_1(x)      # x = [B, C/4, H, W] --> [B, C/2, H/2, W/2]
        x = self.conv_x1(x)     # x = [B, C/2, H/2, W/2]
        x = self.batch_norm_x1(x)
        # Concat + conv
        x = torch.cat([x, c], dim=1)    # x = [B, C, H/2, W/2]
        x = self.fusion1(x)     # x = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]

    # 2x to 4x downsampling layer
        r = x                   # r = [B, C/2, H/2, W/2]
        x = self.conv_2(x)      # x = [B, C/2, H/2, W/2] --> [B, C, H/2, W/2]
        m = x                   # m = [B, C, H/2, W/2]
        # ConvD
        x = self.conv_x2(x)     # x = [B, C, H/4, W/4]
        x = self.batch_norm_x2(x)
        # MaxD
        m = self.max_m(m)       # m = [B, C, H/4, W/4]
        m = self.batch_norm_m(m)
        # CutD
        r = self.cut_r(r)       # r = [B, C, H/4, W/4]
        # Concat + conv
        x = torch.cat([x, r, m], dim=1)  # x = [B, C*3, H/4, W/4]
        x = self.fusion2(x)     # x = [B, C*3, H/4, W/4] --> [B, C, H/4, W/4]
        return x                # x = [B, C, H/4, W/4]


# CutD
class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x


# Deep feature downsampling
class DRFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):       # input: x = [B, C, H, W]
        c = x                   # c = [B, C, H, W]
        x = self.conv(x)        # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x                   # m = [B, 2C, H, W]

        # CutD
        c = self.cut_c(c)       # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD
        x = self.conv_x(x)      # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)
        x = self.batch_norm_x(x)

        # MaxD
        m = self.max_m(m)       # m = [B, 2C, H/2, W/2]
        m = self.batch_norm_m(m)

        # Concat + conv
        x = torch.cat([c, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)      # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]
        B, C, h, w = x.shape
        x = x.view(B, C, -1)  # x = [B, 2C, H/2*W/2]
        x = x.permute(0, 2, 1)  # x = [B, H/2*W/2, 2C]

        return x, h, w


from torch.nn import init
#-------------------------------------------------------------------------#
import time
import math
import copy


import torch
import torch.nn as nn
import torch.nn.functional as F



#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#
class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, sync=False):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        if sync:
            self.bn = nn.SyncBatchNorm(out_planes)
        else:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


# From https://github.com/MichaelFan01/STDC-Seg
# class AddBottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, block_num=3, stride=1):
#         super(AddBottleneck, self).__init__()
#         assert block_num > 1, print("block number should be larger than 1.")
#         self.conv_list = nn.ModuleList()
#         self.stride = stride
#         if stride == 2:
#             self.avd_layer = nn.Sequential(
#                 nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
#                 nn.BatchNorm2d(out_planes//2),
#             )
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
#                 nn.BatchNorm2d(in_planes),
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_planes),
#             )
#             stride = 1
#
#         for idx in range(block_num):
#             if idx == 0:
#                 self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
#             elif idx == 1 and block_num == 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
#             elif idx == 1 and block_num > 2:
#                 self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
#             elif idx < block_num - 1:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
#             else:
#                 self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
#
#     def forward(self, x):
#         out_list = []
#         out = x
#
#         for idx, conv in enumerate(self.conv_list):
#             if idx == 0 and self.stride == 2:
#                 out = self.avd_layer(conv(out))
#             else:
#                 out = conv(out)
#             out_list.append(out)
#
#         if self.stride == 2:
#             x = self.skip(x)
#
#         return torch.cat(out_list, dim=1) + x
# From https://github.com/MichaelFan01/STDC-Seg
class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx +1)),kernel=2*idx+1,stride=1))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx)),kernel=2*idx+1,stride=1))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class ShallowNet_RFD(nn.Module):
    def __init__(self, base=64, in_channels=3,  layers=[2,2], block_num=4, type="cat", dropout=0.20, pretrain_model=''):
        super(ShallowNet_RFD, self).__init__()
        if type == "cat":
            block = CatBottleneck
        # elif type == "add":
        #     block = AddBottleneck
        self.in_channels = in_channels
        self.features = self._make_layers(base, layers, block_num, block)
        # self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[:1])
        self.x8 = nn.Sequential(self.features[1:3])
        self.x16 = nn.Sequential(self.features[3:5])
        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()
        # self.x4test=nn.Sequential(nn.Conv2d(3,32,3,2,1,),
        #                                nn.BatchNorm2d(32),
        #                                nn.ReLU(inplace=True),
        # nn.Conv2d(32,64,3,2,1,),
        #                                nn.BatchNorm2d(64),
        #                                nn.ReLU(inplace=True))
    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]#权重字典键
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if k == 'features.0.conv.weight' and self.in_channels != 3:
                #v = torch.cat([v, v,v], dim=1)#lap+org
                v = torch.cat([v, v], dim=1)#lap
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict, strict=False)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [SRFD(in_channels=self.in_channels, out_channels=64)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x, cas3=False):

        feat4 = self.x4(x)#8 64 306 306
        feat8 = self.x8(feat4)


        feat16 = self.x16(feat8)


        if cas3:
            return feat4, feat8, feat16
        else:
            return feat8, feat16



















