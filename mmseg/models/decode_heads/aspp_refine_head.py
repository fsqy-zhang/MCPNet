import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.utils.wrappers import resize
from ..builder import HEADS
from .refine_decode_head import RefineBaseDecodeHead
from .aspp_head import ASPPModule
from mmseg.registry import MODELS


@MODELS.register_module()
class RefineASPPHead(RefineBaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(RefineASPPHead, self).__init__(**kwargs)
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        #------------------------------------------------------#
        #------------------------------------------------------#
        self.bottleneck_noASPP = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.outputsmooth=ConvModule(
                                        self.in_channels,
                                        self.channels,
                                        3,
                                        padding=1,
                                        conv_cfg=self.conv_cfg,
                                        norm_cfg=self.norm_cfg,
                                        act_cfg=self.act_cfg)
    def forward(self, inputs):
        """Forward function."""
        #------------------------------------------------------#
        #------------------------------------------------------#
        # x = self._transform_inputs(inputs)#resnet118最后一层特征图
        # # 用来存储decoder的中间特征图
        # fm_middle = []
        # aspp_outs = [
        #     resize(
        #         self.image_pool(x),
        #         size=x.size()[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners)
        # ]#2 128 39 39
        # asppshow=self.aspp_modules(x)#list 4 2 128 39 39
        # aspp_outs.extend(self.aspp_modules(x))#其实就是吧aspp（x）的结果和原结果做了一个通道维度的拼接
        # aspp_outs = torch.cat(aspp_outs, dim=1)#这步结合上一步，其实就是吧aspp（x）的结果和原结果做了一个通道维度的拼接 2 640 39 39
        # output = self.bottleneck(aspp_outs)#2 128 39 39
        # # fm_middle: 初期验证，采用最后cls_seg之前最近的一个特征图
        # fm_middle.append(output)#分类头前经过了aspp聚合后的特征
        # output = self.cls_seg(output)#7分类分类头
        # fm_middle.append(output)
        #------------------------------------------------------#
        #------------------------------------------------------#
        fm_middle = []
        x = self._transform_inputs(inputs)#resnet118最后一层特征图
        out1= self.bottleneck_noASPP(x)
        fm_middle.append(out1)
        output = self.cls_seg(out1)#7分类分类头
        fm_middle.append(output)
        # fm_middle= inputs[-1]
        # output=self.cls_seg(self.outputsmooth(fm_middle))#7分类分类头
        return output, fm_middle