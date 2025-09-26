import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .shallow_head import ShallowNet
from ..losses import accuracy
from mmseg.models.utils.wrappers import resize
from mmseg.models.backbones.mambaunet import VSSM


class SegmentationHead(nn.Module):
    def __init__(self, conv_cfg, norm_cfg, act_cfg, in_channels, mid_channels, n_classes, *args, **kwargs):
        super(SegmentationHead, self).__init__()

        self.conv_bn_relu = ConvModule(in_channels, mid_channels, 3,
                                       stride=1,
                                       padding=1,
                                       conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)

        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv_out(x)
        return x


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, gauss_chl=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.gauss_chl = gauss_chl
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda')):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(self.gauss_chl, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr

class SRDecoder(nn.Module):
    # super resolution decoder
    def __init__(self, conv_cfg, norm_cfg, act_cfg, channels=128, up_lists=[2, 2, 2]):
        super(SRDecoder, self).__init__()
        self.conv1 = ConvModule(channels, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up1 = nn.Upsample(scale_factor=up_lists[0])
        self.conv2 = ConvModule(channels // 2, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up2 = nn.Upsample(scale_factor=up_lists[1])
        self.conv3 = ConvModule(channels // 2, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up3 = nn.Upsample(scale_factor=up_lists[2])
        self.conv_sr = SegmentationHead(conv_cfg, norm_cfg, act_cfg, channels, channels // 2, 3, kernel_size=1)

    def forward(self, x, fa=False):#输入x是2 128 39 39
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        feats = self.conv3(x)
        outs = self.conv_sr(feats)#2 3 624 624
        if fa:
            return feats, outs
        else:
            return outs



#将通道数写死

class Reducer(nn.Module):
    # Reduce channel (typically to 128)
    def __init__(self, in_channels=512, reduce=128, bn_relu=True):
        super(Reducer, self).__init__()
        self.bn_relu = bn_relu
        self.conv1 = nn.Conv2d(in_channels, reduce, 1, bias=False)
        if self.bn_relu:
            self.bn1 = nn.BatchNorm2d(reduce)

    def forward(self, x):

        x = self.conv1(x)
        if self.bn_relu:
            x = self.bn1(x)
            x = F.relu(x)

        return x






# from .my_fusion_module.add import Vssm_cnn_Add1,Vssm_cnn_Add2
# from .my_fusion_module.cat import Vssm_cnn_Cat1,Vssm_cnn_Cat2
from .my_fusion_module.sp_con_pag import RelationAwareFusion_coordatt_pag_raf as RCPR
from .CNN_module.RFD import ShallowNet_RFD
# from .CNN_module.convnext import convnext_supertiny
# from .CNN_module.efficientnetv2 import efficientnetv2_s,efficientnetv2_m
# from .CNN_module.segnext import segnext_t
# from .CNN_module.resnet import resnet50,resnet18
# from .CNN_module.model_v2 import MobileNetV2
# from .CNN_module.model_v3 import mobilenet_v3_large
# from .CNN_module.convnext import convnext_tiny
# from .CNN_module.pkinet import PKINet
# from .fusion_module.FAM import FeatureAggregationModule as FAM
# from .fusion_module.FFM import FFM
# from .fusion_module.GFM import Gated_Fusion as GFM
# from .fusion_module.CLM import CrossFusionModule as CFM
# from .fusion_module.CMX import FeatureFusionModule as cmxffm
#-------------------------------------------------------------#
#-------------------------------------------------------------#
@HEADS.register_module()
class ISDHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels, reduce=False,fusion_mode='raf',consist=False,
                 model_cls='mamba',dims=[48, 96, 192, 384],depths=[1, 1, 2, 1],
                 shallow_model_inchan=6,lap=True,
                 **kwargs):
        super(ISDHead, self).__init__(**kwargs)
        self.down_ratio = down_ratio

        self.sr_decoder = SRDecoder(self.conv_cfg, self.norm_cfg, self.act_cfg,
                                    channels=self.channels, up_lists=[4, 2, 2])
        # shallow branch
        self.model_cls=model_cls
        self.dims=dims
        self.depth=depths

        self.shallow_model_inchan=shallow_model_inchan
        self.lap=lap
        if self.model_cls =="mamba":
            self.raf_channel = dims[0]
            self.fuse8 = RelationAwareFusion(self.raf_channel,self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=4)
            self.fuse16 = RelationAwareFusion(self.raf_channel,self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=8)
            self.stdc_net = VSSM(
                patch_size=4,
                in_chans=self.shallow_model_inchan,
                dims=self.dims,#dims=[96, 192, 384, 768]
                depths=self.depth,
                pretrain_model="/media/cm/2c5e0c44-80c0-4ab7-b8af-c5a0997b2a7f/zjb/UHR_Model/checkpoint/vmamba_tiny_e292.pth",)
        elif self.model_cls =="ShallowNet":
            self.raf_channel = 128
            # self.fuse8 = RelationAwareFusion(self.raf_channel, self.channels, self.conv_cfg, self.norm_cfg,
            #                                  self.act_cfg, ext=2)
            # self.fuse16 = RelationAwareFusion(self.raf_channel, self.channels, self.conv_cfg, self.norm_cfg,
            #                                   self.act_cfg, ext=4)

            self.fuse8 =RCPR(self.raf_channel, self.channels, self.conv_cfg, self.norm_cfg,
                                             self.act_cfg, ext=2)
            self.fuse16 = RCPR(self.raf_channel, self.channels, self.conv_cfg, self.norm_cfg,
                                              self.act_cfg, ext=4)

            #fam
            # self.fuse8=FAM(256,128)
            # self.fuse16=FAM(512,128)

            #FFM
            # self.fuse8=FFM(256,128)
            # self.fuse16=FFM(512,128)
            #
            #gfm
            # self.fuse8=GFM(256,256)
            # self.fuse16=GFM(512,256)

            #CFM
            # self.fuse8=CFM(256,128)
            # self.fuse16=CFM(512,128)

            #cmx
            # self.fuse8=cmxffm(dim=128,reduction=1,num_heads=8,smooth_channels=256)
            # self.fuse16=cmxffm(dim=128,reduction=1,num_heads=8,smooth_channels=512)

            #
            # self.fuse8 =Vssm_cnn_Add2(self.raf_channel,self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg,)
            # self.fuse16 =Vssm_cnn_Add1(self.raf_channel,self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg,)


            # self.fuse8 = Vssm_cnn_Cat2(self.raf_channel,self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg,)
            # self.fuse16 = Vssm_cnn_Cat1(self.raf_channel,self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg,)


            # self.stdc_net = ShallowNet(in_channels=self.shallow_model_inchan,
            #                            pretrain_model="")#64.41
            self.stdc_net = ShallowNet_RFD(in_channels=self.shallow_model_inchan,
                                       pretrain_model="")

            # self.stdc_net =efficientnetv2_m()#0.625   1.375
            # self.stdc_net =efficientnetv2_s()#0.5   1.25
            # self.stdc_net = segnext_t()#0.5 1.25
            # self.stdc_net=resnet50()# 4 8
            # self.stdc_net=resnet18()# 1  2
            # self.stdc_net=MobileNetV2()#0.25 0.5
            # self.stdc_net=mobilenet_v3_large()#0.1875   0.625
            # self.stdc_net=convnext_tiny()#1.5 3
            # self.stdc_net=PKINet('SST')#0.5 1
            # self.stdc_net=PKINet('SS')#1 2



        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        self.conv_seg_aux_16 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                                self.channels // 2, self.num_classes, kernel_size=1)
        self.conv_seg_aux_8 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                               self.channels // 2, self.num_classes, kernel_size=1)
        self.conv_seg = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                         self.channels // 2, self.num_classes, kernel_size=1)

        self.reduce = Reducer() if reduce else None
        self.fusion_mode=fusion_mode

        self.channel_reduce1=Reducer(768,128)
        self.consist=consist

    def forward(self, inputs, prev_output, input_16,train_flag=True):
        """Forward function."""
        #input:2 3 1224 1224
        if self.lap:
            prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)#2 3 1224 1224;2 3 612 612
            high_residual_1 = prymaid_results[0]#1224 1224 3 2
            high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
                                            align_corners=False)#2 3 1224 1224
            high_residual_input = torch.cat([high_residual_1, high_residual_2], dim=1)

        elif not self.lap:
            # high_residual_1 = inputs
            # high_residual_2 = inputs
            high_residual_input =inputs


        if self.model_cls =="mamba":
            feature_all=self.stdc_net(high_residual_input)
            shallow_feat8, shallow_feat16 =feature_all[2],feature_all[3]

        else:
            shallow_feat8, shallow_feat16 = self.stdc_net(high_residual_input)#2 256 153 153;2 512 77 77


        deep_feat = prev_output[0]#2 128 39 39 deeplabv3分支的分类前结果
        deep_feat16=input_16
        # deep_feat = prev_output  # 2 128 39 39 deeplabv3分支的分类前结果
        # ----------------------------------------------------#
        # ----------------------------------------------------#


        if self.reduce is not None:
            deep_feat = self.reduce(deep_feat)
        # stage 1

        _, aux_feat16, fused_feat_16 = self.fuse16(shallow_feat16, deep_feat)#然后吧这个deeplabv3的特征与16特征第一次融合
        # stage 2
        _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, fused_feat_16)#吧融合结果与8倍第二次融合
        output = self.cls_seg(fused_feat_8)  # 2 7 153 153




        #
        # _, aux_feat16, fused_feat_16 = self.fuse16(shallow_feat16, deep_feat)#然后吧这个deeplabv3的特征与16特征第一次融合
        # # stage 2
        # _, aux_feat8, fused_feat_8 = self.fuse8(shallow_feat8, fused_feat_16)#吧融合结果与8倍第二次融合
        # output = self.cls_seg(fused_feat_8)#2 7 153 153
        if train_flag:#结构蒸馏损失
            output_aux16 = self.conv_seg_aux_16(aux_feat8)#aux——feat8是上采样后的损失  2 7 153 153  cbr
            output_aux8 = self.conv_seg_aux_8(aux_feat16)#aux-feat16是aware fusion 上采样后的损失  2 7 77 77   cbr
            feats, output_sr = self.sr_decoder(deep_feat , True)#2 128 624 624;2 3 624 624
            #重构损失

            losses_re = self.image_recon_loss(high_residual_input, output_sr, re_weight=0.1)#将拉普拉斯金字塔的结果和deeplab编码再解码的结果做重构损失
            if self.consist:#如果要做特征之间的一致性约束
                loss_consist1=self.gauss_spatial_consistency_loss(self.reduce256_1(shallow_feat8),self.reduce384_1(deep_feat16))
                loss_consist2=self.gauss_spatial_consistency_loss(self.reduce512_1(shallow_feat16), self.reduce128_1(deep_feat))
                losses_fa = self.feature_affinity_loss(deep_feat, feats)#把上采样后的特征与deeplab提取的特征做特征关系损失
                return output, output_aux16, output_aux8, losses_re, losses_fa,loss_consist1,loss_consist2
            else:
                losses_fa = self.feature_affinity_loss(deep_feat, feats)  # 把上采样后的特征与deeplab提取的特征做特征关系损失
                return output, output_aux16, output_aux8, losses_re, losses_fa
        else:
            return output


    def image_recon_loss(self, img, pred, re_weight=0.5):
        loss = dict()
        if pred.size()[2:] != img.size()[2:]:
            pred = F.interpolate(pred, img.size()[2:], mode='bilinear', align_corners=False)
        recon_loss = F.mse_loss(pred, img) * re_weight#均方跟损失
        loss['recon_losses'] = recon_loss
        return loss

    def gaussion_similarity(self,tensor1,tensor2,sigma=1.0):
        distance=torch.sum((tensor1-tensor2)**2,dim=[1,2,3],keepdim=True)
        similarity=torch.exp(-distance/(2*sigma**2))
        return similarity
    def gauss_spatial_consistency_loss(self, img, pred, re_weight=0.5):
        loss = dict()
        if pred.size()[2:] != img.size()[2:]:
            pred = F.interpolate(pred, img.size()[2:], mode='bilinear', align_corners=False)
        #similarity=self.gaussion_similarity(img,pred)
        recon_loss = F.mse_loss(pred, img) * re_weight#均方跟损失
        loss['gauss_consistency_losses'] = recon_loss
        return loss

    def feature_affinity_loss(self, seg_feats, sr_feats, fa_weight=1., eps=1e-6):#上采样后的特征为sr_feats2 128 624 624；deeplab特征为seg_feats2 128 39 39
        if seg_feats.size()[2:] != sr_feats.size()[2:]:
            sr_feats = F.interpolate(sr_feats, seg_feats.size()[2:], mode='bilinear', align_corners=False)#把上采样的特征又采样回去
        loss = dict()
        # flatten:
        seg_feats_flatten = torch.flatten(seg_feats, start_dim=2)#2 128 1521
        sr_feats_flatten = torch.flatten(sr_feats, start_dim=2)#2 128 1521
        # L2 norm
        seg_norm = torch.norm(seg_feats_flatten, p=2, dim=2, keepdim=True)#2 128  1 通道唯独做norm
        sr_norm = torch.norm(sr_feats_flatten, p=2, dim=2, keepdim=True)#2 128 1
        # similiarity
        seg_feats_flatten = seg_feats_flatten / (seg_norm + eps)#2 128 1521
        sr_feats_flatten = sr_feats_flatten / (sr_norm + eps)#2 128 1521
        seg_sim = torch.matmul(seg_feats_flatten.permute(0, 2, 1), seg_feats_flatten)#
        sr_sim = torch.matmul(sr_feats_flatten.permute(0, 2, 1), sr_feats_flatten)#
        # L1 loss
        loss['fa_loss'] = F.l1_loss(seg_sim, sr_sim.detach()) * fa_weight#detach（）分离梯度了
        return loss
    def _stack_batch_gt(self, batch_data_samples) :
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logits,
                     batch_data_samples) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)#那个标签列表会在这个函数中生成为bs 1 w h的标签
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)#将head产生的结果进行上采样
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)#压缩一维

        if not isinstance(self.loss_decode, nn.ModuleList):#调用损失函数
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)#计算损失
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def forward_train(self, inputs, prev_output,input_16, gt_semantic_seg):
        if self.consist:
            seg_logits, seg_logits_aux16, seg_logits_aux8, losses_recon, losses_fa,loss_consist1,loss_consist2= self.forward(inputs, prev_output,input_16,)
            losses = self.loss_by_feat(seg_logits, gt_semantic_seg)
            losses_aux16 = self.loss_by_feat(seg_logits_aux16, gt_semantic_seg)
            losses_aux8 = self.loss_by_feat(seg_logits_aux8, gt_semantic_seg)
            return losses, losses_aux16, losses_aux8, losses_recon, losses_fa,loss_consist1,loss_consist2
        else:
            seg_logits, seg_logits_aux16, seg_logits_aux8, losses_recon, losses_fa = self.forward(
                inputs, prev_output, input_16)
            losses = self.loss_by_feat(seg_logits, gt_semantic_seg)
            losses_aux16 = self.loss_by_feat(seg_logits_aux16, gt_semantic_seg)
            losses_aux8 = self.loss_by_feat(seg_logits_aux8, gt_semantic_seg)
            return losses, losses_aux16, losses_aux8, losses_recon, losses_fa
    def forward_test(self, inputs, prev_output,input_16, ):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        return self.forward(inputs, prev_output,input_16, False)