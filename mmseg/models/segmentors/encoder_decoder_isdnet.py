# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from mmseg.models.utils.wrappers import resize

@MODELS.register_module()
class EncoderDecoderisdnet(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 down_scale,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 refine_input_ratio=1,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 is_frequency=False,


                 ):
        self.is_frequency=is_frequency
        self.down_scale=down_scale
        self.refine_input_ratio=refine_input_ratio


        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)#在这进取会加载权重
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:#在这会将cfg里面的那些解耦头定义到这
        """Initialize ``decode_head``"""
        #___________________________________________________#
        # 将decode,也就是refine-aspp-head的参数初始化到这
        #___________________________________________________#
        self.decode_head = MODELS.build(decode_head[0])#这个MODELS就会将cfg转移到decode
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        # self.out_channels = self.decode_head.out_channels
        #___________________________________________________#
        # 将decode,也就是isdhead的参数初始化到这
        #___________________________________________________#
        self.refine_head=MODELS.build(decode_head[1])


    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)#input
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:#这个其实就是推理的函数
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # 这里值得注意的是，输入图像应该分大分辨率和小分辨率两种
        # 目前的计划：
        # 大分辨率图像：参数传入的image， 输入refine_head中
        # 小分辨率图像：参数传入的image下采样到原来的0.25倍数，输入feature_extractor, 即原有的分支中
        if self.is_frequency:
            deeplab_inputs = self.lap_prymaid_conv.pyramid_decom(inputs)[0]
            img_os2 = nn.functional.interpolate(deeplab_inputs, size=[inputs.shape[-2] // self.down_scale,
                                                                      inputs.shape[-1] // self.down_scale])
        else:
            img_os2 = nn.functional.interpolate(inputs, size=[inputs.shape[-2] // self.down_scale,
                                                           inputs.shape[-1] // self.down_scale])

        if self.refine_input_ratio == 1.:
            img_refine = inputs
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(inputs, size=[int(inputs.shape[-2] * self.refine_input_ratio),
                                                              int(inputs.shape[-1] * self.refine_input_ratio)])
        # torch.cuda.synchronize()
        # start_time1 = time.perf_counter()
        x = self.extract_feat(img_os2)
        # 这里先假设就是只有一个decoder，每个decoder应该返回一组feature map或者是list
        # fm_decoder是decode返回的特征图，或者是一个特征图的list
        out_g, prev_outputs = self.decode_head.forward_test(x)
        # torch.cuda.synchronize()
        # end_time1 = time.perf_counter()
        # print(end_time1 - start_time1)
        # refine_head的输出作为最后的输出特征图
        # 其实这里的refine_head承担了两个功能，第一个是提取大尺度分辨率，第二是融合spatial feature map和context feature map
        # torch.cuda.synchronize()
        # start_time2 = time.perf_counter()
        input_16=x[2]
        out = self.refine_head.forward_test(img_refine, prev_outputs, input_16)
        out = resize(
            input=out,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # torch.cuda.synchronize()
        # end_time2 = time.perf_counter()
        # print(end_time2 - start_time2)
        # print("--------------------------")
        # ;PYTORCH_NO_CUDA_MEMORY_CACHING=1
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.max_memory_allocated())
        # print(torch.cuda.memory_summary())

        return out


        # x = self.extract_feat(inputs)
        # seg_logits = self.decode_head.forward_test(x, batch_img_metas,
        #                                       self.test_cfg)
        #
        # return seg_logits
#
    def _decode_head_forward_train(self, inputs: List[Tensor],img_refine,
                                   data_samples: SampleList) -> dict:#这个函数要定义两个decode
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        #__________________________________________________#
        #构建第一个decode,
        #__________________________________________________#
        loss_decode, prev_features = self.decode_head.forward_train(
            inputs, data_samples)#第一个decode
        losses.update(add_prefix(loss_decode, 'decode'))
        # loss_decode = self.decode_head.loss(inputs, data_samples,
        #                                     self.train_cfg)
        #
        # losses.update(add_prefix(loss_decode, 'decode'))

        #__________________________________________________#
        #构建第二个decode,
        #__________________________________________________#
        input_16=inputs[2]
        loss_refine, *loss_contrsative_list = self.refine_head.forward_train(img_refine, prev_features,input_16,  data_samples,)
        losses.update(add_prefix(loss_refine, 'refine'))

        j = 1
        for loss_aux in loss_contrsative_list:
            losses.update(add_prefix(loss_aux, 'aux_' + str(j)))
            j += 1
        return losses


        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    # def pre_process(self):

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:#模型先进的是这个loss函数
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # img_os2:将deeplabv3输入的图像size下采样为原来的一半
        if self.is_frequency:
            deeplab_inputs = self.lap_prymaid_conv.pyramid_decom(inputs)[0]
            img_os2 = nn.functional.interpolate(deeplab_inputs, size=[inputs.shape[-2]//self.down_scale, inputs.shape[-1]//self.down_scale])
        else:
            img_os2 = nn.functional.interpolate(inputs, size=[inputs.shape[-2]//self.down_scale, inputs.shape[-1]//self.down_scale])

        x = self.extract_feat(img_os2)#resnet18主干网络提取的特征
        if self.refine_input_ratio == 1.:
            img_refine = inputs
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(inputs, size=[int(inputs.shape[-2] * self.refine_input_ratio), int(inputs.shape[-1] * self.refine_input_ratio)])


        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_refine,data_samples)#包含了refine-aspp和isdhead的结果
        losses.update(loss_decode)




        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 7
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
