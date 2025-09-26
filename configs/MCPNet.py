

_base_ = [
    '../_base_/models/isdnet_r50-d8.py', '../_base_/datasets/deepglobe.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (1224, 1224)
data_preprocessor = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    size=crop_size )
model = dict(
    type='EncoderDecoderisdnet',
    down_scale=4,
    backbone=dict(
        type='VSSM',
        patch_size=4,
        in_chans=3,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        vssm_mode='vssm',
        pretrain_model="/media/cm/D450B76750B74ECC/MPNet/checkpoint/vmamba_tiny_e292.pth",
       ),
    decode_head=[
        dict(
            type='RefineASPPHead',
            in_channels=768,#将输入改称768
            in_index=3,
            channels=128,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=7,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='ISDHead',

            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            channels=128,
            num_classes=7,
            dropout_ratio=0.1,
            fusion_mode='raf',#pag_fusion;raf;cat_fusion;cat_fusion
            model_cls='ShallowNet',#mamba;ShallowNet
            dims=[12, 24, 48, 96],
            depths=[1, 1, 2, 1],
            shallow_model_inchan=3,
            lap=False,
            consist=False,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),

        ),
    ],

    auxiliary_head=dict(in_channels=384, channels=64, num_classes=7),
    data_preprocessor=data_preprocessor)
