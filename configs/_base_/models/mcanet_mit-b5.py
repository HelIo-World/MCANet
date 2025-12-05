# model settings
norm_cfg = dict(type='BN', requires_grad=True)

global_intra_attn_flag = True
local_inter_attn_flag = True

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='MCANet',
        encoder_channels=[64,128,320,512],
        num_stages=4,
        return_attn_map=True,
        global_intra_attn_flag=global_intra_attn_flag,
        local_inter_attn_flag=local_inter_attn_flag,
        multi_scale_cross_attn_flag=True,
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='MCANetHead',
        in_channels=64,
        in_index=3,
        channels=16,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,   
        global_intra_attn_flag=global_intra_attn_flag,
        local_inter_attn_flag=local_inter_attn_flag,
        receive_attn_map=True,
        label_downsample_rate=1/16,
        ignore_index=None,
        lambda_l=0.1,
        lambda_g=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=256, stride=170))
