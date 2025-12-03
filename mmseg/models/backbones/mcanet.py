import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.models.utils.up_conv_block import UpConvBlock
from mmseg.registry import MODELS
from ..utils import resize
from .mit import MixVisionTransformer
from .unet import BasicConvBlock
from mmcv.cnn import ConvModule,build_upsample_layer
from mmengine.model.weight_init import constant_init

class AttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, return_attn_map, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg,sr=1):
        super().__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.return_attn_map = return_attn_map
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None
        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        if key_downsample is None and sr > 1:
            self.key_downsample = nn.Conv2d(
                in_channels=key_in_channels,
                out_channels=key_in_channels,
                kernel_size=sr,
                stride=sr
            )
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        if self.return_attn_map:
            return context,sim_map
        return context

class AttentionGate(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.proj1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1),nn.BatchNorm2d(in_channels))
        self.proj2 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1),nn.BatchNorm2d(in_channels))
        self.proj_final = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1),nn.BatchNorm2d(in_channels))

    def forward(self,attn1,attn2):
        return torch.sigmoid(
            self.proj_final(
                F.relu(
                    self.proj1(attn1) + self.proj2(attn2)
                    )
                )
            )

class MultiScaleCrossAttentionBlock(nn.Module):
    """
    Multi-Scale Cross Attention Block in decoder for MCANet.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 xl_channels,
                 out_channels,
                 global_intra_attn,
                 return_attn_map,
                 cross_attn_g,
                 cross_attn_l,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv')):
        super().__init__()

        self.global_intra_attn = global_intra_attn
        self.cross_attn_g = cross_attn_g
        self.cross_attn_l = cross_attn_l
        self.return_attn_map = return_attn_map

        self.ag = AttentionGate(xl_channels)
        self.proj_xm = ConvModule(in_channels=2 * xl_channels,out_channels=xl_channels,kernel_size=1)
        self.proj_f = ConvModule(in_channels=2 * xl_channels,out_channels=xl_channels,kernel_size=1)

        self.conv_block = conv_block(
            in_channels=2 * xl_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=xl_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.upsample = ConvModule(
                in_channels,
                xl_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, xl, x):
        """Forward function."""
        xg,attn_maps = None,None
        if self.return_attn_map:
            xg,attn_maps = self.global_intra_attn(x,x)
        else:
            xg = self.global_intra_attn(x,x)
        xg = self.upsample(xg)
        xf = self.cross_attn_g(xl,xg) + self.cross_attn_l(xg,xl)
        g = self.ag(xg,xl)
        xm = self.proj_xm(torch.cat([g * xg,(1 - g) * xl],dim=1))
        out = torch.cat([xf, xm], dim=1)
        out = self.conv_block(out)

        if self.return_attn_map:
            return out,attn_maps
        return out
        

@MODELS.register_module()
class MCANet(BaseModule):
    """
    MCANet backbone.

    Args:
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Default: 64.
        num_stages (int): Number of stages in encoder, normally 5. Default: 5.
        strides (Sequence[int 1 | 2]): Strides of each stage in encoder.
            len(strides) is equal to num_stages. Normally the stride of the
            first stage in encoder is 1. If strides[i]=2, it uses stride
            convolution to downsample in the correspondence encoder stage.
            Default: (1, 1, 1, 1, 1).
        dec_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence decoder stage.
            Default: (2, 2, 2, 2).
        downsamples (Sequence[int]): Whether use MaxPool to downsample the
            feature map after the first stage of encoder
            (stages: [1, num_stages)). If the correspondence encoder stage use
            stride convolution (strides[i]=2), it will never use MaxPool to
            downsample, even downsamples[i-1]=True.
            Default: (True, True, True, True).
        enc_dilations (Sequence[int]): Dilation rate of each stage in encoder.
            Default: (1, 1, 1, 1, 1).
        dec_dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Notice:
        The input image size should be divisible by the whole downsample rate
        of the encoder. More detail of the whole downsample rate can be found
        in MCANet._check_input_divisible.
    """

    def __init__(self,
                 num_stages=4,
                 return_attn_map=True,
                 global_intra_attn_flag=True,
                 local_inter_attn_flag=True,
                 multi_scale_cross_attn_flag=True,
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 dec_dilations=(1, 1, 1, 1),
                 encoder_channels=[64,128,320,512],
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_stages = num_stages
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.encoder_channels = encoder_channels
        self.return_attn_map = return_attn_map
        self.global_intra_attn_flag = global_intra_attn_flag
        self.local_inter_attn_flag = local_inter_attn_flag
        self.multi_scale_cross_attn_flag = multi_scale_cross_attn_flag

        # mit-b2
        self.encoder = MixVisionTransformer(
            in_channels=3,
            embed_dims=64,
            num_stages=4,
            num_layers=[3, 4, 6, 3],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            pretrained='checkpoints/mit_b2_20220624-66e8bf70.pth'
        )

        if self.local_inter_attn_flag:
            self.skip_input_proj = nn.ModuleList([
                ConvModule(
                        in_channels=encoder_channels[i],
                        out_channels=encoder_channels[0],
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                ) for i in range(num_stages)
            ])
            self.skip_output_proj = nn.ModuleList([
                ConvModule(
                        in_channels=encoder_channels[0] * num_stages,
                        out_channels=encoder_channels[i],
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                ) for i in range(num_stages)]
            )
            self.skip_output_conv = nn.ModuleList([
                ConvModule(
                        in_channels=encoder_channels[i],
                        out_channels=encoder_channels[i],
                        kernel_size=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                ) for i in range(num_stages)
            ])
            self.local_inter_attn_block = AttentionBlock(
                        key_in_channels=encoder_channels[0] * num_stages,
                        query_in_channels=encoder_channels[0] * num_stages,
                        channels=encoder_channels[0] * num_stages,
                        out_channels=encoder_channels[0] * num_stages,
                        return_attn_map=return_attn_map,
                        share_key_query=False,
                        query_downsample=None,
                        key_downsample=None,
                        key_query_num_convs=1,
                        value_out_num_convs=1,
                        key_query_norm=True,
                        value_out_norm=True,
                        matmul_norm=True,
                        with_out=False,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                    )
        
        self.decoder = nn.ModuleList()
        
        if multi_scale_cross_attn_flag:
            self.global_intra_attn_blocks = nn.ModuleList()
            self.cross_attn_g = nn.ModuleList()
            self.cross_attn_l = nn.ModuleList()
            for i in range(1,num_stages):
                self.global_intra_attn_blocks.append(
                    AttentionBlock(
                        key_in_channels=encoder_channels[i],
                        query_in_channels=encoder_channels[i],
                        channels=encoder_channels[i],
                        out_channels=encoder_channels[i],
                        return_attn_map=return_attn_map,
                        sr=2 ** (num_stages - i - 1),
                        share_key_query=False,
                        query_downsample=None,
                        key_downsample=None,
                        key_query_num_convs=1,
                        value_out_num_convs=1,
                        key_query_norm=True,
                        value_out_norm=True,
                        matmul_norm=True,
                        with_out=False,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                    )
                )
                self.cross_attn_l.append(
                    AttentionBlock(
                        key_in_channels=encoder_channels[i - 1],
                        query_in_channels=encoder_channels[i - 1],
                        channels=encoder_channels[i - 1],
                        out_channels=encoder_channels[i - 1],
                        return_attn_map=False,
                        sr=2 ** (num_stages - i),
                        share_key_query=False,
                        query_downsample=None,
                        key_downsample=None,
                        key_query_num_convs=1,
                        value_out_num_convs=1,
                        key_query_norm=True,
                        value_out_norm=True,
                        matmul_norm=True,
                        with_out=False,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                    )
                )
                self.cross_attn_g.append(
                    AttentionBlock(
                        key_in_channels=encoder_channels[i - 1],
                        query_in_channels=encoder_channels[i - 1],
                        channels=encoder_channels[i - 1],
                        out_channels=encoder_channels[i - 1],
                        return_attn_map=False,
                        sr=2 ** (num_stages - i),
                        share_key_query=False,
                        query_downsample=None,
                        key_downsample=None,
                        key_query_num_convs=1,
                        value_out_num_convs=1,
                        key_query_norm=True,
                        value_out_norm=True,
                        matmul_norm=True,
                        with_out=False,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg
                    )
                )
                self.decoder.append(
                    MultiScaleCrossAttentionBlock(
                        conv_block=BasicConvBlock,
                        in_channels=encoder_channels[i],
                        xl_channels=encoder_channels[i - 1],
                        out_channels=encoder_channels[i - 1],
                        global_intra_attn=self.global_intra_attn_blocks[i - 1],
                        return_attn_map=return_attn_map,
                        cross_attn_g=self.cross_attn_g[i - 1],
                        cross_attn_l=self.cross_attn_l[i - 1],
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg
                        )
                    )
        else:
            self.decoder.append(
                UpConvBlock(
                    conv_block=BasicConvBlock,
                    in_channels=encoder_channels[i],
                    skip_channels=encoder_channels[i - 1],
                    out_channels=encoder_channels[i - 1],
                    num_convs=dec_num_convs[i - 1],
                    stride=1,
                    dilation=dec_dilations[i - 1],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    upsample_cfg=upsample_cfg
                )
            )



    def local_inter_attntion(self,enc_out):
        enc_rs = [resize(enc_out[i],(64,32),mode='bilinear') for i in range(self.num_stages)]
        enc_proj = [self.skip_input_proj[i](enc_rs[i]) for i in range(self.num_stages)]
        enc_cat = torch.cat(enc_proj,dim=1)
        l,local_inter_attn_maps = None,None
        if self.return_attn_map:
            l,local_inter_attn_maps = self.local_inter_attn_block(enc_cat,enc_cat)
        else:
            l = self.local_inter_attn_block(enc_cat,enc_cat)
        skip_out_proj = [self.skip_output_proj[i](l) for i in range(self.num_stages)]
        skip_out_rs = [resize(skip_out_proj[i],size=enc_out[i].shape[-2:]) for i in range(self.num_stages)]
        skip_out = [self.skip_output_conv[i](skip_out_rs[i]) for i in range(self.num_stages)]
        return skip_out,local_inter_attn_maps


    def forward(self, x):
        self._check_input_divisible(x)
        enc_out = self.encoder(x)
        dec_out = [enc_out[-1]]
        x = enc_out[-1]
        global_intra_attn_maps = []
        local_inter_attn_maps = None

        if self.local_inter_attn_flag:
            enc_out,local_inter_attn_maps = self.local_inter_attntion(enc_out)

        for i in reversed(range(len(self.decoder))):
            if self.return_attn_map:
                x,global_attn_map = self.decoder[i](enc_out[i], x)
                global_intra_attn_maps.append(global_attn_map)
            else:
                x = self.decoder[i](enc_out[i],x)
            dec_out.append(x)

        if self.return_attn_map:
            return dec_out,local_inter_attn_maps,global_intra_attn_maps
        return dec_out

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, and downsamples '\
            f'is {self.downsamples}.'
