import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from models.pos_embed import get_2d_sincos_pos_embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

from .base_model import MAEBaseModel
from .layers import ModuleInjection, Attention, reduce_tensor, LayerNorm, Mlp, PatchEmbed

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        classifier=('head', 'head_dist')),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        classifier=('head', 'head_dist')),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        classifier=('head', 'head_dist')),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0, classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}


def norm_targets(targets, patch_size):
    assert patch_size % 2 == 1

    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.

    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                       count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                 count_include_pad=True) * (patch_size ** 2)

    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)

    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5

    return targets_


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_flops(self, num_patches):
        flops = 0
        dim = self.norm1.normalized_shape[0]
        flops += 2*dim*num_patches
        attn_flops = self.attn.get_flops(num_patches)
        flops += attn_flops
        mlp_flops = self.mlp.get_flops(num_patches)
        flops += mlp_flops
        return flops


class MAEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm, head_search=False, channel_search=False, attn_search=True, mlp_search=True):
        super().__init__()
        self.in_feature = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn = ModuleInjection.make_searchable_maeattn(self.attn, head_search, channel_search, attn_search)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = ModuleInjection.make_searchable_maemlp(self.mlp, mlp_search)

    def forward(self, input):
        x, weighted_mask_embed, weighted_embed = input

        self.weighted_mask_embed = weighted_mask_embed
        if weighted_mask_embed is not None and torch.sum(torch.multiply(weighted_mask_embed < 1,  weighted_mask_embed > 0)) != 0:
            x_reserved = x[..., weighted_mask_embed[0] > 0]
            x_dropped = x[..., weighted_mask_embed[0] <= 0]
            x = torch.cat([self.norm1(x_reserved), x_dropped], dim=-1)
            x = x + self.drop_path(self.attn(x, weighted_mask_embed, weighted_embed))
            x_reserved = x[..., weighted_mask_embed[0] > 0]
            x_dropped = x[..., weighted_mask_embed[0] <= 0]
            x = torch.cat([self.norm2(x_reserved), x_dropped], dim=-1)
            x = x + self.drop_path(self.mlp(x, weighted_mask_embed, weighted_embed))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), weighted_mask_embed, weighted_embed))
            x = x + self.drop_path(self.mlp(self.norm2(x), weighted_mask_embed, weighted_embed))
        return (x, weighted_mask_embed)

    def get_flops(self, num_patches, active_patches):
        flops = 0
        searched_flops = 0
        dim = self.in_feature
        active_dim = self.norm1.normalized_shape[0]
        flops += 2 * dim * num_patches
        searched_flops += 2 * active_dim * active_patches
        attn_flops, attn_searched_flops = self.attn.get_flops(num_patches, active_patches)
        flops += attn_flops
        searched_flops += attn_searched_flops
        mlp_flops, mlp_searched_flops = self.mlp.get_flops(num_patches, active_patches)
        flops += mlp_flops
        searched_flops += mlp_searched_flops
        return flops, searched_flops

class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 distilled=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed,
                 norm_layer=None, act_layer=None, weight_init=''):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        skip_list = ['pos_embed', 'cls_token', 'dist_token']
        return skip_list

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x

    def get_flops(self):

        patch_size = self.patch_embed.patch_size[0]
        num_patch = self.patch_embed.num_patches
        patch_embed_flops = num_patch*self.patch_embed.proj.out_channels*3*(patch_size**2)

        blocks_flops = 0
        for block in self.blocks:
            block_flops = block.get_flops(num_patch)
            blocks_flops += block_flops

        if self.head_dist:
            head_flops = 2*self.patch_embed.proj.out_channels*self.num_classes
        else:
            head_flops = self.patch_embed.proj.out_channels*self.num_classes

        total_flops = patch_embed_flops+blocks_flops+head_flops
        return total_flops


class MIMVisionTransformer(MAEBaseModel):
    """ 
    Vision Transformer with MAE
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 distilled=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', head_search=False, channel_search=False, attn_search=True,
                 mlp_search=True, embed_search=True, patch_search=True, mae=True, norm_pix_loss=False, mask_ratio=1.0):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            head_search: (bool): search for head number dimension
            channel_search: (bool): search for the QKV channel dimension in attn blocks
            attn_search: (bool): search for attn block dimension
            mlp_search: (bool): search for the mlp channel dimension
            embed_search: (bool): search for the patch embedding channel dimension
            patch_search: (bool): search for the masking ratio of patch number
            mae: (bool): training model with MAE strategy (decoding the masked patches)
            norm_pix_loss: (bool): normalize the reconstructed pixels for the loss computation
            mask_ratio: (bool): constant masking ratio if not searching the masking ratio
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.finish_search = False
        self.execute_prune = False
        self.fused = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed = ModuleInjection.make_searchable_patchembed(self.patch_embed, embed_search)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            MAEBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                head_search=head_search, channel_search=channel_search, attn_search=attn_search, mlp_search=mlp_search)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # --------------------------------------------------------------------------
        # search space setting of Patch number
        self.mae = mae
        if patch_search:
            self.patch_ratio_list = np.linspace(0.5, 1.0, 5).tolist()
            self.alpha_patch = nn.Parameter(torch.rand(1, len(self.patch_ratio_list)))
            self.switch_cell_patch = self.alpha_patch > 0
            self.patch_search_mask = torch.zeros(len(self.patch_ratio_list), 1, self.num_patches, 1)
            for i, r in enumerate(self.patch_ratio_list):
                patch_keep = int(self.num_patches * r)
                self.patch_search_mask[i, :, :patch_keep, :] = 1
        else:
            self.patch_ratio_list = [mask_ratio]
            self.alpha_patch = nn.Parameter(torch.tensor([[1.]]))
            self.switch_cell_patch = self.alpha_patch > 0
            self.patch_search_mask = torch.zeros(len(self.patch_ratio_list), 1, self.num_patches, 1)
            for i, r in enumerate(self.patch_ratio_list):
                patch_keep = int(self.num_patches * r)
                self.patch_search_mask[i, :, :patch_keep, :] = 1
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        if self.mae:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.decoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.num_features,
                    out_channels=patch_size ** 2 * 3, kernel_size=1),
                nn.PixelShuffle(patch_size),
            )
            self.norm_pix_loss = norm_pix_loss
        else: self.mask_token = None
        # --------------------------------------------------------------------------
        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            trunc_normal_(self.cls_token, std=.02)
            if self.mae:
                trunc_normal_(self.mask_token, std=.02)
            self.apply(_init_vit_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def adjust_masking_ratio(self, epoch, warmup_epochs, total_epochs, min_ratio=0.75, max_ratio=0.95, method='linear'):
        if epoch <= warmup_epochs:
            self.patch_ratio_list = [max_ratio - (max_ratio - min_ratio) * epoch / warmup_epochs]

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        skip_list = ['pos_embed', 'cls_token', 'dist_token', 'scale_weight', 'mask_token', 'score']
        return skip_list

    def freeze_decoder(self):
        if self.mask_token is not None:
            self.mask_token.requires_grad = False
        for name, p in self.named_parameters():
            if 'decoder' in name:
                p.requires_grad = False

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def reset_mask_ratio(self, mask_ratio):
        self.patch_ratio_list = [mask_ratio]

    def correct_require_grad(self, w_head, w_mlp, w_patch, w_embedding):
        if w_head == 0:
            for l_block in self.searchable_modules:
                if hasattr(l_block, 'num_heads'):
                    l_block.alpha.requires_grad = False
        if w_embedding == 0:
            for l_block in self.searchable_modules:
                if hasattr(l_block, 'embed_ratio_list'):
                    l_block.alpha.requires_grad = False
        if w_mlp == 0:
            for l_block in self.searchable_modules:
                if not hasattr(l_block, 'num_heads') and not hasattr(l_block, 'embed_ratio_list'):
                    l_block.alpha.requires_grad = False
        if w_patch == 0:
            self.alpha_patch.requires_grad = False

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def patch_masking(self, x):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keeps = [int(L * r) for index, r in enumerate(self.patch_ratio_list) if self.switch_cell_patch[:, index]]

        if len_keeps != [L]:
            
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keeps[0]] = 0
            
            mask = torch.gather(mask, dim=1, index=ids_restore)
            x_masked = x * (1 - mask).unsqueeze(-1)
            return x_masked, mask
            # TODO: learnable patch masking
        else:
            return x, None

    def forward_features(self, x):
        x = self.patch_embed(x)
        
        if not self.patch_embed.finish_search:
            mask_restore, embed_score = self.patch_embed.get_weight()
            weighted_embedding = (1 - self.patch_embed.w_p) * mask_restore + self.patch_embed.w_p * embed_score
            weighted_mask_embedding = mask_restore
        elif not self.fused:
            weighted_embedding, weighted_mask_embedding = self.patch_embed.score, self.patch_embed.weighted_mask
        else:
            weighted_embedding, weighted_mask_embedding = None, self.patch_embed.weighted_mask
        
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # add pos embed w/o cls token
        x += (self.pos_embed[:, self.num_tokens:, :] * weighted_embedding) if weighted_embedding is not None else self.pos_embed[:, self.num_tokens:, :]

        if self.training:
            # masking: length -> length * mask_ratio
            x, mask = self.patch_masking(x)
            if self.mask_token is not None and mask is not None:
                if weighted_embedding is None:
                    x += mask.unsqueeze(-1) * self.mask_token.expand_as(x)
                else:
                    x += mask.unsqueeze(-1) * self.mask_token.expand_as(x) * weighted_embedding
        else: mask = None

        if isinstance(x, tuple):
            x, score_sorted, weight_sorted = x
        else: score_sorted, weight_sorted = None, None

        # append cls token
        if weighted_embedding is not None:
            cls_token = (self.cls_token + self.pos_embed[:, :1, :]) * weighted_embedding
        else:
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            if weighted_embedding is not None:
                dist_token = (self.dist_token + self.pos_embed[:, 1:self.num_tokens, :]) * weighted_embedding
            else:
                dist_token = self.dist_token + self.pos_embed[:, 1:self.num_tokens, :]
            dist_token = dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        x = self.pos_drop(x)
        for block in self.blocks:
            x, _ = block((x, weighted_mask_embedding, weighted_embedding))

        if not self.patch_embed.finish_search:
            x_reserved = x[..., weighted_mask_embedding[0] > 0]
            x_dropped = x[..., weighted_mask_embedding[0] <= 0]
            x = torch.cat([self.norm(x_reserved), x_dropped * weighted_mask_embedding[..., weighted_mask_embedding <= 0]], dim=-1)
        else:
            x = self.norm(x)
        return x, mask, score_sorted, weight_sorted

    def forward_decoder(self, x, ids_restore, mask):
        if isinstance(x, list):
            x = sum([w_s * x_s for w_s, x_s in zip(self.scale_weight, x)])
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + self.num_tokens - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :] if self.dist_token is None else x[:, 2:, :], mask_tokens], dim=1)  # no cls or distill token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :] if self.dist_token is None else x[:, :2, :], x_], dim=1)  # append cls and distill token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :] if self.dist_token is None else x[:, 2:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum() # mean loss on removed patches
        return loss

    def forward(self, imgs):
        latent, mask, score_sorted, _ = self.forward_features(imgs)
        if self.mae and mask is not None:
            z = latent[:, 1:, :] if self.head_dist is None else latent[:, 2:, :]
            B, L, C = z.shape
            H = W = int(L ** 0.5)
            x_rec = self.decoder(z.transpose(1, 2).contiguous().reshape(B, C, H, W))
            mask = mask.view(B, H, W)
            mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
            # norm target as prompted
            targets = norm_targets(imgs, 47)
            decoder_loss = F.l1_loss(targets, x_rec, reduction='none')
            decoder_loss = (decoder_loss * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
            
        else: decoder_loss = 0.
        if score_sorted is not None:
            score_loss = torch.sum(score_sorted) / score_sorted.shape[0] * 1e-4
        else:
            score_loss = None
        if self.head_dist is not None:
            x, x_dist = self.head(latent[:, 0, :]), self.head_dist(latent[:, 1, :])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return (x, x_dist), decoder_loss
            else:
                return (x + x_dist) / 2, decoder_loss
        else:
            x = self.head(self.pre_logits(latent[:, 0, :]))
        return x, (decoder_loss, score_loss)

    def fuse(self):
        assert self.finish_search == True
        self.fused = True
        weighted_embedding = self.patch_embed.score
        torch.cuda.synchronize()
        self.mask_token = nn.Parameter(self.mask_token.data.clone() * weighted_embedding.data.clone().unsqueeze(-2)) if self.mask_token is not None else None
        self.cls_token = nn.Parameter(self.cls_token.data.clone() * weighted_embedding.data.clone().unsqueeze(-2))
        self.dist_token = nn.Parameter(self.dist_token.data.clone() * weighted_embedding.data.clone().unsqueeze(-2)) if self.dist_token is not None else None
        self.pos_embed = nn.Parameter(self.pos_embed.data.clone() * weighted_embedding.data.clone().unsqueeze(-2))
        for m in self.searchable_modules:
            m.fuse()

    def get_flops(self):
        patch_size = self.patch_embed.patch_size[0]
        num_patch = self.patch_embed.num_patches
        patch_embed_flops = num_patch * self.embed_dim * 3 * (patch_size ** 2)
        active_embed = self.patch_embed.weighted_mask.sum()
        patch_embed_flops_searched = num_patch * active_embed * 3 * (patch_size ** 2)
        
        blocks_flops = 0
        blocks_flops_searched = 0
        active_patches = self.weighted_mask.sum() if hasattr(self, 'weighted_mask') else num_patch
        for block in self.blocks:
            block_flops, block_flops_searched = block.get_flops(num_patch, active_patches)
            blocks_flops += block_flops
            blocks_flops_searched += block_flops_searched

        if self.head_dist:
            head_flops = 2 * self.embed_dim * self.num_classes
            head_flops_searched = 2 * active_embed * self.num_classes
        else:
            head_flops = self.embed_dim * self.num_classes
            head_flops_searched = active_embed * self.num_classes

        total_flops = patch_embed_flops + blocks_flops + head_flops
        searched_flops = patch_embed_flops_searched + blocks_flops_searched + head_flops_searched
        return total_flops / 1e9, searched_flops / 1e9

    def compress(self, thresh=0.2, optimizer_params=None, optimizer_decoder=None, optimizer_archs=None):
        """compress the network to make alpha exactly 1 and 0"""

        # compress the patch number
        execute_prune_patch = False
        if self.switch_cell_patch.sum() == 1:
            finish_search_patch = True
            self.alpha_patch.requires_grad = False
        else:
            finish_search_patch = False
            torch.cuda.synchronize()
            alpha_reduced = reduce_tensor(self.alpha_patch)
            # print(f'--Reduced Patch Alpha: {alpha_reduced}--')
            alpha_norm = torch.softmax(alpha_reduced[self.switch_cell_patch].view(-1), dim=0).detach()
            thresh_patch = thresh / self.switch_cell_patch.sum()
            min_alpha = torch.min(alpha_norm)
            if min_alpha <= thresh_patch:
                print(f'--Patch Alpha: {self.alpha_patch}--')
                execute_prune_patch = True
                alpha = alpha_reduced.detach() - torch.where(self.switch_cell_patch.to(alpha_reduced.device), torch.zeros_like(alpha_reduced), torch.ones_like(alpha_reduced) * float('inf')).to(alpha_reduced.device)
                alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(alpha)
                self.switch_cell_patch = (alpha > thresh_patch).detach()
                self.alpha_patch = torch.nn.Parameter(torch.where(self.switch_cell_patch, alpha_reduced, torch.zeros_like(alpha).to(self.alpha_patch.device)))
                print(f'---Normalized Alpha: {alpha_norm}---')
                print(f'------Prune patch: {(alpha_norm <= thresh_patch).sum()} cells------')
                alpha = self.alpha_patch - torch.where(self.switch_cell_patch, torch.zeros_like(self.alpha_patch),
                                                       torch.ones_like(self.alpha_patch) * float('inf')).to(self.alpha_patch.device)
                alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(self.alpha_patch)
                self.weighted_mask = sum(alpha[i, j] * self.patch_search_mask[j, ...].to(alpha.device)
                                         for i, j in product(range(alpha.size(0)), range(alpha.size(1)))
                                         if self.switch_cell_patch[i][j])
                print(f'---Updated Weighted Mask of Patch Dimension: {self.weighted_mask}---')
                if self.switch_cell_patch.sum() == 1:
                    finish_search_patch = True
                    self.alpha_patch.requires_grad = False
                    self.weighted_mask = self.weighted_mask.detach()

        # compress other dimensions
        if self.searchable_modules == []:
            self.searchable_modules = [m for m in self.modules() if hasattr(m, 'alpha')]

        finish_search_embedding = False
        execute_prune_embedding = False
        keep_index = None
        for l_block in self.searchable_modules:
            if hasattr(l_block, 'embed_ratio_list'):
                torch.cuda.synchronize()
                keep_index, optimizer_params, optimizer_decoder, optimizer_archs = l_block.compress(thresh, optimizer_params, 
                                                                                                    optimizer_decoder, optimizer_archs, 'patch_embed')
                torch.cuda.synchronize()
                finish_search_embedding = l_block.finish_search
                execute_prune_embedding = l_block.execute_prune
                if (finish_search_embedding and execute_prune_embedding) or keep_index is not None:
                    assert keep_index is not None
                    ori_mask_token = self.mask_token if self.mask_token is not None else None
                    ori_cls_token = self.cls_token
                    ori_dist_token = self.dist_token if self.dist_token is not None else None
                    ori_pos_embed = self.pos_embed
                    torch.cuda.synchronize()
                    self.mask_token = nn.Parameter(self.mask_token.data.clone()[..., keep_index]) if self.mask_token is not None else None
                    self.cls_token = nn.Parameter(self.cls_token.data.clone()[..., keep_index])
                    self.dist_token = nn.Parameter(self.dist_token.data.clone()[..., keep_index]) if self.dist_token is not None else None
                    self.pos_embed = nn.Parameter(self.pos_embed.data.clone()[..., keep_index])

                    if optimizer_params is not None:
                        if self.mask_token is not None:
                            optimizer_params.update(ori_mask_token, self.mask_token, 'mask_token', 0, keep_index, dim=-1)
                        if self.dist_token is not None:
                            optimizer_params.update(ori_dist_token, self.dist_token, 'dist_token', 0, keep_index, dim=-1)
                        optimizer_params.update(ori_cls_token, self.cls_token, 'cls_token', 0, keep_index, dim=-1)
                        optimizer_params.update(ori_pos_embed, self.pos_embed, 'pos_embed', 0, keep_index, dim=-1)

                    ori_norm_weight = self.norm.weight
                    ori_norm_bias = self.norm.bias
                    self.norm.normalized_shape[0] = len(keep_index)
                    torch.cuda.synchronize()
                    self.norm.weight = torch.nn.Parameter(self.norm.weight.data.clone()[keep_index])
                    self.norm.bias = torch.nn.Parameter(self.norm.bias.data.clone()[keep_index])

                    if optimizer_params is not None:
                        optimizer_params.update(ori_norm_weight, self.norm.weight, 'norm.weight', 0, keep_index, dim=-1)
                        optimizer_params.update(ori_norm_bias, self.norm.bias, 'norm.bias', 0, keep_index, dim=-1)

                    for idx, block in enumerate(self.blocks):
                        ori_block_norm1_weight = block.norm1.weight
                        ori_block_norm1_bias = block.norm1.bias
                        ori_block_norm2_weight = block.norm2.weight
                        ori_block_norm2_bias = block.norm2.bias
                        block.norm1.normalized_shape[0] = len(keep_index)
                        torch.cuda.synchronize()
                        block.norm1.weight = torch.nn.Parameter(block.norm1.weight.data.clone()[keep_index])
                        block.norm1.bias = torch.nn.Parameter(block.norm1.bias.data.clone()[keep_index])
                        block.norm2.normalized_shape[0] = len(keep_index)
                        block.norm2.weight = torch.nn.Parameter(block.norm2.weight.data.clone()[keep_index])
                        block.norm2.bias = torch.nn.Parameter(block.norm2.bias.data.clone()[keep_index])
                        if optimizer_params is not None:
                            optimizer_params.update(ori_block_norm1_weight, block.norm1.weight, f'blocks.{idx}.norm1.weight', 0, keep_index, dim=-1)
                            optimizer_params.update(ori_block_norm1_bias, block.norm1.bias, f'blocks.{idx}.norm1.bias', 0, keep_index, dim=-1)
                            optimizer_params.update(ori_block_norm2_weight, block.norm2.weight, f'blocks.{idx}.norm2.weight', 0, keep_index, dim=-1)
                            optimizer_params.update(ori_block_norm2_bias, block.norm2.bias, f'blocks.{idx}.norm2.bias', 0, keep_index, dim=-1)
                    if not isinstance(self.pre_logits, nn.Identity):
                        ori_fc_weight = self.pre_logits.fc.weight
                        self.pre_logits.fc.in_features = len(keep_index)
                        torch.cuda.synchronize()
                        self.pre_logits.fc.weight = torch.nn.Parameter(self.pre_logits.fc.weight.data.clone()[:, keep_index])
                        if optimizer_params is not None:
                            optimizer_params.update(ori_fc_weight, self.pre_logits.fc.weight, 'pre_logits.fc.weight', 1, keep_index, dim=-1)
                    if isinstance(self.head, nn.Linear):
                        ori_head_weight = self.head.weight
                        self.head.in_features = len(keep_index)
                        torch.cuda.synchronize()
                        self.head.weight = torch.nn.Parameter(self.head.weight.data.clone()[:, keep_index])
                        if optimizer_params is not None:
                            optimizer_params.update(ori_head_weight, self.head.weight, 'head.weight', 1, keep_index, dim=-1)

                    if isinstance(self.head_dist, nn.Linear):
                        ori_head_dist_weight = self.head_dist.weight
                        self.head_dist.in_features = len(keep_index)
                        torch.cuda.synchronize()
                        self.head_dist.weight = torch.nn.Parameter(self.head_dist.weight.data.clone()[:, keep_index])
                        if optimizer_params is not None:
                            optimizer_params.update(ori_head_dist_weight, self.head_dist.weight, 'head_dist.weight', 1, keep_index, dim=-1)

                    if self.mae:
                        ori_decoder_weight = self.decoder[0].weight
                        self.decoder[0].in_channels = len(keep_index)
                        torch.cuda.synchronize()
                        self.decoder[0].weight = nn.Parameter(self.decoder[0].weight.data.clone()[:, keep_index, ...])
                        if optimizer_decoder is not None:
                            optimizer_decoder.update(ori_decoder_weight, self.decoder[0].weight, 'decoder.0.weight', 1, keep_index, dim=1)
                break

        self.finish_search = finish_search_patch and finish_search_embedding
        self.execute_prune = execute_prune_patch or execute_prune_embedding
        module_name_list = list(dict(self.named_parameters()).keys())
        module_value_list = list((id(p) for p in dict(self.named_parameters()).values()))
        for l_block in self.searchable_modules:
            finish_search_block = l_block.finish_search
            execute_prune_block = l_block.execute_prune

            id_block = id(l_block.alpha)
            block_name = module_name_list[module_value_list.index(id_block)][:-6]
            if hasattr(l_block, 'num_heads'):
                if (not finish_search_block) or execute_prune_block:
                    torch.cuda.synchronize()
                    optimizer_params, optimizer_decoder, optimizer_archs = l_block.compress(thresh, optimizer_params, 
                                                                                            optimizer_decoder, optimizer_archs, block_name)
                torch.cuda.synchronize()
                if (finish_search_embedding and execute_prune_embedding) or keep_index is not None:
                    optimizer_params, optimizer_decoder, optimizer_archs = l_block.compress_patchembed(keep_index, optimizer_params, 
                                                                                                       optimizer_decoder, optimizer_archs, block_name)
            elif hasattr(l_block, 'embed_ratio_list'):
                continue
            else:
                if (not finish_search_block) or execute_prune_block:
                    torch.cuda.synchronize()
                    optimizer_params, optimizer_decoder, optimizer_archs = l_block.compress(thresh, optimizer_params, 
                                                                                            optimizer_decoder, optimizer_archs, block_name)
                torch.cuda.synchronize()
                if (finish_search_embedding and execute_prune_embedding) or keep_index is not None:
                    optimizer_params, optimizer_decoder, optimizer_archs = l_block.compress_patchembed(keep_index, optimizer_params, 
                                                                                                       optimizer_decoder, optimizer_archs, block_name)
            self.finish_search &= l_block.finish_search
            self.execute_prune |= l_block.execute_prune
        torch.cuda.synchronize()
        return self.finish_search, self.execute_prune, optimizer_params, optimizer_decoder, optimizer_archs


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2).contiguous()
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).contiguous().reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed, getattr(model, 'num_tokens', 1))
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, mae=False, pretrained_strict=True, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        MIMVisionTransformer if mae else VisionTransformer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=pretrained_strict,
        **kwargs)
    
    return model


@register_model
def vit_small_patch16_224(pretrained=False, mae=False, **kwargs):
    """ My custom 'small' ViT model. embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.
    NOTE:
        * this differs from the DeiT based 'small' definitions with embed_dim=384, depth=12, num_heads=6
        * this model does not have a bias for QKV (unlike the official ViT and DeiT models)
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3.,
        qkv_bias=False, norm_layer=nn.LayerNorm, **kwargs)
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        model_kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, mae=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224(pretrained=False, mae=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, mae=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_base_patch32_384(pretrained=False, mae=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, mae=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224(pretrained=False, mae=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384(pretrained=False, mae=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_large_patch32_384(pretrained=False, mae=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, mae=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, mae=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=768, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, mae=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, mae=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, mae=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_deit_tiny_patch16_224(pretrained=False, mae=False, **kwargs):
    """ DeiT-tiny model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_deit_tiny_patch16_224', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_deit_small_patch16_224(pretrained=False, mae=False, **kwargs):
    """ DeiT-small model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_deit_small_patch16_224', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_deit_base_patch16_224(pretrained=False, mae=False, **kwargs):
    """ DeiT base model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_deit_base_patch16_224', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_deit_base_patch16_384(pretrained=False, mae=False, **kwargs):
    """ DeiT base model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_deit_base_patch16_384', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_deit_tiny_distilled_patch16_224(pretrained=False, mae=False, **kwargs):
    """ DeiT-tiny distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_tiny_distilled_patch16_224', pretrained=pretrained, mae=mae,  distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_small_distilled_patch16_224(pretrained=False, mae=False, **kwargs):
    """ DeiT-small distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_small_distilled_patch16_224', pretrained=pretrained, mae=mae,  distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_base_distilled_patch16_224(pretrained=False, mae=False, **kwargs):
    """ DeiT-base distilled model @ 224x224 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_base_distilled_patch16_224', pretrained=pretrained, mae=mae,  distilled=True, **model_kwargs)
    return model


@register_model
def vit_deit_base_distilled_patch16_384(pretrained=False, mae=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'vit_deit_base_distilled_patch16_384', pretrained=pretrained, mae=mae, distilled=True, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, mae=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, mae=mae, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil(pretrained=False, mae=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, mae=mae, **model_kwargs)
    return model