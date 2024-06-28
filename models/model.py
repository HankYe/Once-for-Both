# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from .vision_transformer import VisionTransformer, _cfg, MIMVisionTransformer
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from .layers import ModuleInjection, PatchEmbed, LayerNorm


__all__ = [
    'deit_tiny_patch16_224',
    'deit_small_patch16_224_mim', 'deit_small_patch16_224_finetune',
    'deit_base_patch16_224_mim', 'deit_base_patch16_224_finetune',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2

    @torch.jit.ignore
    def no_weight_decay(self):
        skip_list = ['pos_embed', 'cls_token', 'dist_token']
        return skip_list


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    ModuleInjection.method = kwargs.pop('method', 'full')
    ModuleInjection.searchable_modules = []
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.searchable_modules = ModuleInjection.searchable_modules
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224_mim(pretrained=False, mae=True, pretrained_strict=False, head_search=False, channel_search=False, **kwargs):
    ModuleInjection.method = kwargs.pop('method', 'full')
    ModuleInjection.searchable_modules = []
    model = MIMVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), embed_layer=PatchEmbed, mae=mae, head_search=head_search, channel_search=channel_search, **kwargs)
    model.searchable_modules = [m for m in model.modules() if hasattr(m, 'alpha')]
    model.default_cfg = _cfg()
    if pretrained:
        try:
            checkpoint = torch.load('deit_small_patch16_224-cd65a155.pth', map_location="cpu")
        except:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
        )
        if checkpoint["model"]['head.weight'].shape != model.head.weight.shape:
            checkpoint['model'].pop('head.weight')
            checkpoint['model'].pop('head.bias')
        if checkpoint["model"]['pos_embed'].shape != model.pos_embed.shape:
            checkpoint['model'].pop('pos_embed')
        model.load_state_dict(checkpoint["model"], pretrained_strict)
    return model

@register_model
def deit_small_patch16_224_finetune(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), embed_layer=PatchEmbed, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    ModuleInjection.method = kwargs.pop('method', 'full')
    ModuleInjection.searchable_modules = []
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.searchable_modules = [m for m in model.modules() if hasattr(m, 'alpha')]
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_mim(pretrained=False, mae=True, pretrained_strict=False, head_search=False, channel_search=False, **kwargs):
    ModuleInjection.method = kwargs.pop('method', 'full')
    ModuleInjection.searchable_modules = []
    model = MIMVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), embed_layer=PatchEmbed, mae=mae, head_search=head_search, channel_search=channel_search, **kwargs)
    # model.searchable_modules = ModuleInjection.searchable_modules
    model.searchable_modules = [m for m in model.modules() if hasattr(m, 'alpha')]
    model.default_cfg = _cfg()
    if pretrained:
        try:
            checkpoint = torch.load('deit_base_patch16_224-b5f2ef4d.pth', map_location="cpu")
        except:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
        if checkpoint["model"]['head.weight'].shape != model.head.weight.shape:
            checkpoint['model'].pop('head.weight')
            checkpoint['model'].pop('head.bias')
        if checkpoint["model"]['pos_embed'].shape != model.pos_embed.shape:
            checkpoint['model'].pop('pos_embed')
        model.load_state_dict(checkpoint["model"], pretrained_strict)
    return model


@register_model
def deit_base_patch16_224_finetune(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), embed_layer=PatchEmbed, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    ModuleInjection.method = kwargs.pop('method', 'full')
    ModuleInjection.searchable_modules = []
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.searchable_modules = ModuleInjection.searchable_modules
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    ModuleInjection.method = kwargs.pop('method', 'full')
    ModuleInjection.searchable_modules = []
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.searchable_modules = ModuleInjection.searchable_modules
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    ModuleInjection.method = kwargs.pop('method', 'full')
    ModuleInjection.searchable_modules = []
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.searchable_modules = ModuleInjection.searchable_modules
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(LayerNorm, eps=1e-6), embed_layer=PatchEmbed, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    ModuleInjection.method = kwargs.pop('method', 'full')
    ModuleInjection.searchable_modules = []
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.searchable_modules = ModuleInjection.searchable_modules
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    ModuleInjection.method = kwargs.pop('method', 'full')
    ModuleInjection.searchable_modules = []
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.searchable_modules = ModuleInjection.searchable_modules
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

def add_search_params(model, method):
    """Returns the requested model, ready for training/searching with the specified method.
    :param model: A deit model
    :param method: full or search
    :return: A deit model ready for training/searching
    """
    ModuleInjection.method = method
    ModuleInjection.searchable_modules = []
    model.searchable_modules = ModuleInjection.searchable_modules
    return model
