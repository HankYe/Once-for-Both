import torch
import torch.nn as nn
from itertools import product
from timm.models.layers.helpers import to_2tuple
from torch.nn import functional as F
from timm.models.layers import trunc_normal_


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.barrier()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt


class LayerNorm(nn.Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = nn.LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = nn.LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)
    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True) -> None:
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape,]
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = torch.nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.norm_layer = norm_layer

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x


class MAEPatchEmbed(PatchEmbed):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patchmodule, embed_search=True):
        super().__init__(patchmodule.img_size[0], patchmodule.patch_size[0], patchmodule.proj.in_channels, 
                         patchmodule.proj.out_channels, patchmodule.norm_layer)
        self.finish_search = False
        self.execute_prune = False
        self.fused = False
        embed_dim = patchmodule.proj.out_channels
        if embed_search:
            self.embed_ratio_list = [i / embed_dim
                                     for i in range(embed_dim // 2,
                                                    embed_dim + 1,
                                                    min(embed_dim // 32, 12))]
            self.alpha = nn.Parameter(torch.rand(1, len(self.embed_ratio_list)))

            self.switch_cell = self.alpha > 0
            embed_mask = torch.zeros(len(self.embed_ratio_list), embed_dim)  # -1, H, 1, d(1)
            for i, r in enumerate(self.embed_ratio_list):
                embed_mask[i, :int(r * embed_dim)] = 1
            self.mask = embed_mask
            self.score = nn.Parameter(torch.rand(1, embed_dim))
            trunc_normal_(self.score, std=.2)
        else:
            self.embed_ratio_list = [1.0]
            embed_mask = torch.zeros(len(self.embed_ratio_list), embed_dim)
            self.alpha = nn.Parameter(torch.tensor([1.]))
            self.switch_cell = self.alpha > 0
            for i, r in enumerate(self.embed_ratio_list):
                embed_mask[i, :int(r * embed_dim)] = 1
            self.weighted_mask = self.mask = embed_mask
            self.score = torch.ones(1, embed_dim)
            self.finish_search = True
        self.embed_dim = embed_dim
        self.w_p = 0.99

    def update_w(self, cur_epoch, warmup_epochs, max=0.99, min=0.1):
        if cur_epoch <= warmup_epochs:
            self.w_p = (min - max) / warmup_epochs * cur_epoch + max

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()
        if not self.finish_search:
            alpha = self.alpha - torch.where(self.switch_cell.to(self.alpha.device), torch.zeros_like(self.alpha),
                                             torch.ones_like(self.alpha) * float('inf'))
            alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(self.alpha)
            self.weighted_mask = sum(
                alpha[i][j] * self.mask[j, :].to(alpha.device) for i, j in product(range(alpha.size(0)), range(alpha.size(1)))
                if self.switch_cell[i][j]).unsqueeze(-2)  # 1, d

            ids_shuffle_channel = torch.argsort(self.score, dim=-1,
                                                descending=True)  # descend: large is keep, small is remove
            ids_restore_channel = torch.argsort(ids_shuffle_channel, dim=-1)
            prob_score = self.score.sigmoid()
            weight_restore = torch.gather(self.weighted_mask, dim=-1, index=ids_restore_channel)
            x *= self.w_p * prob_score + (1 - self.w_p) * weight_restore
            x_reserved = x[..., weight_restore[0] > 0]
            x_dropped = x[..., weight_restore[0] <= 0]

            x = torch.cat([self.norm(x_reserved), x_dropped * weight_restore[..., weight_restore <= 0]], dim=-1)
        elif not self.fused:
            x = self.norm(x * self.score)
        else:
            x = self.norm(x)
        return x

    def fuse(self):
        self.fused = True
        self.score.requires_grad = False
        self.proj.weight = torch.nn.Parameter(self.proj.weight.data.clone() * self.score.data.clone().squeeze().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        self.proj.bias = torch.nn.Parameter(self.proj.bias.data.clone() * self.score.data.clone().squeeze())
        
    def get_alpha(self):
        return self.alpha, self.switch_cell.to(self.alpha.device)

    def get_weight(self):
        ids_shuffle_channel = torch.argsort(self.score, dim=-1, descending=True)  # descend: large is keep, small is remove
        ids_restore_channel = torch.argsort(ids_shuffle_channel, dim=-1)
        prob_score = self.score.sigmoid()
        weight_restore = torch.gather(self.weighted_mask, dim=-1, index=ids_restore_channel)
        return weight_restore, prob_score

    def compress(self, thresh, optimizer_params, optimizer_decoder, optimizer_archs, prefix=''):
        if self.switch_cell.sum() == 1:
            self.finish_search = True
            self.execute_prune = False
            self.alpha.requires_grad = False
        else:
            torch.cuda.synchronize()
            try:
                alpha_reduced = reduce_tensor(self.alpha.data)
            except: alpha_reduced = self.alpha.data
            torch.cuda.synchronize()
            # print(f'--Reduced Embed Alpha: {alpha_reduced}--')
            alpha_norm = torch.softmax(alpha_reduced[self.switch_cell].view(-1), dim=0).detach()
            threshold = thresh / self.switch_cell.sum()
            min_alpha = torch.min(alpha_norm)
            if min_alpha <= threshold:
                print(f'--Embed Alpha: {alpha_reduced}--')
                self.execute_prune = True
                alpha = alpha_reduced.detach() - torch.where(self.switch_cell.to(alpha_reduced.device), 
                                                             torch.zeros_like(alpha_reduced), 
                                                             torch.ones_like(alpha_reduced) * float('inf')).to(alpha_reduced.device)
                alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(alpha)
                self.switch_cell = (alpha > threshold).detach()
                ori_alpha = self.alpha
                torch.cuda.synchronize()
                self.alpha = nn.Parameter(torch.where(self.switch_cell, alpha_reduced, torch.zeros_like(alpha).to(self.alpha.device)))
                if optimizer_archs is not None:
                    torch.cuda.synchronize()
                    optimizer_archs.update(ori_alpha, self.alpha, '.'.join([prefix, 'alpha']), 0, 
                                           torch.arange(self.alpha.shape[-1]).to(self.alpha.device), dim=-1, initialize=True)

                alpha = self.alpha - torch.where(self.switch_cell, torch.zeros_like(self.alpha),
                                                 torch.ones_like(self.alpha) * float('inf')).to(self.alpha.device)
                alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(alpha)
                self.weighted_mask = sum(alpha[i][j] * self.mask[j, :].to(alpha.device) 
                                         for i, j in product(range(alpha.size(0)), range(alpha.size(1)))
                                         if self.switch_cell[i][j]).unsqueeze(-2)  # 1, d
                print(f'---Normalized Alpha: {alpha_norm}---')
                print(f'------Prune {self}: {(alpha_norm <= threshold).sum()} cells------')
                print(f'---Updated Weighted Mask of Patch Embed Dimension: {self.weighted_mask}---')
                if self.switch_cell.sum() == 1:
                    self.finish_search = True
                    self.alpha.requires_grad = False
                    self.weighted_mask = self.weighted_mask.detach()
                    if optimizer_archs is not None:
                        torch.cuda.synchronize()
                        optimizer_archs.update(self.alpha, self.alpha, '.'.join([prefix, 'alpha']), 0, None, dim=-1)
                    index = torch.nonzero(self.switch_cell)
                    assert index.shape[0] == 1
                    self.proj.out_channels = int(self.embed_ratio_list[index[0, 1]] * self.embed_dim)
                    channel_index = torch.argsort(self.score, dim=1, descending=True)[:, :self.proj.out_channels]
                    keep_index = channel_index.reshape(-1).detach()
                    ori_score = self.score
                    ori_proj_weight = self.proj.weight
                    ori_proj_bias = self.proj.bias
                    self.weighted_mask = self.weighted_mask[:, :len(keep_index)]
                    torch.cuda.synchronize()
                    self.score = nn.Parameter(self.w_p * self.score.sigmoid().data.clone()[:, keep_index] + (1 - self.w_p) * self.weighted_mask.data.clone())
                    self.proj.weight = torch.nn.Parameter(self.proj.weight.data.clone()[keep_index, ...])
                    self.proj.bias = torch.nn.Parameter(self.proj.bias.data.clone()[keep_index])
                    if optimizer_params is not None:
                        torch.cuda.synchronize()
                        optimizer_params.update(ori_score, self.score, '.'.join([prefix, 'score']), 0, keep_index, dim=-1, initialize=True)
                        optimizer_params.update(ori_proj_weight, self.proj.weight, '.'.join([prefix, 'proj.weight']), 1, keep_index, dim=0)
                        optimizer_params.update(ori_proj_bias, self.proj.bias, '.'.join([prefix, 'proj.bias']), 0, keep_index, dim=-1)
                    if self.norm_layer:
                        ori_norm_weight = self.norm.weight
                        ori_norm_bias = self.norm.bias
                        self.norm.normalized_shape[0] = self.proj.out_channels
                        torch.cuda.synchronize()
                        self.norm.weight = torch.nn.Parameter(self.norm.weight.data.clone()[keep_index])
                        self.norm.bias = torch.nn.Parameter(self.norm.bias.data.clone()[keep_index])
                        if optimizer_params is not None:
                            optimizer_params.update(ori_norm_weight, self.norm.weight, '.'.join([prefix, 'norm.weight']), 0, keep_index, dim=-1)
                            optimizer_params.update(ori_norm_bias, self.norm.bias, '.'.join([prefix, 'norm.bias']), 0, keep_index, dim=-1)

                    return keep_index, optimizer_params, optimizer_decoder, optimizer_archs
                elif self.switch_cell[:, -1] == 0:
                    index = torch.nonzero(self.switch_cell)
                    ori_alpha = self.alpha
                    torch.cuda.synchronize()
                    self.alpha = nn.Parameter(self.alpha.data.clone()[:, :index[-1, 1] + 1])
                    if optimizer_archs is not None:
                        optimizer_archs.update(ori_alpha, self.alpha, '.'.join([prefix, 'alpha']), 0, 
                                               torch.arange(int(index[-1, 1]) + 1).to(self.alpha.device), dim=-1)

                    self.mask = self.mask[:index[-1, 1] + 1, :int(self.embed_ratio_list[index[-1, 1]] * self.embed_dim)]
                    self.switch_cell = self.switch_cell[:, :index[-1, 1] + 1]
                    self.weighted_mask = self.weighted_mask[:, :int(self.embed_ratio_list[index[-1, 1]] * self.embed_dim)]
                    self.proj.out_channels = int(self.embed_ratio_list[index[-1, 1]] * self.embed_dim)
                    channel_index = torch.argsort(self.score, dim=1, descending=True)[:, :self.proj.out_channels]
                    keep_index = channel_index.reshape(-1).detach()
                    ori_score = self.score
                    ori_proj_weight = self.proj.weight
                    ori_proj_bias = self.proj.bias
                    torch.cuda.synchronize()
                    self.score = nn.Parameter(self.score.data.clone()[:, keep_index])

                    self.proj.weight = torch.nn.Parameter(self.proj.weight.data.clone()[keep_index, ...])
                    self.proj.bias = torch.nn.Parameter(self.proj.bias.data.clone()[keep_index])
                    if optimizer_params is not None:
                        optimizer_params.update(ori_score, self.score, '.'.join([prefix, 'score']), 0, keep_index, dim=-1)
                        optimizer_params.update(ori_proj_weight, self.proj.weight, '.'.join([prefix, 'proj.weight']), 1, keep_index, dim=0)
                        optimizer_params.update(ori_proj_bias, self.proj.bias, '.'.join([prefix, 'proj.bias']), 0, keep_index, dim=-1)

                    if self.norm_layer:
                        ori_norm_weight = self.norm.weight
                        ori_norm_bias = self.norm.bias
                        self.norm.normalized_shape[0] = self.proj.out_channels
                        torch.cuda.synchronize()
                        self.norm.weight = torch.nn.Parameter(self.norm.weight.data.clone()[keep_index])
                        self.norm.bias = torch.nn.Parameter(self.norm.bias.data.clone()[keep_index])
                        if optimizer_params is not None:
                            optimizer_params.update(ori_norm_weight, self.norm.weight, '.'.join([prefix, 'norm.weight']), 0, keep_index, dim=-1)
                            optimizer_params.update(ori_norm_bias, self.norm.bias, '.'.join([prefix, 'norm.bias']), 0, keep_index, dim=-1)

                    return keep_index, optimizer_params, optimizer_decoder, optimizer_archs
            else:
                self.execute_prune = False
            torch.cuda.synchronize()
        return None, optimizer_params, optimizer_decoder, optimizer_archs

    def decompress(self):
        self.execute_prune = False
        self.alpha.requires_grad = True
        self.finish_search = False

    def get_params_count(self):
        dim1 = self.proj.in_channels
        dim2 = self.embed_dim
        kernel_size = self.proj.kernel_size[0] * self.proj.kernel_size[1]
        active_dim2 = self.weighted_mask.sum()
        total_params = dim1 * dim2 * kernel_size + dim2 + dim2 * 2
        active_params = dim1 * active_dim2 * kernel_size + active_dim2 + active_dim2 * 2
        return total_params, active_params, active_dim2

    def get_flops(self, num_patches):
        total_params, active_params, active_dim = self.get_params_count()
        conv_params = total_params - self.embed_dim * 2
        total_flops = conv_params * num_patches + (4 * self.embed_dim + 1) * num_patches
        active_conv_params = active_params - active_dim * 2
        active_flops = active_conv_params * num_patches + (4 * active_dim + 1) * num_patches
        return total_flops, active_flops

    @staticmethod
    def from_patchembed(patchmodule, embed_search=True):
        patchmodule = MAEPatchEmbed(patchmodule, embed_search)
        return patchmodule


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., num_patches=197):
        super().__init__()
        self.num_heads = num_heads
        self.num_patches = num_patches
        self.head_dim = dim // num_heads
        self.qk_scale = qk_scale
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_params_count(self):
        dim_in = self.qkv.in_features
        dim_out = self.qkv.out_features
        dim_embed = self.proj.out_features
        total_params = dim_in * dim_out + dim_out
        total_params += self.proj.in_features * dim_embed + dim_embed
        return total_params

    def get_flops(self, num_patches):
        H = self.num_heads
        N = num_patches
        d = self.qkv.out_features // H // 3
        active_embed = self.proj.out_features
        total_flops = N * (active_embed * (3 * H * d)) + 3 * N * H * d  # linear: qkv
        total_flops += H * N * d * N + H * N * N  # q@k
        total_flops += 5 * H * N * N  # softmax
        total_flops += H * N * N * d  # attn@v
        total_flops += N * (H * d * active_embed) + N * active_embed  # linear: proj
        return total_flops

class MAESparseAttention(Attention):
    def __init__(self, attn_module, head_search=False, channel_search=False, attn_search=True):
        super().__init__(attn_module.qkv.in_features, attn_module.num_heads, True, attn_module.scale,
                         attn_module.attn_drop.p, attn_module.proj_drop.p)
        self.finish_search = False
        self.execute_prune = False
        self.fused = False
        if attn_search:
            if head_search:
                self.head_num_list = list(range(2, self.num_heads + 1, 2))
                alpha_head = nn.Parameter(torch.rand(len(self.head_num_list), 1)) # -1, 1
                switch_cell_head = alpha_head > 0
                head_mask = torch.zeros(len(self.head_num_list), self.num_heads, 1, self.head_dim) # -1, H, 1, d(1)
                for i, r in enumerate(self.head_num_list):
                    head_mask[i, :r, :, :] = 1
                self.alpha = alpha_head
                self.switch_cell = switch_cell_head
                self.mask = head_mask
                self.score = nn.Parameter(torch.rand(self.num_heads, 1))
            elif channel_search:
                self.qkv_channel_ratio_list = [i / self.head_dim
                                               for i in range(self.head_dim // 4,
                                                              self.head_dim + 1,
                                                              max(self.head_dim // 8, 1))]
                alpha_channel = nn.Parameter(torch.rand(1, len(self.qkv_channel_ratio_list))) # 1, -1
                switch_cell_channel = alpha_channel > 0
                channel_mask = torch.zeros(1, self.num_heads, len(self.qkv_channel_ratio_list), self.head_dim) # 1, H, -1, d
                for i, r in enumerate(self.qkv_channel_ratio_list):
                    channel_mask[:, :, i, :int(self.head_dim * r)] = 1
                self.alpha = alpha_channel
                self.switch_cell = switch_cell_channel
                self.mask = channel_mask
                self.score = nn.Parameter(torch.rand(1, self.head_dim))
            else:
                self.head_num_list = list(range(2, self.num_heads + 1, 2))
                self.qkv_channel_ratio_list = [i / self.head_dim
                                               for i in range(self.head_dim // 4,
                                                              self.head_dim + 1,
                                                              max(self.head_dim // 8, 1))]
                alpha_joint = nn.Parameter(torch.rand(len(self.head_num_list), len(self.qkv_channel_ratio_list)))

                switch_cell_joint = alpha_joint > 0
                joint_mask = torch.zeros(len(self.head_num_list), self.num_heads,
                                         len(self.qkv_channel_ratio_list), self.head_dim)
                for i, n in enumerate(self.head_num_list):
                    for j, r in enumerate(self.qkv_channel_ratio_list):
                        joint_mask[i, :n, j, :int(self.head_dim * r)] = 1
                self.alpha = alpha_joint
                self.switch_cell = switch_cell_joint
                self.mask = joint_mask
                self.score = nn.Parameter(torch.rand(self.num_heads, self.head_dim))
            trunc_normal_(self.score, std=.2)
        else:
            self.head_num_list = [self.num_heads]
            self.qkv_channel_ratio_list = [1.0]
            self.alpha = nn.Parameter(torch.ones(len(self.head_num_list), len(self.qkv_channel_ratio_list)))
            self.switch_cell = self.alpha > 0
            joint_mask = torch.zeros(len(self.head_num_list), self.num_heads,
                                     len(self.qkv_channel_ratio_list), self.head_dim)
            for i, n in enumerate(self.head_num_list):
                for j, r in enumerate(self.qkv_channel_ratio_list):
                    joint_mask[i, :n, j, :int(self.head_dim * r)] = 1
            self.weighted_mask = self.mask = joint_mask
            self.finish_search = True
            self.score = torch.ones(self.num_heads, self.head_dim)
        self.in_features = self.qkv.in_features
        self.w_p = 0.99

    def update_w(self, cur_epoch, warmup_epochs, max=0.99, min=0.1):
        if cur_epoch <= warmup_epochs:
            self.w_p = (min - max) / warmup_epochs * cur_epoch + max

    def forward(self, x, mask_embed=None, weighted_embed=None):
        self.weighted_mask_embed = mask_embed # 1, embed_dim
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.head_num if hasattr(self, 'head_num') else self.num_heads, -1).permute(2, 0, 3, 1, 4).contiguous()  # 3, B, H, N, d(C/H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) # B, H, N, d
        if not self.finish_search:
            alpha = self.alpha - torch.where(self.switch_cell.to(self.alpha.device), torch.zeros_like(self.alpha), torch.ones_like(self.alpha) * float('inf'))
            alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(self.alpha)
            self.weighted_mask = sum(alpha[i][j] * self.mask[i, :, j, :].to(alpha.device) for i, j in product(range(alpha.size(0)), range(alpha.size(1)))
                                          if self.switch_cell[i][j]).unsqueeze(-2) # H, 1, d

            ids_shuffle_channel = torch.argsort(self.score.unsqueeze(-2).expand_as(self.weighted_mask), dim=-1, descending=True)  # descend: large is keep, small is remove
            ids_restore_channel = torch.argsort(ids_shuffle_channel, dim=-1)
            prob_score = self.score.sigmoid().unsqueeze(-2)
            head_score = prob_score.sum(-1, keepdim=True).expand_as(self.weighted_mask)
            ids_shuffle_head = torch.argsort(head_score, dim=0, descending=True)
            ids_restore_head = torch.argsort(ids_shuffle_head, dim=0)
            weight_restore = torch.gather(self.weighted_mask, dim=0, index=ids_restore_head)
            weight_restore = torch.gather(weight_restore, dim=-1, index=ids_restore_channel)
            q *= (1 - self.w_p) * weight_restore + self.w_p * prob_score
            k *= (1 - self.w_p) * weight_restore + self.w_p * prob_score
            v *= (1 - self.w_p) * weight_restore + self.w_p * prob_score
            attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, -1)
            x = self.proj(x)
            
            x = self.proj_drop(x)
        elif not self.fused:
            q *= self.score.unsqueeze(-2)
            k *= self.score.unsqueeze(-2)
            v *= self.score.unsqueeze(-2)
            attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

    def fuse(self):
        self.fused = True
        self.score.requires_grad = False
        self.qkv.weight = torch.nn.Parameter(self.qkv.weight.data.clone() * self.score.data.clone().reshape(-1).repeat(3).unsqueeze(-1))
        self.qkv.bias = torch.nn.Parameter(self.qkv.bias.data.clone() * self.score.data.clone().reshape(-1).repeat(3)) if self.qkv.bias is not None else None
        
    def get_alpha(self):
        return self.alpha, self.switch_cell.to(self.alpha.device)

    def get_weight(self):
        ids_shuffle_channel = torch.argsort(self.score.unsqueeze(-2).expand_as(self.weighted_mask), dim=-1, descending=True)  # descend: large is keep, small is remove
        ids_restore_channel = torch.argsort(ids_shuffle_channel, dim=-1)
        prob_score = self.score.sigmoid()
        head_score = prob_score.sum(-1, keepdim=True).unsqueeze(-2).expand_as(self.weighted_mask)
        ids_shuffle_head = torch.argsort(head_score, dim=0, descending=True)
        ids_restore_head = torch.argsort(ids_shuffle_head, dim=0)
        weight_restore = torch.gather(self.weighted_mask, dim=0, index=ids_restore_head)
        weight_restore = torch.gather(weight_restore, dim=-1, index=ids_restore_channel)
        return weight_restore.squeeze(-2), prob_score

    def compress(self, thresh, optimizer_params, optimizer_decoder, optimizer_archs, prefix=''):
        if self.switch_cell.sum() == 1:
            self.finish_search = True
            self.execute_prune = False
            self.alpha.requires_grad = False
        else:
            torch.cuda.synchronize()
            try:
                alpha_reduced = reduce_tensor(self.alpha.data)
            except: alpha_reduced = self.alpha.data
            # print(f'--Reduced Head Alpha: {alpha_reduced}--')
            torch.cuda.synchronize()
            alpha_norm = torch.softmax(alpha_reduced[self.switch_cell].view(-1), dim=0).detach()
            threshold_attn = thresh / self.switch_cell.sum()
            min_alpha = torch.min(alpha_norm)
            if min_alpha <= threshold_attn:
                print(f'--Head Alpha: {alpha_reduced}--')
                self.execute_prune = True
                alpha = alpha_reduced.detach() - torch.where(self.switch_cell.to(alpha_reduced.device), 
                                                             torch.zeros_like(alpha_reduced), 
                                                             torch.ones_like(alpha_reduced) * float('inf')).to(alpha_reduced.device)
                alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(alpha)
                self.switch_cell = (alpha > threshold_attn).detach()
                ori_alpha = self.alpha
                torch.cuda.synchronize()
                self.alpha = nn.Parameter(torch.where(self.switch_cell.to(self.alpha.device), alpha_reduced, torch.zeros_like(alpha).to(self.alpha.device)))
                if optimizer_archs is not None:
                    optimizer_archs.update(ori_alpha, self.alpha, '.'.join([prefix, 'alpha']), 0, 
                                           torch.arange(self.alpha.shape[-1]).to(self.alpha.device), dim=-1, initialize=True)

                alpha = self.alpha - torch.where(self.switch_cell, torch.zeros_like(self.alpha), 
                                                 torch.ones_like(self.alpha) * float('inf')).to(self.alpha.device)
                alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(alpha)
                self.weighted_mask = sum(alpha[i][j] * self.mask[i, :, j, :].to(alpha.device) 
                                         for i, j in product(range(alpha.size(0)), range(alpha.size(1))) 
                                         if self.switch_cell[i][j]).unsqueeze(-2) # H, 1, d
                print(f'---Normalized Alpha: {alpha_norm}---')
                print(f'------Prune {self}: {(alpha_norm <= threshold_attn).sum()} cells------')
                print(f'---Updated Weighted Mask of Head Dimension: {self.weighted_mask}---')
                if self.switch_cell.sum() == 1:
                    self.finish_search = True
                    self.alpha.requires_grad = False
                    if optimizer_archs is not None:
                        optimizer_archs.update(self.alpha, self.alpha, '.'.join([prefix, 'alpha']), 0, None, dim=-1)

                    self.weighted_mask = self.weighted_mask.detach()
                    feature_index = torch.arange(self.qkv.out_features).reshape(3, self.head_num if hasattr(self, 'head_num') else self.num_heads, -1)
                    proj_index = torch.arange(self.proj.in_features).reshape(self.head_num if hasattr(self, 'head_num') else self.num_heads, -1)
                    index = torch.nonzero(self.switch_cell)
                    assert index.shape[0] == 1
                    self.head_num = self.head_num_list[index[0, 0]] if hasattr(self, 'head_num_list') else self.num_heads
                    dim_ratio = self.qkv_channel_ratio_list[index[0, 1]] if hasattr(self, 'qkv_channel_ratio_list') else 1
                    self.scale = self.qk_scale or int(dim_ratio * self.head_dim) ** -0.5
                    self.qkv.out_features = self.head_num * int(dim_ratio * self.head_dim) * 3

                    head_index = torch.argsort(self.score.sigmoid().sum(-1), dim=0, 
                                               descending=True)[:self.head_num] if self.score.shape[0] != 1 else torch.arange(self.head_num)
                    channel_index = torch.argsort(self.score, dim=1, descending=True)[:, :int(dim_ratio * self.head_dim)]
                    channel_index = torch.gather(channel_index, dim=0, index=head_index.unsqueeze(-1).repeat(1, channel_index.shape[-1]))
                    keep_index = torch.gather(feature_index.to(head_index.device), dim=1, 
                                              index=head_index.unsqueeze(0).unsqueeze(-1).repeat(3, 1, feature_index.shape[-1]))
                    keep_index = torch.gather(keep_index, dim=-1, index=channel_index.unsqueeze(0).repeat(3, 1, 1)).reshape(-1).detach()

                    ori_score = self.score
                    ori_qkv_weight = self.qkv.weight
                    ori_qkv_bias = self.qkv.bias
                    torch.cuda.synchronize()
                    self.score = nn.Parameter(torch.gather(self.score.data.clone(), dim=0, index=head_index.unsqueeze(-1).repeat(1, self.score.shape[-1])))
                    self.score = nn.Parameter(torch.gather(self.score.data.clone(), dim=-1, index=channel_index))
                    self.weighted_mask = self.weighted_mask[:len(head_index), :, :channel_index.shape[-1]]
                    self.score = nn.Parameter(self.w_p * self.score.sigmoid().data.clone() + (1 - self.w_p) * self.weighted_mask.squeeze().data.clone())
                    self.qkv.weight = torch.nn.Parameter(self.qkv.weight.data.clone()[keep_index, :])
                    self.qkv.bias = torch.nn.Parameter(self.qkv.bias.data.clone()[keep_index])
                    if optimizer_params is not None:
                        optimizer_params.update(ori_score, self.score, '.'.join([prefix, 'score']), 0, [head_index, channel_index], dim=[0, -1], initialize=True)
                        optimizer_params.update(ori_qkv_weight, self.qkv.weight, '.'.join([prefix, 'qkv.weight']), 1, keep_index, dim=0)
                        if self.qkv.bias is not None: optimizer_params.update(ori_qkv_bias, self.qkv.bias, '.'.join([prefix, 'qkv.bias']), 0, keep_index, dim=-1)

                    keep_index = torch.gather(proj_index.to(head_index.device), dim=0, index=head_index.unsqueeze(-1).repeat(1, proj_index.shape[-1]))
                    keep_index = torch.gather(keep_index, dim=-1, index=channel_index).reshape(-1).detach()
                    self.proj.in_features = len(keep_index)
                    ori_proj_weight = self.proj.weight
                    torch.cuda.synchronize()
                    self.proj.weight = nn.Parameter(self.proj.weight.data.clone()[:, keep_index])
                    if optimizer_params is not None:
                        optimizer_params.update(ori_proj_weight, self.proj.weight, '.'.join([prefix, 'proj.weight']), 1, keep_index, dim=-1)

                elif self.switch_cell[:, -1].sum() == 0 or self.switch_cell[-1, :].sum() == 0:
                    index = torch.nonzero(self.switch_cell)
                    index = [index[:, 0].max(), index[:, 1].max()]
                    feature_index = torch.arange(self.qkv.out_features).reshape(3, self.head_num if hasattr(self, 'head_num') else self.num_heads, -1)
                    proj_index = torch.arange(self.proj.in_features).reshape(self.head_num if hasattr(self, 'head_num') else self.num_heads, -1)

                    self.head_num = self.head_num_list[index[0]] if hasattr(self, 'head_num_list') else self.num_heads
                    dim_ratio = self.qkv_channel_ratio_list[index[1]] if hasattr(self, 'qkv_channel_ratio_list') else 1
                    ori_alpha = self.alpha
                    torch.cuda.synchronize()
                    self.alpha = nn.Parameter(self.alpha.data.clone()[:index[0] + 1, :index[1] + 1])
                    if optimizer_archs is not None:
                        optimizer_archs.update(ori_alpha, self.alpha, '.'.join([prefix, 'alpha']), 0, 
                                               [torch.arange(int(index[0] + 1)).to(self.alpha.device), torch.arange(int(index[1] + 1)).to(self.alpha.device)], dim=[0, -1])
                    self.mask = self.mask[:index[0] + 1, :self.head_num, :index[1] + 1, :int(dim_ratio * self.head_dim)]
                    self.switch_cell = self.switch_cell[:index[0] + 1, :index[1] + 1]
                    self.weighted_mask = self.weighted_mask[:self.head_num, :, :int(dim_ratio * self.head_dim)]
                    self.scale = self.qk_scale or int(dim_ratio * self.head_dim) ** -0.5
                    self.qkv.out_features = self.head_num * int(dim_ratio * self.head_dim) * 3

                    head_index = torch.argsort(self.score.sigmoid().sum(-1), dim=0, descending=True)[:self.head_num] if self.score.shape[0] != 1 else torch.arange(self.head_num)
                    channel_index = torch.argsort(self.score, dim=1, descending=True)[:, :int(dim_ratio * self.head_dim)]
                    channel_index = torch.gather(channel_index, dim=0, index=head_index.unsqueeze(-1).repeat(1, channel_index.shape[-1]))
                    keep_index = torch.gather(feature_index.to(head_index.device), dim=1, index=head_index.unsqueeze(0).unsqueeze(-1).repeat(3, 1, feature_index.shape[-1]))
                    keep_index = torch.gather(keep_index, dim=-1, index=channel_index.unsqueeze(0).repeat(3, 1, 1)).reshape(-1).detach()

                    ori_score = self.score
                    ori_qkv_weight = self.qkv.weight
                    ori_qkv_bias = self.qkv.bias
                    torch.cuda.synchronize()
                    self.score = nn.Parameter(torch.gather(self.score.data.clone(), dim=0, index=head_index.unsqueeze(-1).repeat(1, self.score.shape[-1])))
                    self.score = nn.Parameter(torch.gather(self.score.data.clone(), dim=-1, index=channel_index))
                    self.qkv.weight = torch.nn.Parameter(self.qkv.weight.data.clone()[keep_index, :])
                    self.qkv.bias = torch.nn.Parameter(self.qkv.bias.data.clone()[keep_index]) if self.qkv.bias is not None else None
                    if optimizer_params is not None:
                        optimizer_params.update(ori_score, self.score, '.'.join([prefix, 'score']), 0, [head_index, channel_index], dim=[0, -1])
                        optimizer_params.update(ori_qkv_weight, self.qkv.weight, '.'.join([prefix, 'qkv.weight']), 1, keep_index, dim=0)
                        if self.qkv.bias is not None: optimizer_params.update(ori_qkv_bias, self.qkv.bias, '.'.join([prefix, 'qkv.bias']), 0, keep_index, dim=-1)

                    keep_index = torch.gather(proj_index.to(head_index.device), dim=0, index=head_index.unsqueeze(-1).repeat(1, proj_index.shape[-1]))
                    keep_index = torch.gather(keep_index, dim=-1, index=channel_index).reshape(-1).detach()
                    self.proj.in_features = len(keep_index)
                    ori_proj_weight = self.proj.weight
                    torch.cuda.synchronize()
                    self.proj.weight = nn.Parameter(self.proj.weight.data.clone()[:, keep_index])
                    if optimizer_params is not None:
                        optimizer_params.update(ori_proj_weight, self.proj.weight, '.'.join([prefix, 'proj.weight']), 1, keep_index, dim=-1)

            else: self.execute_prune = False
            torch.cuda.synchronize()
        return optimizer_params, optimizer_decoder, optimizer_archs

    def compress_patchembed(self, info, optimizer_params, optimizer_decoder, optimizer_archs, prefix=''):
        if isinstance(info, torch.Tensor):
            keep_index = info
            ori_qkv_weight = self.qkv.weight
            ori_proj_weight = self.proj.weight
            ori_proj_bias = self.proj.bias
            self.qkv.in_features = len(keep_index)
            self.qkv.weight = nn.Parameter(self.qkv.weight.data.clone()[:, keep_index])
            self.proj.out_features = self.qkv.in_features
            self.proj.weight = torch.nn.Parameter(self.proj.weight.data.clone()[keep_index, ...])
            self.proj.bias = torch.nn.Parameter(self.proj.bias.data.clone()[keep_index]) if self.proj.bias is not None else None
            if optimizer_params is not None:
                optimizer_params.update(ori_qkv_weight, self.qkv.weight, '.'.join([prefix, 'qkv.weight']), 1, keep_index, dim=-1)
                optimizer_params.update(ori_proj_weight, self.proj.weight, '.'.join([prefix, 'proj.weight']), 1, keep_index, dim=0)
                if self.proj.bias is not None: optimizer_params.update(ori_proj_bias, self.proj.bias, '.'.join([prefix, 'proj.bias']), 0, keep_index, dim=-1)
        else:
            keep_ratio = info
            ori_qkv_weight = self.qkv.weight
            ori_proj_weight = self.proj.weight
            ori_proj_bias = self.proj.bias
            self.qkv.in_features = int(self.in_features * keep_ratio) if isinstance(keep_ratio, float) else keep_ratio
            self.qkv.weight = nn.Parameter(self.qkv.weight.data.clone()[:, :self.qkv.in_features])
            self.proj.out_features = self.qkv.in_features
            self.proj.weight = torch.nn.Parameter(self.proj.weight.data.clone()[:self.proj.out_features, ...])
            self.proj.bias = torch.nn.Parameter(self.proj.bias.data.clone()[:self.proj.out_features]) if self.proj.bias is not None else None
            keep_index = torch.arange(self.qkv.in_features)
            if optimizer_params is not None:
                optimizer_params.update(ori_qkv_weight, self.qkv.weight, '.'.join([prefix, 'qkv.weight']), 1, keep_index, dim=-1)
                optimizer_params.update(ori_proj_weight, self.proj.weight, '.'.join([prefix, 'proj.weight']), 1, keep_index, dim=0)
                if self.proj.bias is not None: optimizer_params.update(ori_proj_bias, self.proj.bias, '.'.join([prefix, 'proj.bias']), 0, keep_index, dim=-1)
        return optimizer_params, optimizer_decoder, optimizer_archs

    def decompress(self):
        self.execute_prune = False
        self.alpha.requires_grad = True
        self.finish_search = False

    def get_params_count(self):
        dim = self.in_features
        active_dim = self.qkv.in_features
        active_embedding_dim = self.weighted_mask_embed.sum() if self.weighted_mask_embed is not None and torch.sum(
                torch.multiply(self.weighted_mask_embed < 1, self.weighted_mask_embed > 0)) != 0 else active_dim
        active_qkv_dim = self.weighted_mask.sum()
        total_params = dim * dim * 3 + dim * 3
        total_params += dim * dim + dim
        active_params = active_embedding_dim * active_qkv_dim * 3 + active_qkv_dim * 3
        active_params += active_qkv_dim * active_embedding_dim + active_embedding_dim
        return total_params, active_params

    def get_flops(self, num_patches, active_patches):
        H = self.num_heads
        active_H = self.head_num if hasattr(self, 'head_num') else H
        N = num_patches
        n = active_patches
        d = self.head_dim
        sd = self.weighted_mask.sum()
        active_embed = self.weighted_mask_embed.sum()
        total_flops = N * (H * d * (3 * H * d)) + 3 * N * H * d  # linear: qkv
        total_flops += H * N * d * N + H * N * N  # q@k
        total_flops += 5 * H * N * N  # softmax
        total_flops += H * N * N * d  # attn@v
        total_flops += N * (H * d * (H * d)) + N * H * d  # linear: proj

        active_flops = n * (active_embed * (3 * sd)) + 3 * n * sd  # linear: qkv
        active_flops += n * n * sd + active_H * n * n  # q@k
        active_flops += 5 * active_H * n * n  # softmax
        active_flops += n * n * sd  # attn@v
        active_flops += n * (sd * active_embed) + n * active_embed  # linear: proj
        return total_flops, active_flops

    @staticmethod
    def from_attn(attn_module, head_search=False, channel_search=False, attn_search=True):
        attn_module = MAESparseAttention(attn_module, head_search, channel_search, attn_search)
        return attn_module


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def get_params_count(self):
        dim1 = self.fc1.in_features
        dim2 = self.fc1.out_features
        dim3 = self.fc2.out_features
        total_params = dim1 * dim2 + dim2 * dim3 + dim2 + dim3
        return total_params

    def get_flops(self, num_patches):
        total_params = self.get_params_count()
        return total_params * num_patches


class MAESparseMlp(Mlp):
    def __init__(self, mlp_module, mlp_search=True):
        super().__init__(mlp_module.fc1.in_features, mlp_module.fc1.out_features, mlp_module.fc2.out_features,
                         act_layer=nn.GELU, drop=mlp_module.drop.p)
        self.finish_search = False
        self.execute_prune = False
        self.fused = False
        hidden_features = self.fc1.out_features
        if mlp_search:
            self.hidden_ratio_list = [i / hidden_features
                                      for i in range(hidden_features // 4,
                                                     hidden_features + 1,
                                                     hidden_features // 8)]
            self.alpha = nn.Parameter(torch.rand(1, len(self.hidden_ratio_list)))
            self.switch_cell = self.alpha > 0
            hidden_mask = torch.zeros(len(self.hidden_ratio_list), hidden_features)  # -1, H, 1, d(1)
            for i, r in enumerate(self.hidden_ratio_list):
                hidden_mask[i, :int(r * hidden_features)] = 1
            self.mask = hidden_mask
            self.score = nn.Parameter(torch.rand(1, hidden_features))
            trunc_normal_(self.score, std=.2)
        else:
            self.hidden_ratio_list = [1.0]
            self.alpha = nn.Parameter(torch.ones(1, len(self.hidden_ratio_list)))
            self.switch_cell = self.alpha > 0
            hidden_mask = torch.zeros(len(self.hidden_ratio_list), hidden_features)  # -1, H, 1, d(1)
            for i, r in enumerate(self.hidden_ratio_list):
                hidden_mask[i, :int(r * hidden_features)] = 1
            self.weighted_mask = self.mask = hidden_mask
            self.finish_search = True
            self.score = torch.ones(1, hidden_features)
        self.in_features = self.fc1.in_features
        self.hidden_features = hidden_features
        self.w_p = 0.99

    def update_w(self, cur_epoch, warmup_epochs, max=0.99, min=0.1):
        if cur_epoch <= warmup_epochs:
            self.w_p = (min - max) / warmup_epochs * cur_epoch + max

    def forward(self, x, mask_embed=None, weighted_embed=None):
        self.weighted_mask_embed = mask_embed
        x = self.fc1(x)
        if not self.finish_search:
            alpha = self.alpha - torch.where(self.switch_cell.to(self.alpha.device), torch.zeros_like(self.alpha),
                                             torch.ones_like(self.alpha) * float('inf'))
            alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(self.alpha)
            self.weighted_mask = sum(
                alpha[i][j] * self.mask[j, :].to(alpha.device) for i, j in product(range(alpha.size(0)), range(alpha.size(1)))
                if self.switch_cell[i][j]).unsqueeze(-2)  # 1, d

            ids_shuffle_channel = torch.argsort(self.score, dim=-1, descending=True)  # descend: large is keep, small is remove
            ids_restore_channel = torch.argsort(ids_shuffle_channel, dim=-1)
            prob_score = self.score.sigmoid()
            weight_restore = torch.gather(self.weighted_mask, dim=-1, index=ids_restore_channel)
            x *= self.w_p * prob_score + (1 - self.w_p) * weight_restore
        elif not self.fused:
            x *= self.score
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def fuse(self):
        self.fused = True
        self.score.requires_grad = False
        self.fc1.weight = torch.nn.Parameter(self.fc1.weight.data.clone() * self.score.data.clone().squeeze().unsqueeze(-1))
        self.fc1.bias = torch.nn.Parameter(self.fc1.bias.data.clone() * self.score.data.clone().squeeze())
        
    def get_alpha(self):
        return self.alpha, self.switch_cell.to(self.alpha.device)

    def get_weight(self):
        ids_shuffle_channel = torch.argsort(self.score, dim=-1, descending=True)  # descend: large is keep, small is remove
        ids_restore_channel = torch.argsort(ids_shuffle_channel, dim=-1)
        prob_score = self.score.sigmoid()
        weight_restore = torch.gather(self.weighted_mask, dim=-1, index=ids_restore_channel)
        return weight_restore, prob_score

    def compress(self, thresh, optimizer_params, optimizer_decoder, optimizer_archs, prefix=''):
        if self.switch_cell.sum() == 1:
            self.finish_search = True
            self.execute_prune = False
            self.alpha.requires_grad = False
        else:
            torch.cuda.synchronize()
            try:
                alpha_reduced = reduce_tensor(self.alpha.data)
            except: alpha_reduced = self.alpha.data
            torch.cuda.synchronize()
            # print(f'--Reduced MLP Alpha: {alpha_reduced}--')
            alpha_norm = torch.softmax(alpha_reduced[self.switch_cell].view(-1), dim=0).detach()
            threshold = thresh / self.switch_cell.sum()
            min_alpha = torch.min(alpha_norm)
            if min_alpha <= threshold:
                print(f'--MLP Alpha: {alpha_reduced}--')
                self.execute_prune = True
                alpha = alpha_reduced.detach() - torch.where(self.switch_cell.to(alpha_reduced.device), torch.zeros_like(alpha_reduced), 
                                                             torch.ones_like(alpha_reduced) * float('inf')).to(alpha_reduced.device)
                alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(alpha)
                self.switch_cell = (alpha > threshold).detach()
                ori_alpha = self.alpha
                torch.cuda.synchronize()
                self.alpha = nn.Parameter(torch.where(self.switch_cell, alpha_reduced, torch.zeros_like(alpha).to(self.alpha.device)))
                if optimizer_archs is not None:
                    optimizer_archs.update(ori_alpha, self.alpha, '.'.join([prefix, 'alpha']), 0, torch.arange(self.alpha.shape[-1]).to(self.alpha.device), 
                                           dim=-1, initialize=True)

                alpha = self.alpha - torch.where(self.switch_cell, torch.zeros_like(self.alpha),
                                                 torch.ones_like(self.alpha) * float('inf')).to(self.alpha.device)
                alpha = torch.softmax(alpha.view(-1), dim=0).reshape_as(alpha)
                self.weighted_mask = sum(alpha[i][j] * self.mask[j, :].to(alpha.device) for i, j in
                                         product(range(alpha.size(0)), range(alpha.size(1)))
                                         if self.switch_cell[i][j]).unsqueeze(-2)  # 1, d
                print(f'---Normalized Alpha: {alpha_norm}---')
                print(f'------Prune {self}: {(alpha_norm <= threshold).sum()} cells------')
                print(f'---Updated Weighted Mask of MLP Dimension: {self.weighted_mask}---')
                if self.switch_cell.sum() == 1:
                    self.finish_search = True
                    self.alpha.requires_grad = False
                    if optimizer_archs is not None:
                        optimizer_archs.update(self.alpha, self.alpha, '.'.join([prefix, 'alpha']), 0, None, dim=-1)

                    self.weighted_mask = self.weighted_mask.detach()
                    index = torch.nonzero(self.switch_cell)
                    assert index.shape[0] == 1
                    self.fc1.out_features = int(self.hidden_ratio_list[index[0, 1]] * self.hidden_features)

                    channel_index = torch.argsort(self.score, dim=1, descending=True)[:, :self.fc1.out_features]
                    keep_index = channel_index.reshape(-1).detach()
                    ori_score = self.score
                    ori_fc1_weight = self.fc1.weight
                    ori_fc1_bias = self.fc1.bias
                    self.weighted_mask = self.weighted_mask[:, :len(keep_index)]
                    torch.cuda.synchronize()
                    self.score = nn.Parameter(self.w_p * self.score.sigmoid()[:, keep_index].data.clone() + (1 - self.w_p) * self.weighted_mask.data.clone())
                    self.fc1.weight = torch.nn.Parameter(self.fc1.weight.data.clone()[keep_index, ...])
                    self.fc1.bias = torch.nn.Parameter(self.fc1.bias.data.clone()[keep_index])
                    if optimizer_params is not None:
                        optimizer_params.update(ori_score, self.score, '.'.join([prefix, 'score']), 0, keep_index, dim=-1, initialize=True)
                        optimizer_params.update(ori_fc1_weight, self.fc1.weight, '.'.join([prefix, 'fc1.weight']), 1, keep_index, dim=0)
                        optimizer_params.update(ori_fc1_bias, self.fc1.bias, '.'.join([prefix, 'fc1.bias']), 0, keep_index, dim=-1)

                    self.fc2.in_features = int(self.hidden_ratio_list[index[0, 1]] * self.hidden_features)

                    ori_fc2_weight = self.fc2.weight
                    torch.cuda.synchronize()
                    self.fc2.weight = torch.nn.Parameter(self.fc2.weight.data.clone()[:, keep_index])
                    if optimizer_params is not None:
                        optimizer_params.update(ori_fc2_weight, self.fc2.weight, '.'.join([prefix, 'fc2.weight']), 1, keep_index, dim=-1)

                elif self.switch_cell[:, -1] == 0:
                    index = torch.nonzero(self.switch_cell)
                    ori_alpha = self.alpha
                    torch.cuda.synchronize()
                    self.alpha = nn.Parameter(self.alpha.data.clone()[:, :index[-1, 1] + 1])
                    if optimizer_archs is not None:
                        optimizer_archs.update(ori_alpha, self.alpha, '.'.join([prefix, 'alpha']), 0, torch.arange(int(index[-1, 1]) + 1).to(self.alpha.device), dim=-1)
                    self.mask = self.mask[:index[-1, 1] + 1, :int(self.hidden_ratio_list[index[-1, 1]] * self.hidden_features)]
                    self.switch_cell = self.switch_cell[:, :index[-1, 1] + 1]
                    self.weighted_mask = self.weighted_mask[:, :int(self.hidden_ratio_list[index[-1, 1]] * self.hidden_features)]
                    self.fc1.out_features = int(self.hidden_ratio_list[index[-1, 1]] * self.hidden_features)

                    channel_index = torch.argsort(self.score, dim=1, descending=True)[:, :self.fc1.out_features]
                    keep_index = channel_index.reshape(-1).detach()
                    ori_score = self.score
                    ori_fc1_weight = self.fc1.weight
                    ori_fc1_bias = self.fc1.bias
                    torch.cuda.synchronize()
                    self.score = nn.Parameter(self.score.data.clone()[:, keep_index])

                    self.fc1.weight = torch.nn.Parameter(self.fc1.weight.data.clone()[keep_index, ...])
                    self.fc1.bias = torch.nn.Parameter(self.fc1.bias.data.clone()[keep_index])
                    if optimizer_params is not None:
                        optimizer_params.update(ori_score, self.score, '.'.join([prefix, 'score']), 0, keep_index, dim=-1)
                        optimizer_params.update(ori_fc1_weight, self.fc1.weight, '.'.join([prefix, 'fc1.weight']), 1, keep_index, dim=0)
                        optimizer_params.update(ori_fc1_bias, self.fc1.bias, '.'.join([prefix, 'fc1.bias']), 0, keep_index, dim=-1)

                    self.fc2.in_features = int(self.hidden_ratio_list[index[-1, 1]] * self.hidden_features)

                    ori_fc2_weight = self.fc2.weight
                    torch.cuda.synchronize()
                    self.fc2.weight = torch.nn.Parameter(self.fc2.weight.data.clone()[:, keep_index])
                    if optimizer_params is not None:
                        optimizer_params.update(ori_fc2_weight, self.fc2.weight, '.'.join([prefix, 'fc2.weight']), 1, keep_index, dim=-1)

            else: self.execute_prune = False
            torch.cuda.synchronize()
        return optimizer_params, optimizer_decoder, optimizer_archs

    def compress_patchembed(self, info, optimizer_params, optimizer_decoder, optimizer_archs, prefix=''):
        if isinstance(info, torch.Tensor):
            keep_index = info
            ori_fc1_weight = self.fc1.weight
            ori_fc2_weight = self.fc2.weight
            ori_fc2_bias = self.fc2.bias
            self.fc1.in_features = len(keep_index)
            self.fc1.weight = torch.nn.Parameter(self.fc1.weight.data.clone()[:, keep_index])
            self.fc2.out_features = self.fc1.in_features
            self.fc2.weight = torch.nn.Parameter(self.fc2.weight.data.clone()[keep_index, ...])
            self.fc2.bias = torch.nn.Parameter(self.fc2.bias.data.clone()[keep_index]) if self.fc2.bias is not None else None
            if optimizer_params is not None:
                optimizer_params.update(ori_fc1_weight, self.fc1.weight, '.'.join([prefix, 'fc1.weight']), 1, keep_index, dim=-1)
                optimizer_params.update(ori_fc2_weight, self.fc2.weight, '.'.join([prefix, 'fc2.weight']), 1, keep_index, dim=0)
                if self.fc2.bias is not None: optimizer_params.update(ori_fc2_bias, self.fc2.bias, '.'.join([prefix, 'fc2.bias']), 0, keep_index, dim=-1)
        else:
            keep_ratio = info
            ori_fc1_weight = self.fc1.weight
            ori_fc2_weight = self.fc2.weight
            ori_fc2_bias = self.fc2.bias
            self.fc1.in_features = int(self.in_features * keep_ratio) if isinstance(keep_ratio, float) else keep_ratio
            self.fc1.weight = torch.nn.Parameter(self.fc1.weight.data.clone()[:, :self.fc1.in_features])
            self.fc2.out_features = self.fc1.in_features
            self.fc2.weight = torch.nn.Parameter(self.fc2.weight.data.clone()[:self.fc2.out_features, ...])
            self.fc2.bias = torch.nn.Parameter(self.fc2.bias.data.clone()[:self.fc2.out_features]) if self.fc2.bias is not None else None
            keep_index = torch.arange(self.fc2.out_features).to(self.fc2.weight.device)
            if optimizer_params is not None:
                optimizer_params.update(ori_fc1_weight, self.fc1.weight, '.'.join([prefix, 'fc1.weight']), 1, keep_index, dim=-1)
                optimizer_params.update(ori_fc2_weight, self.fc2.weight, '.'.join([prefix, 'fc2.weight']), 1, keep_index, dim=0)
                if self.fc2.bias is not None: optimizer_params.update(ori_fc2_bias, self.fc2.bias, '.'.join([prefix, 'fc2.bias']), 0, keep_index, dim=-1)

        return optimizer_params, optimizer_decoder, optimizer_archs

    def decompress(self):
        self.execute_prune = False
        self.alpha.requires_grad = True
        self.finish_search = False

    def get_params_count(self):
        dim1 = self.in_features
        dim2 = self.hidden_features
        active_dim1 = self.fc1.in_features
        active_dim2 = self.weighted_mask.sum()
        active_embedding_dim = self.weighted_mask_embed.sum() if self.weighted_mask_embed is not None else active_dim1
        total_params = 2 * (dim1 * dim2) + dim1 + dim2
        active_params = active_embedding_dim * active_dim2 + active_dim2 * active_embedding_dim + active_embedding_dim + active_dim2
        return total_params, active_params

    def get_flops(self, num_patches, active_patches):
        total_params, active_params = self.get_params_count()
        return total_params * num_patches, active_params * active_patches

    @staticmethod
    def from_mlp(mlp_module, mlp_search=True):
        mlp_module = MAESparseMlp(mlp_module, mlp_search)
        return mlp_module


class ModuleInjection:
    method = 'full'
    searchable_modules = []

    @staticmethod
    def make_searchable_patchembed(patchmodule, embed_search=True):
        if ModuleInjection.method == 'full':
            return patchmodule
        patchmodule = MAEPatchEmbed.from_patchembed(patchmodule, embed_search)
        if embed_search:
            ModuleInjection.searchable_modules.append(patchmodule)
        return patchmodule

    @staticmethod
    def make_searchable_maeattn(attn_module, head_search=False, channel_search=False, attn_search=True):
        if ModuleInjection.method == 'full':
            return attn_module
        attn_module = MAESparseAttention.from_attn(attn_module, head_search, channel_search, attn_search)
        if attn_search:
            ModuleInjection.searchable_modules.append(attn_module)
        return attn_module

    @staticmethod
    def make_searchable_maemlp(mlp_module, mlp_search=True):
        if ModuleInjection.method == 'full':
            return mlp_module
        mlp_module = MAESparseMlp.from_mlp(mlp_module, mlp_search)
        if mlp_search:
            ModuleInjection.searchable_modules.append(mlp_module)
        return mlp_module
