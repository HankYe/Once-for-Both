import argparse
import datetime
import numpy as np
import time
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import json
from apex import amp

from pathlib import Path
from os.path import exists
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from datasets import build_dataset
from engine import evaluate, search_one_epoch
from losses import DistillationLoss, OFBSearchLOSS
from samplers import RASampler
import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import ModelEma
from models.layers import LayerNorm
from optim import AdamW
from lr_sched import create_scheduler

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT Searching script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum-iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='deit_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--mae', action='store_true')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--mask-ratio', default=1.0, type=float, help='mask ratio')
    parser.add_argument('--fuse_point', default=50, type=int)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', default='', type=str,
                        help='path of resuming from checkpoint model.')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-eps-arch', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-eps-decoder', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--opt-betas-arch', default=(0.5, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--opt-betas-decoder', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--momentum-decoder', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--weight-decay-arch', type=float, default=1e-3,
                        help='weight decay (default: 1e-3)')
    parser.add_argument('--weight-decay-decoder', type=float, default=1e-3,
                        help='weight decay (default: 1e-3)')
    # Learning rate schedule parameters (if sched is none, warmup and min dont matter)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "none"')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_decoder', type=float, default=None, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr_arch', type=float, default=None, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--blr', type=float, default=2.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--blr_decoder', type=float, default=2.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--blr_arch', type=float, default=2.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")

    # Dataset parameters
    parser.add_argument('--data-path', default='/root/data/ILSVRC2015/Data/CLS-LOC/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19', 'IMNET100'],
                        type=str, help='Image Net dataset path')
    # parser.add_argument('--proxy-ratio', type=float, default=1.0,
    #                     help='Probability of sampling proxy dataset')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='runs/test',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu', default='0',
                        help='devices to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # searching parameters
    parser.add_argument('--w_head', default=0.5, type=float, help='weightage to attn head dimension')
    parser.add_argument('--w_embedding', default=0.5, type=float, help='weightage to patch embedding dimension')
    parser.add_argument('--w_mlp', default=0.5, type=float, help='weightage to mlp channel dimension')
    parser.add_argument('--w_patch', default=0, type=float, help='weightage to patch number dimension')
    parser.add_argument('--w_flops', default=5, type=float, help='weightage to the flops loss')
    parser.add_argument('--w_decoder', default=1, type=float, help='weightage to the decoder loss')
    parser.add_argument('--target_flops', default=1.0, type=float)
    parser.add_argument('--max_ratio', default=0.95, type=float)
    parser.add_argument('--min_ratio', default=0.75, type=float)
    parser.add_argument('--pretrained_path', default='', type=str)
    parser.add_argument('--head_search', action='store_true', help='whether to search the head number')
    parser.add_argument('--channel_search', action='store_true', help='whether to search the qkv channel number')
    parser.add_argument('--attn_search', action='store_true', help='whether to search the attn number')
    parser.add_argument('--mlp_search', action='store_true', help='whether to search the mlp number')
    parser.add_argument('--embed_search', action='store_true', help='whether to search the embed number')
    parser.add_argument('--patch_search', action='store_true', help='whether to search the patch number')
    parser.add_argument('--freeze_weights', action='store_true')
    parser.add_argument('--no-progressive', action='store_true')
    parser.add_argument('--no-entropy', action='store_true')
    parser.add_argument('--no-var', action='store_true')
    parser.add_argument('--no-norm', action='store_true')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)
    parser.add_argument('--vis-score', action='store_true')
    return parser

def intersect(model, pretrained_model):
    state = pretrained_model.state_dict()
    counted = []
    for k, v in list(model.named_modules()):
        have_layers = [i.isdigit() for i in k.split('.')]
        if any(have_layers):
            model_id = []
            for i, ele in enumerate(k.split('.')):
                if have_layers[i]:
                    model_id[-1] = model_id[-1] + f'[{ele}]'
                else:
                    model_id.append(ele)
            model_id = '.'.join(model_id)
        else:
            model_id = k
        if hasattr(v, 'weight') and f'{k}.weight' in state.keys():
            layer = eval(f'model.{model_id}')
            layer.weight = torch.nn.Parameter(state[f'{k}.weight'].data.clone())
            if hasattr(layer, 'out_channels'):
                layer.out_channels = layer.weight.shape[0]
                layer.in_channels = layer.weight.shape[1]
            if hasattr(layer, 'out_features'):
                layer.out_features = layer.weight.shape[0]
                layer.in_features = layer.weight.shape[1]
            if layer.bias is not None:
                layer.bias = torch.nn.Parameter(state[f'{k}.bias'].data.clone())
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.num_features = layer.weight.shape[0]
                layer.running_mean = state[f'{k}.running_mean'].data.clone()
                layer.running_var = state[f'{k}.running_var'].data.clone()
                layer.num_batches_tracked = state[f'{k}.num_batches_tracked'].data.clone()
            if isinstance(layer, LayerNorm):
                layer.normalized_shape[0] = layer.weight.shape[-1]
            exec('m = layer', {'m': f'model.{model_id}', 'layer': layer})
            counted.append(model_id)
            print(f'Update model.{model_id}: {eval(f"model.{model_id}")}')
        elif isinstance(v, torch.Tensor):
            layer = eval(f'model.{model_id}')
            assert isinstance(layer, torch.nn.Parameter)
            layer = torch.nn.Parameter(state[f'{k}'].data.clone())
            exec('m = layer', {'m': f'model.{model_id}', 'layer': layer})
            counted.append(model_id)
            print(f'Update model.{model_id}: {eval(f"model.{model_id}")}')
        elif hasattr(v, 'num_heads'):
            layer = eval(f'model.{model_id}')
            try:
                layer.num_heads = eval(f'pretrained_model.{model_id}.head_num')
            except:
                layer.num_heads = eval(f'pretrained_model.{model_id}.num_heads')
            layer.qk_scale = eval(f'pretrained_model.{model_id}.qk_scale')
            exec('m = layer', {'m': f'model.{model_id}', 'layer': layer})
            counted.append(model_id)
            print(f'Update model.{model_id}: {eval(f"model.{model_id}")}')
        if hasattr(v, 'alpha'):
            layer = eval(f'model.{model_id}')
            layer.finish_search = eval(f'pretrained_model.{model_id}.finish_search')
            layer.weighted_mask = eval(f'pretrained_model.{model_id}.weighted_mask')
            layer.switch_cell = eval(f'pretrained_model.{model_id}.switch_cell')
            layer.alpha = eval(f'pretrained_model.{model_id}.alpha')
            layer.mask = eval(f'pretrained_model.{model_id}.mask')
            layer.score = eval(f'pretrained_model.{model_id}.score')
            exec('m = layer', {'m': f'model.{model_id}', 'layer': layer})
            print(f'Update the search results of model.{model_id}: {eval(f"model.{model_id}")}')
    model.cls_token = torch.nn.Parameter(state['cls_token'].data.clone())
    model.pos_embed = torch.nn.Parameter(state['pos_embed'].data.clone())
    model.mask_token = torch.nn.Parameter(state['mask_token'].data.clone())
    model.finish_search = pretrained_model.finish_search
    model.weighted_mask = pretrained_model.weighted_mask
    model.switch_cell_patch = pretrained_model.switch_cell_patch
    model.alpha_patch = pretrained_model.alpha_patch
    model.patch_search_mask = pretrained_model.patch_search_mask
    print(f'Update total {len(counted) + 3} parameters.') # cls_token, pos_embed, mask_token
    return model


def resume(args, checkpoint_path, model_ema, device):
    print(f'Loading from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = checkpoint['model'].to(device)
    finish_search = model.finish_search
    decay_params, no_decay_params, decay_decoder, no_decay_decoder, archs = [], [], [], [], []
    decay_params_name, no_decay_params_name, decay_decoder_name, no_decay_decoder_name, archs_name = [], [], [], [], []
    skip = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if len(p.shape) == 1 or name.endswith(".bias") or any([ele in name for ele in skip]):
            if 'decoder' not in name:
                no_decay_params.append(p)
                no_decay_params_name.append(name)
            else:
                no_decay_decoder.append(p)
                no_decay_decoder_name.append(name)
        elif "alpha" in name:
            archs.append(p)
            archs_name.append(name)
        else:
            if 'decoder' not in name:
                decay_params.append(p)
                decay_params_name.append(name)
            else:
                decay_decoder.append(p)
                decay_decoder_name.append(name)
    kwargs_optim = dict(lr=args.lr)
    if getattr(args, 'opt_eps', None) is not None: kwargs_optim['eps'] = args.opt_eps
    if getattr(args, 'opt_betas', None) is not None: kwargs_optim['betas'] = args.opt_betas
    if getattr(args, 'opt_args', None) is not None: kwargs_optim.update(args.opt_args)

    kwargs_optim_arch = dict(lr=args.lr_arch)
    if getattr(args, 'opt_eps_arch', None) is not None: kwargs_optim_arch['eps'] = args.opt_eps_arch
    if getattr(args, 'opt_betas_arch', None) is not None: kwargs_optim_arch['betas'] = args.opt_betas_arch
    if getattr(args, 'opt_args_arch', None) is not None: kwargs_optim_arch.update(args.opt_args_arch)

    kwargs_optim_decoder = dict(lr=args.lr_decoder)
    if getattr(args, 'opt_eps_decoder', None) is not None: kwargs_optim_decoder['eps'] = args.opt_eps_decoder
    if getattr(args, 'opt_betas_decoder', None) is not None: kwargs_optim_decoder['betas'] = args.opt_betas_decoder
    if getattr(args, 'opt_args_decoder', None) is not None: kwargs_optim_decoder.update(args.opt_args_decoder)

    param_names = {0: no_decay_params_name, 1: decay_params_name}
    optimizer_param = AdamW([{'params': no_decay_params, 'weight_decay': 0.},
                             {'params': decay_params, 'weight_decay': args.weight_decay}], param_names, **kwargs_optim)
    if len(decay_decoder):
        decoder_names = {0: no_decay_decoder_name, 1: decay_decoder_name}
        optimizer_decoder = AdamW([{'params': no_decay_decoder, 'weight_decay': 0.},
                                   {'params': decay_decoder, 'weight_decay': args.weight_decay_decoder}], decoder_names,
                                  **kwargs_optim_decoder)
    else: optimizer_decoder = None
    if len(archs):
        archs_names = {0: archs_name}
        optimizer_arch = AdamW(archs, archs_names, **kwargs_optim_arch, weight_decay=1e-3)
    else: optimizer_arch = None
    loss_scaler = NativeScaler()

    try:
        optimizer_param.load_state_dict(checkpoint['optimizer_param'])
        if 'optimizer_decoder' in checkpoint and optimizer_decoder is not None and checkpoint['optimizer_decoder'] is not None:
            optimizer_decoder.load_state_dict(checkpoint['optimizer_decoder'])
        if 'optimizer_arch' in checkpoint and optimizer_arch is not None and checkpoint['optimizer_arch'] is not None:
            optimizer_arch.load_state_dict(checkpoint['optimizer_arch'])
        if model_ema is not None and checkpoint['model_ema'] is not None:
            utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
    except: pass
    args.start_epoch = checkpoint['epoch'] + 1
    return model, model_ema, finish_search, optimizer_param, optimizer_arch, optimizer_decoder

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    print(f"Creating model: {args.model}")
    
    model = create_model(
        args.model,
        pretrained=True,
        mae=args.mae,
        pretrained_strict=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        method='search',
        head_search=args.head_search,
        channel_search=args.channel_search,
        norm_pix_loss=args.norm_pix_loss,
        attn_search=args.attn_search,
        mlp_search=args.mlp_search,
        embed_search=args.embed_search,
        patch_search=args.patch_search,
        mask_ratio=args.mask_ratio
        )
    if args.pretrained_path != '':
        print(f'Loading from {args.pretrained_path}')
        assert exists(args.pretrained_path)
        state_dict = torch.load(args.pretrained_path, map_location='cpu')['model']
        model = intersect(model, state_dict)

    model.to(device)
    model.correct_require_grad(args.w_head, args.w_mlp, args.w_patch, args.w_embedding)

    if args.freeze_weights:
        for name, p in model.named_parameters():
            if any([key in name for key in ['alpha', 'score', 'norm', 'token', 'decoder', 'mask', 'head']]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    finish_search = model.finish_search

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False, )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    decay_params, no_decay_params, decay_decoder, no_decay_decoder, archs = [], [], [], [], []
    decay_params_name, no_decay_params_name, decay_decoder_name, no_decay_decoder_name, archs_name = [], [], [], [], []
    skip = model.no_weight_decay() if hasattr(model, 'no_weight_decay') else {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if len(p.shape) == 1 or name.endswith(".bias") or any([ele in name for ele in skip]):
            if 'decoder' not in name:
                no_decay_params.append(p)
                no_decay_params_name.append(name)
            else:
                no_decay_decoder.append(p)
                no_decay_decoder_name.append(name)
        elif "alpha" in name:
            archs.append(p)
            archs_name.append(name)
        else:
            if 'decoder' not in name:
                decay_params.append(p)
                decay_params_name.append(name)
            else:
                decay_decoder.append(p)
                decay_decoder_name.append(name)
    eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.lr_arch is None:  # only base_lr is specified
        args.lr_arch = args.blr_arch * eff_batch_size / 256

    if args.lr_decoder is None:  # only base_lr is specified
        args.lr_decoder = args.blr_decoder * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("base arch lr: %.2e" % (args.lr_arch * 256 / eff_batch_size))
    print("actual arch lr: %.2e" % args.lr_arch)
    print("base decoder lr: %.2e" % (args.lr_decoder * 256 / eff_batch_size))
    print("actual decoder lr: %.2e" % args.lr_decoder)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    kwargs_optim = dict(
        lr=args.lr)
    if getattr(args, 'opt_eps', None) is not None:
        kwargs_optim['eps'] = args.opt_eps
    if getattr(args, 'opt_betas', None) is not None:
        kwargs_optim['betas'] = args.opt_betas
    if getattr(args, 'opt_args', None) is not None:
        kwargs_optim.update(args.opt_args)

    kwargs_optim_arch = dict(lr=args.lr_arch)
    if getattr(args, 'opt_eps_arch', None) is not None: kwargs_optim_arch['eps'] = args.opt_eps_arch
    if getattr(args, 'opt_betas_arch', None) is not None: kwargs_optim_arch['betas'] = args.opt_betas_arch
    if getattr(args, 'opt_args_arch', None) is not None: kwargs_optim_arch.update(args.opt_args_arch)

    kwargs_optim_decoder = dict(lr=args.lr_decoder)
    if getattr(args, 'opt_eps_decoder', None) is not None: kwargs_optim_decoder['eps'] = args.opt_eps_decoder
    if getattr(args, 'opt_betas_decoder', None) is not None: kwargs_optim_decoder['betas'] = args.opt_betas_decoder
    if getattr(args, 'opt_args_decoder', None) is not None: kwargs_optim_decoder.update(args.opt_args_decoder)

    param_names = {0: no_decay_params_name, 1: decay_params_name}
    optimizer_param = AdamW([{'params': no_decay_params, 'weight_decay': 0.},
                             {'params': decay_params, 'weight_decay': args.weight_decay}], param_names, **kwargs_optim)
    if len(decay_decoder):
        decoder_names = {0: no_decay_decoder_name, 1: decay_decoder_name}
        optimizer_decoder = AdamW([{'params': no_decay_decoder, 'weight_decay': 0.},
                                   {'params': decay_decoder, 'weight_decay': args.weight_decay_decoder}], decoder_names, **kwargs_optim_decoder)
    else: optimizer_decoder = None
    if len(archs):
        archs_names ={0: archs_name}
        optimizer_arch = AdamW(archs, archs_names, **kwargs_optim_arch, weight_decay=1e-3)
    else: optimizer_arch = None
    loss_scaler = NativeScaler()

    output_dir = Path(args.output_dir)
    sa_dict, sp_dict, ss_dict = {}, {}, {}
    if args.resume:
        if global_rank == 0 and (output_dir / 'saliency.npy').exists():
            sa_dict = np.load(output_dir / 'saliency.npy', allow_pickle=True).item()
            sp_dict = np.load(output_dir / 'sparsity.npy', allow_pickle=True).item()
            ss_dict = np.load(output_dir / 'joint.npy', allow_pickle=True).item()
        model, model_ema, finish_search, optimizer_param, optimizer_arch, optimizer_decoder = resume(args, args.checkpoint, model_ema, device)
        model.correct_require_grad(args.w_head, args.w_mlp, args.w_patch, args.w_embedding)

    lr_scheduler_params, _ = create_scheduler(args.epochs, args.warmup_epochs, args.warmup_lr, 
                                              args.min_lr, args, optimizer_param, len(data_loader_train))
    if optimizer_arch is not None: lr_scheduler_arch, _ = create_scheduler(args.epochs, args.warmup_epochs, args.warmup_lr, 
                                                                           args.min_lr, args, optimizer_arch, len(data_loader_train))
    else: lr_scheduler_arch = None
    if optimizer_decoder is not None: lr_scheduler_decoder, _ = create_scheduler(args.epochs, args.warmup_epochs, args.warmup_lr, 
                                                                                 args.min_lr, args, optimizer_decoder, len(data_loader_train))
    else: lr_scheduler_decoder = None

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing: criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else: criterion = torch.nn.CrossEntropyLoss()

    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    if args.use_amp:
        if optimizer_arch is not None and optimizer_decoder is not None:
            model, [optimizer_param, optimizer_decoder, optimizer_arch] = amp.initialize(model, 
                                                                                         [optimizer_param, optimizer_decoder, optimizer_arch], num_losses=2)
        elif optimizer_arch is not None:
            model, [optimizer_param, optimizer_arch] = amp.initialize(model, [optimizer_param, optimizer_arch], num_losses=2)
        elif optimizer_decoder is not None:
            model, [optimizer_param, optimizer_decoder] = amp.initialize(model, [optimizer_param, optimizer_decoder])
    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )
    criterion = OFBSearchLOSS(
        criterion, device, attn_w=args.w_head, mlp_w=args.w_mlp,
        patch_w=args.w_patch, embedding_w=args.w_embedding, flops_w=args.w_flops,
        entropy=not args.no_entropy, var=not args.no_var, norm=not args.no_norm
    )


    print(f"Start training for {args.epochs} epochs")
    target_flops = args.target_flops
    start_time = time.time()
    max_soft_accuracy = 0.0
    flag = True
    execute_prune = False
    for epoch in range(args.start_epoch, args.epochs):
        if finish_search and flag:
            flag = False
            if hasattr(model, 'module'):
                model.module.reset_mask_ratio(1.0)
                model.module.freeze_decoder()
            else:
                model.reset_mask_ratio(1.0)
                model.freeze_decoder()
            optimizer_decoder = None
            lr_scheduler_decoder = None
            mixup_fn = Mixup(
                mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.nb_classes)
            criterion.base_criterion.base_criterion = SoftTargetCrossEntropy()

            max_soft_accuracy = 0.0
        
        torch.cuda.synchronize()
        if args.distributed: data_loader_train.sampler.set_epoch(epoch)

        train_stats, finish_search, execute_prune, optimizer_param, optimizer_decoder, optimizer_arch = search_one_epoch(
            model, criterion, target_flops, data_loader_train,
            optimizer_param, optimizer_decoder, optimizer_arch,
            lr_scheduler_params, lr_scheduler_arch, lr_scheduler_decoder, device, epoch,
            args.clip_grad, model_ema, mixup_fn, use_amp=args.use_amp, finish_search=finish_search, args=args,
            progressive=not args.no_progressive, max_ratio=args.max_ratio, min_ratio=args.min_ratio
        )

        torch.cuda.synchronize()
        if args.output_dir:
            if finish_search and execute_prune:
                checkpoint_path = output_dir / 'model_pruned.pth'
                utils.save_on_master({
                    'model': model_without_ddp,
                    'optimizer_param': optimizer_param.state_dict(),
                    'optimizer_arch': optimizer_arch.state_dict() if optimizer_arch is not None else None,
                    'optimizer_decoder': optimizer_decoder.state_dict() if optimizer_decoder is not None else None,
                    'epoch': epoch,
                    'model_ema': model_ema.ema.state_dict() if args.model_ema else model_ema,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

            ### save the score map and sparsity map
            if (not finish_search or execute_prune) and global_rank == 0 and args.vis_score:
                for idx, m in enumerate(model_without_ddp.searchable_modules):
                    sparsity_score, saliency_score = m.weighted_mask.detach(), torch.sigmoid(m.score.detach())
                    sp, sa = sparsity_score.squeeze().data.cpu().numpy(), saliency_score.squeeze().data.cpu().numpy()
                    sa_sorted = np.sort(sa, axis=-1)
                    if hasattr(m, 'num_heads'):
                        index = np.argsort(sa_sorted.sum(axis=-1))[::-1]
                        sa_sorted = sa_sorted[index]
                        sa_sorted = sa_sorted[:, ::-1]
                    else:
                        index = np.argsort(sa_sorted)[::-1]
                        sa_sorted = sa_sorted[index]
                    ss = (1 - m.w_p) * sp + m.w_p * sa_sorted
                    if len(sa_dict) and idx in sa_dict:
                        if sa_dict[idx][-1].size == sa_sorted.size and (sa_dict[idx][-1] == sa_sorted).all(): continue
                        sa_dict[idx].append(sa_sorted)
                        sp_dict[idx].append(sp)
                        ss_dict[idx].append(ss)
                    else:
                        sa_dict[idx] = [sa_sorted]
                        sp_dict[idx] = [sp]
                        ss_dict[idx] = [ss]
                np.save(output_dir / 'saliency.npy', sa_dict)
                np.save(output_dir / 'sparsity.npy', sp_dict)
                np.save(output_dir / 'joint.npy', ss_dict)
            checkpoint_paths = [output_dir / 'running_ckpt.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp,
                    'optimizer_param': optimizer_param.state_dict(),
                    'optimizer_arch': optimizer_arch.state_dict() if optimizer_arch is not None else None,
                    'optimizer_decoder': optimizer_decoder.state_dict() if optimizer_decoder is not None else None,
                    'epoch': epoch,
                    'model_ema': model_ema.ema.state_dict() if args.model_ema else model_ema,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        torch.cuda.synchronize()
        if global_rank in [-1, 0]:
            test_stats = evaluate(data_loader_val, model, device, use_amp=False)
            print(f"Soft Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_soft_accuracy = max(max_soft_accuracy, test_stats["acc1"])
            print(f'Max soft accuracy: {max_soft_accuracy:.2f}%')

            if args.output_dir and test_stats["acc1"] >= max_soft_accuracy:
                checkpoint_paths = [output_dir / 'best.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp,
                        'epoch': epoch,
                        'model_ema': model_ema.ema.state_dict() if args.model_ema else model_ema,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
            n_parameters_updated = sum(p.numel() for name, p in model.named_parameters() 
                                       if p.requires_grad and 'decoder' not in name and 'alpha' not in name and 'score' not in name)
            flops = model.module.get_flops()[1].item() if hasattr(model, 'module') else model.get_flops()[1].item()
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'soft_test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters_updated,
                         'n_gflops': flops}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if not finish_search:
                    alphas_attn, alphas_mlp, alphas_patch, alphas_embed = model.module.give_alphas()
                    if args.mae:
                        log_alphas = {'epoch': epoch,
                                      'attn': alphas_attn,
                                      'mlp': alphas_mlp,
                                      'patch': alphas_patch,
                                      'embed': alphas_embed
                                      }
                    else:
                        log_alphas = {'epoch': epoch,
                                      'attn': alphas_attn,
                                      'mlp': alphas_mlp,
                                      'patch': alphas_patch,
                                      'embed': alphas_embed
                                      }
                    with open(output_dir / 'alpha.txt', "a") as f:
                        f.write(json.dumps(log_alphas) + "\n")

        torch.cuda.synchronize()
        if epoch == args.fuse_point and hasattr(model_without_ddp, 'fused') and not model_without_ddp.fused: break

    if utils.is_main_process() and finish_search and not execute_prune and hasattr(model_without_ddp, 'fused') and not model_without_ddp.fused:
        best_state = torch.load(output_dir / 'best.pth', map_location='cpu')
        best_model = best_state['model']
        best_model = best_model.cuda()
        best_model.fuse()
        test_stats = evaluate(data_loader_val, best_model, device, use_amp=False)
        print(f"Soft Accuracy of the fused network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        checkpoint_paths = [output_dir / 'model_fused.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': best_model,
                'epoch': best_state['epoch']
            }, checkpoint_path)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT searching script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
