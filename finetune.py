import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import json
import gc
from apex import amp
from models.layers import LayerNorm
from pathlib import Path
from os.path import exists
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from lr_sched import create_scheduler
from datasets import build_dataset
from engine import train_one_epoch, evaluate_finetune
from losses import DistillationLoss
from samplers import RASampler
import models
import lr_decay as lrd
import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import ModelEma
import matplotlib.pyplot as plt


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT finetune script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--accum-iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained_path', default='', type=str, metavar='PRETRAIN',
                        help='Name of model to train')
    parser.add_argument('--finetune', default='', type=str, metavar='FINETUNE',
                        help='Name of model to finetune')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters (if sched is none, warmup and min dont matter)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.95,
                        help='layer-wise lr decay from ELECTRA/BEiT')
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
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
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
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0,
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
    parser.add_argument('--data-path', default='/root/data/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR10', 'CIFAR100', 'IMNET', 'INAT', 'INAT19', 'IMNET100'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7',
                        help='devices to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def intersect(model, pretrained_model, exclude=None):
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
        try:
            layer_pretrained = eval(f'pretrained_model.{model_id}')
            if hasattr(layer_pretrained, 'finish_search') and not layer_pretrained.finish_search:
                pretrained_model.compress(1.0)
                state = pretrained_model.state_dict()
        except: pass
        if exclude and any([ee in k for ee in exclude]):
            if 'head' in k:
                layer = torch.nn.Linear(state[f'{k}.weight'].shape[1], v.weight.shape[0])
                model.head = layer
                counted.append(model_id)
                print(f'Update model.{model_id}: {eval(f"model.{model_id}")}')
            continue
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
            layer.num_heads = eval(f'pretrained_model.{model_id}.head_num') if hasattr(eval(f'pretrained_model.{model_id}'), 'head_num') \
                else eval(f'pretrained_model.{model_id}.num_heads')
            layer.qk_scale = eval(f'pretrained_model.{model_id}.qk_scale')
            exec('m = layer', {'m': f'model.{model_id}', 'layer': layer})
            counted.append(model_id)
            print(f'Update model.{model_id}: {eval(f"model.{model_id}")}')
    model.cls_token = torch.nn.Parameter(state['cls_token'].data.clone())
    model.pos_embed = torch.nn.Parameter(state['pos_embed'].data.clone())
    print(f'Update total {len(counted) + 2} parameters.')
    return model

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

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
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
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

    print(f"Loading model: {args.model}")

    model = create_model(
        args.model,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None
    )
    if args.pretrained_path:
        state_dict = torch.load(args.pretrained_path, map_location='cpu')['model']
        model = intersect(model, state_dict)

    if args.finetune:
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
        model = intersect(model, state_dict, exclude=['head', 'head_dist'])
        # interpolate position embedding
        pos_embed_checkpoint = state_dict.pos_embed
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - state_dict.patch_embed.num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        model.pos_embed = torch.nn.Parameter(new_pos_embed.data.clone())
    del state_dict
    gc.collect()

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
    eff_batch_size = args.batch_size * args.accum_iter * utils.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    kwargs_optim = dict(lr=args.lr)
    if getattr(args, 'opt_eps', None) is not None: kwargs_optim['eps'] = args.opt_eps
    if getattr(args, 'opt_betas', None) is not None: kwargs_optim['betas'] = args.opt_betas
    if getattr(args, 'opt_args', None) is not None: kwargs_optim.update(args.opt_args)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
                                        no_weight_decay_list=model.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )

    optimizer_param = torch.optim.AdamW(param_groups, **kwargs_optim)

    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args.epochs, args.warmup_epochs, args.warmup_lr, 
                                       args.min_lr, args, optimizer_param, len(data_loader_train))

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

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
        model, optimizer_param = amp.initialize(model, optimizer_param)
    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and all(key not in name for key in ['decoder', 'alpha', 'score']))
    n_flops = model_without_ddp.get_flops()
    print('number of params:', n_parameters)
    print('GFLOPs: ', n_flops / 1e9)

    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer_param, lr_scheduler, device, epoch, loss_scaler,
                args.clip_grad, model_ema, mixup_fn, use_amp=args.use_amp, args=args, set_training_mode=args.finetune==''
        )
        if args.output_dir:
            checkpoint_paths = [output_dir / 'running_ckpt.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp,
                    'optimizer': optimizer_param.state_dict(),
                    # 'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'model_ema': model_ema.ema.state_dict() if args.model_ema else model_ema,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

        # if global_rank in [-1, 0]:
        test_stats = evaluate_finetune(data_loader_val, model, device, use_amp=False)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        torch.cuda.synchronize()

        if args.output_dir and test_stats["acc1"] >= max_accuracy:
            checkpoint_paths = [output_dir / 'best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp,
                    'epoch': epoch,
                    'model_ema': model_ema.ema.state_dict() if args.model_ema else model_ema,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        torch.cuda.synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT fintuning script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
