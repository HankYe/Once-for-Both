# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import gc
import torch
from apex import amp

from timm.data import Mixup
from timm.utils import accuracy
from utils import ModelEma
import utils

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_schedule,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode = True, use_amp=False, args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    accum_iter = args.accum_iter
    print_freq = 10
    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if max_norm is not None: torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm)
        else:
            loss.backward()
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_schedule.step_update(epoch * len(data_loader) + data_iter_step)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def search_one_epoch(model: torch.nn.Module, criterion, target_flops,
                    data_loader: Iterable, optimizer_param: torch.optim.Optimizer,
                    optimizer_decoder: torch.optim.Optimizer, optimizer_arch: torch.optim.Optimizer,
                    lr_scheduler_param, lr_scheduler_arch, lr_scheduler_decoder,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode = True, use_amp=False, finish_search=False, args=None, progressive=True, max_ratio=0.95, min_ratio=0.75):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr_param', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    accum_iter = args.accum_iter
    if optimizer_decoder is not None:
        metric_logger.add_meter('lr_decoder', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        optimizer_decoder.zero_grad()
    optimizer_param.zero_grad()
    if not finish_search:
        optimizer_arch.zero_grad()
    execute_pruned = False
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            if progressive:
                if hasattr(model, 'module'):
                    model.module.adjust_masking_ratio(data_iter_step / len(data_loader) + epoch, args.warmup_epochs, 
                                                      args.epochs, max_ratio=max_ratio, min_ratio=min_ratio)
                else:
                    model.adjust_masking_ratio(data_iter_step / len(data_loader) + epoch, args.warmup_epochs, 
                                               args.epochs, max_ratio=max_ratio, min_ratio=min_ratio)
            if hasattr(model, 'module'):
                for m in model.module.searchable_modules:
                    if not m.finish_search:
                        m.update_w(data_iter_step / len(data_loader) + epoch, args.warmup_epochs)
            else:
                for m in model.searchable_modules:
                    if not m.finish_search:
                        m.update_w(data_iter_step / len(data_loader) + epoch, args.warmup_epochs)
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs, aux_loss = model(samples)
                decoder_loss, score_loss = aux_loss
                loss = criterion(samples, outputs, targets, model, 'arch', target_flops, finish_search)
                if decoder_loss != 0.:
                    w_decoder = (loss / decoder_loss).data.clone()
                    loss_total = loss + w_decoder * decoder_loss
                else:
                    loss_total = loss
                if score_loss is not None:
                    loss_total += score_loss
        else:
            outputs, aux_loss = model(samples)
            decoder_loss, score_loss = aux_loss
            loss = criterion(samples, outputs, targets, model, 'arch', target_flops, finish_search)
            if isinstance(loss, tuple):
                base_loss, arch_loss = loss
                loss_total = base_loss + arch_loss
            else:
                base_loss = loss.item()
                loss_total = loss
            if decoder_loss != 0.:
                w_decoder = (base_loss / decoder_loss).data.clone()
                loss_total += w_decoder * decoder_loss
            if score_loss is not None:
                loss_total += score_loss

        loss_value = loss_total.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_total /= accum_iter
        if use_amp:
            if optimizer_decoder is not None and optimizer_arch is not None:
                optimizer_group = [optimizer_param, optimizer_arch, optimizer_decoder]
            elif optimizer_arch is not None:
                optimizer_group = [optimizer_param, optimizer_arch]
            elif optimizer_decoder is not None:
                optimizer_group = [optimizer_param, optimizer_decoder]
            with amp.scale_loss(loss_total, optimizer_group, loss_id=0) as scaled_loss:
                scaled_loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_param), max_norm)
                if optimizer_arch is not None:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_arch), max_norm)
                if optimizer_decoder is not None:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_decoder), max_norm)
        else:
            loss_total.backward()
        if (data_iter_step + 1) % accum_iter == 0:
            torch.cuda.synchronize()
            optimizer_param.step()
            if optimizer_arch is not None:
                optimizer_arch.step()
            if optimizer_decoder is not None:
                optimizer_decoder.step()
            optimizer_param.zero_grad()
            lr_scheduler_param.step_update(epoch * len(data_loader) + data_iter_step)
            if optimizer_arch is not None:
                optimizer_arch.zero_grad()
                lr_scheduler_arch.step_update(epoch * len(data_loader) + data_iter_step)
            if optimizer_decoder is not None:
                optimizer_decoder.zero_grad()
                lr_scheduler_decoder.step_update(epoch * len(data_loader) + data_iter_step)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss_param=base_loss)
        metric_logger.update(loss_total=loss_value)
        metric_logger.update(lr_param=optimizer_param.param_groups[0]["lr"])
        if optimizer_arch is not None:
            metric_logger.update(loss_arch=arch_loss.item())
            metric_logger.update(lr_arch=optimizer_arch.param_groups[0]["lr"])
        if optimizer_decoder is not None and not isinstance(decoder_loss, float):
            metric_logger.update(loss_decoder=decoder_loss.item())
            metric_logger.update(lr_decoder=optimizer_decoder.param_groups[0]["lr"])
        
        # UPDATING ARCHs
        if not finish_search and (data_iter_step + 1) % accum_iter == 0 and ((data_iter_step + 1) // accum_iter) % (len(data_loader) // 3 // accum_iter) == 0:
            print('Start Compression')
            torch.cuda.synchronize()
            finish_search, execute_prune, optimizer_param, optimizer_decoder, optimizer_arch = model.module.compress(
                0.2, optimizer_param, optimizer_decoder, optimizer_arch)
            execute_pruned |= execute_prune
            if finish_search:
                optimizer_arch = None
                lr_scheduler_arch = None
            
            torch.cuda.synchronize()
            if model_ema is not None:
                model_ema.update(model)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats, finish_search, execute_pruned, optimizer_param, optimizer_decoder, optimizer_arch


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output, _ = model(images)
                loss = criterion(output, target)
        else:
            output, _ = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_finetune(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}