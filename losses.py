# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs.float(), labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd.float() / T, dim=1),
                F.log_softmax(teacher_outputs.float() / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd.float(), teacher_outputs.float().argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

class OFBSearchLOSS(torch.nn.Module):
    def __init__(self, base_criterion, device, attn_w=0.0001, mlp_w=0.0001, patch_w=0.0001, embedding_w=0.0001, flops_w=0.0001, entropy=True, var=True, norm=True):
        super().__init__()
        self.base_criterion = base_criterion
        self.w1 = attn_w
        self.w2 = mlp_w
        self.w3 = patch_w
        self.w4 = embedding_w
        self.w5 = flops_w
        self.entropy = entropy
        self.var = var
        self.norm = norm
        self.device = device

    def forward(self, inputs, outputs, labels, model, phase: str, target_flops=1.0, finish_search=False):
        if isinstance(outputs, tuple):
            preds, decoder_pred = outputs
            base_loss = self.base_criterion(inputs, preds, labels)
            kl_loss = F.kl_div(F.log_softmax(decoder_pred, dim=-1), F.softmax(preds, dim=-1), reduction='batchmean')
            decoder_pred_loss = self.base_criterion(inputs, decoder_pred, labels) + kl_loss
            base_loss += decoder_pred_loss
        else:
            preds = outputs
            base_loss = self.base_criterion(inputs, preds, labels)

        if not finish_search:
            if 'arch' in phase:
                loss_flops = model.module.get_flops_loss(target_flops).to(self.device)
                loss_attn, loss_mlp, loss_patch, loss_embedding = model.module.get_sparsity_loss(self.device, self.entropy, self.var, self.norm)
                if loss_attn.isnan() or loss_mlp.isnan() or loss_patch.isnan() or loss_embedding.isnan():
                    print(loss_attn)
                return (base_loss,
                       self.w1 * loss_attn \
                       + self.w2 * loss_mlp \
                       + self.w3 * loss_patch \
                       + self.w4 * loss_embedding \
                       + self.w5 * loss_flops)
            else:
                return base_loss
        else:
            return base_loss