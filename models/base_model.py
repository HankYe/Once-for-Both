import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from math import pi

class MAEBaseModel(nn.Module):
    def __init__(self):
        super(MAEBaseModel, self).__init__()
        self.searchable_modules = []

    def give_alphas(self):
        alphas_attn = []
        alphas_mlp = []
        alphas_embed = []
        alphas_patch = self.alpha_patch.cpu().detach().reshape(-1).numpy().tolist()
        for l_block in self.searchable_modules:
            alpha, _ = l_block.get_alpha()
            if hasattr(l_block, 'num_heads'):
                alphas_attn.append(alpha.cpu().detach().reshape(-1).numpy().tolist())
            elif hasattr(l_block, 'embed_ratio_list'):
                alphas_embed.append(alpha.cpu().detach().reshape(-1).numpy().tolist())
            else:
                alphas_mlp.append(alpha.cpu().detach().reshape(-1).numpy().tolist())
        return alphas_attn, alphas_mlp, alphas_patch, alphas_embed

    def get_flops(self):
        pass

    def get_flops_loss(self, target_flops):
        ori_flops, searched_flops = self.get_flops()
        print(f'Original FLOPs: {ori_flops:.1f} GFLOPs, Searched FLOPs: {searched_flops:.1f} GFLOPs, Target FLOPs: {target_flops:.1f}')
        flops_loss = torch.mean(((searched_flops - target_flops) / ori_flops) ** 2)
        return flops_loss

    def get_sparsity_loss(self, device, entropy=True, var=True, norm=True):
        alpha_patch, switch_cell_patch = self.alpha_patch, self.switch_cell_patch.to(device)
        if switch_cell_patch.sum() != 1:
            prob_patch_act = torch.softmax(alpha_patch[switch_cell_patch], dim=-1)
            loss_patch = - (prob_patch_act * prob_patch_act.float().log()).sum()

            if loss_patch.isnan():
                print(loss_patch)
            mean_prob_act = torch.mean(prob_patch_act)
            target_sigma_patch = 1. - 1. / switch_cell_patch.sum()
            sigma_patch = ((prob_patch_act - mean_prob_act) ** 2).sum()
            sigma_prob = sigma_patch / target_sigma_patch
            assert sigma_prob <= 1.
            loss_patch += torch.tan(pi / 2 - pi * sigma_prob)
        else: loss_patch = torch.tensor(0.).to(device)
        
        loss_attn, loss_mlp, loss_embedding = torch.tensor(0.).to(device), torch.tensor(0.).to(device), torch.tensor(0.).to(device)
        for l_block in self.searchable_modules:

            alpha, switch_cell = l_block.get_alpha()
            if switch_cell.sum() == 1:
                continue
            
            prob_act = torch.softmax(alpha[switch_cell], dim=-1)
            if entropy:
                loss = - (prob_act * prob_act.float().log()).sum()
            else: loss = torch.tensor(0.).to(device)
            if var:
                mean_prob_act = torch.mean(prob_act)
                target_sigma = 1. - 1. / switch_cell.sum()
                sigma = ((prob_act - mean_prob_act) ** 2).sum()
                sigma_prob = sigma / target_sigma
                assert sigma_prob <= 1.
                loss += torch.tan(pi / 2 - pi * sigma_prob) / switch_cell.sum()

            if norm:
                mask_restore, prob_score = l_block.get_weight()
                if hasattr(l_block, 'num_heads'):
                    score_loss = torch.sum(prob_score) * 4e-4
                else:
                    score_loss = torch.sum(prob_score) * 1e-4
                loss += score_loss

            if hasattr(l_block, 'num_heads'):
                loss_attn += loss
            elif hasattr(l_block, 'embed_ratio_list'):
                loss_embedding += loss
            else:
                loss_mlp += loss
        return loss_attn.to(device), loss_mlp.to(device), loss_patch.to(device), loss_embedding.to(device)

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

    def get_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        searched_params = total_params
        for l_block in self.searchable_modules:
            searched_params -= l_block.get_params_count()[0]
            searched_params += l_block.get_params_count()[1]
        return total_params, searched_params.item()