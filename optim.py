import torch
# from torch.optim import adamw
import math
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, param_names, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)
        self.param_names = param_names

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def update(self, ori_w, cur_w, w_name, group_idx, keep_idx, dim, initialize=False):
        w_index = self.param_names[group_idx].index(w_name)
        if cur_w.requires_grad:
            self.param_groups[group_idx]['params'][w_index] = cur_w
            ori_state = self.state.pop(ori_w)
            exp_avg = ori_state['exp_avg']
            exp_avg_sq = ori_state['exp_avg_sq']
            if not isinstance(keep_idx, list):
                if not initialize:
                    if dim == 0:
                        exp_avg = exp_avg[keep_idx] if len(keep_idx.shape) == 1 else torch.gather(exp_avg, dim=dim, index=keep_idx)
                        exp_avg_sq = exp_avg_sq[keep_idx] if len(keep_idx.shape) == 1 else torch.gather(exp_avg_sq, dim=dim, index=keep_idx)
                    elif dim == -1:
                        exp_avg = exp_avg[..., keep_idx] if len(keep_idx.shape) == 1 else torch.gather(exp_avg, dim=dim, index=keep_idx)
                        exp_avg_sq = exp_avg_sq[..., keep_idx] if len(keep_idx.shape) == 1 else torch.gather(exp_avg_sq, dim=dim, index=keep_idx)
                    elif dim == 1:
                        exp_avg = exp_avg[:, keep_idx, ...] if len(keep_idx.shape) == 1 else torch.gather(exp_avg, dim=dim, index=keep_idx)
                        exp_avg_sq = exp_avg_sq[:, keep_idx, ...] if len(keep_idx.shape) == 1 else torch.gather(exp_avg_sq, dim=dim, index=keep_idx)
                    self.state[cur_w] = {
                        'step': ori_state['step'],
                        'exp_avg': exp_avg,
                        'exp_avg_sq': exp_avg_sq
                    }
                else:
                    self.state[cur_w] = {
                        'step': 0,
                        # Exponential moving average of gradient values
                        'exp_avg': torch.zeros_like(cur_w, memory_format=torch.preserve_format),
                        # Exponential moving average of squared gradient values
                        'exp_avg_sq': torch.zeros_like(cur_w, memory_format=torch.preserve_format)
                    }
            else:
                if not initialize:
                    assert isinstance(dim, list)
                    for i, d in enumerate(dim):
                        if d == 0:
                            exp_avg = exp_avg[keep_idx[i]] if len(keep_idx[i].shape) == 1 else torch.gather(exp_avg, dim=dim[i], index=keep_idx[i])
                            exp_avg_sq = exp_avg_sq[keep_idx[i]] if len(keep_idx[i].shape) == 1 else torch.gather(exp_avg_sq, dim=dim[i], index=keep_idx[i])
                        elif d == -1:
                            exp_avg = exp_avg[..., keep_idx[i]] if len(keep_idx[i].shape) == 1 else torch.gather(exp_avg, dim=dim[i], index=keep_idx[i])
                            exp_avg_sq = exp_avg_sq[..., keep_idx[i]] if len(keep_idx[i].shape) == 1 else torch.gather(exp_avg_sq, dim=dim[i], index=keep_idx[i])
                        elif d == 1:
                            exp_avg = exp_avg[:, keep_idx[i], ...] if len(keep_idx[i].shape) == 1 else torch.gather(exp_avg, dim=dim[i], index=keep_idx[i])
                            exp_avg_sq = exp_avg_sq[:, keep_idx[i], ...] if len(keep_idx[i].shape) == 1 else torch.gather(exp_avg_sq, dim=dim[i], index=keep_idx[i])
                    self.state[cur_w] = {
                        'step': ori_state['step'],
                        'exp_avg': exp_avg,
                        'exp_avg_sq': exp_avg_sq
                    }
                else:
                    self.state[cur_w] = {
                        'step': 0,
                        # Exponential moving average of gradient values
                        'exp_avg': torch.zeros_like(cur_w, memory_format=torch.preserve_format),
                        # Exponential moving average of squared gradient values
                        'exp_avg_sq': torch.zeros_like(cur_w, memory_format=torch.preserve_format)
                    }
        else:
            del self.param_names[group_idx][w_index]
            del self.param_groups[group_idx]['params'][w_index]
            self.state.pop(ori_w)
