import torch
from typing import Iterable, Callable, Dict
import math
from torch.optim import Optimizer
class AdamW(Optimizer):
    def __init__(self,
                 params = Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3,
                 betas: tuple[float,float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 correct_bias: bool = True):
        if not lr>= 0:
            raise ("learning rate {} must be >= 0".format(lr))
        if not 0<=betas[0] <1:
            raise ("beta1 {} must be >=0 and <1".format(betas[0]))
        if not 0<=betas[1] <1:
            raise ("beta2 {} must be >=0 and <1".format(betas[1]))
        if not eps>=0:
            raise ("epsilon {} must be >=0".format(eps))
        defaults = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay, correct_bias = correct_bias)
        super().__init__(params, defaults)
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            alpha, betas, eps , wd, correct_bias = group['lr'], group['betas'], group['eps'], group['weight_decay'], group['correct_bias']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                state = self.state
                if len(state)==0:
                    state['t'] = 0
                    state['m_t'] = torch.zeros_like(p.data)
                    state['v_t'] = torch.zeros_like(p.data)
                state['t']+=1
                beta1, beta2 = betas
                m_t = state['m_t']
                v_t = state['v_t']
                t = state['t']
                # #algorithsm
                m_t = beta1*m_t + (1.0-beta1)*p.grad
                state['m_t'] = m_t
                # v_t = beta2*v_t + (1-beta2)*p.grad*p.grad 
                # if correct_bias:
                #     alpha = alpha*math.sqrt(1-beta2**t)/(1-beta1**t)
                #     p.data = p.data - alpha*m_t/(torch.sqrt(v_t)+eps)
                # else: 
                #     m_hat = m_t/(1-beta1**t)
                #     v_hat = v_t/(1-beta2**t)
                #     p.data = p.data - alpha*m_hat/(math.sqrt(v_hat)+eps)
                # p.data -= wd*alpha*p.data
                # if len(state) == 0:
                #     state['step'] = 0
                #     # Exponential moving average of gradient values
                #     state['exp_avg'] = torch.zeros_like(p.data)
                #     # Exponential moving average of squared gradient values
                #     state['exp_avg_sq'] = torch.zeros_like(p.data)
                # exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # beta1, beta2 = group['betas']

                # state['step'] += 1
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                #m_t.mul_(beta1).add_(1.0 - beta1, p.grad)
                v_t.mul_(beta2).addcmul_(1.0 - beta2, p.grad, p.grad)
                denom = v_t.sqrt().add_(group['eps'])

                step_size = group['lr']
                if group['correct_bias']:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state['t']
                    bias_correction2 = 1.0 - beta2 ** state['t']
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, m_t, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group['weight_decay'] > 0.0:
                    p.data.add_(-group['lr'] * group['weight_decay'], p.data)
        return loss
