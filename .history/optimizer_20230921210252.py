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
            t = 0
            m_t = 0
            v_t = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                t+=1
                alpha, betas, eps , wd, correct_bias = group['lr'], group['betas'], group['eps'], group['weight_decay'], group['correct_bias']
                beta1 = betas[0]
                beta2 = betas[1]
                #algorithsm
                m_t = beta1*m_t + (1-beta1)*p.grad.data
                v_t = beta2*v_t + (1-beta2)*p.grad.data**2
                if correct_bias:
                    m_hat = m_t/(1-beta1**t)
                    v_hat = v_t/(1-beta2**t)
                    p.data = p.data - alpha*m_hat/(torch.sqrt(v_hat)+eps) - wd*p.data
                    alpha = alpha - wd*alpha
                else: 
                    alpha_t = alpha*math.sqrt(1-beta2**t)/(1-beta1**t)
                    p.data = p.data - alpha_t*m_hat/(math.sqrt(v_hat)+eps) -wd*p.data
            
        return loss
