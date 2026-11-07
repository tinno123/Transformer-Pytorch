import math

from torch import  optim


class CosineDecayLR(object):
    def __init__(self, max_lr, min_lr, total_epochs,warmup_epochs=0.01):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = int(warmup_epochs * total_epochs)
        self.remain_epoch = self.total_epochs - self.warmup_epochs

    def warmup(self,epoch):
        if type(self.max_lr) == list:
            lr = []
            for i in self.max_lr:
                lr.append(i* epoch / self.warmup_epochs)
            return lr
        else:
            return self.max_lr * epoch / self.warmup_epochs

    def cosine_decay(self,epoch):
        current_epoch = epoch - self.warmup_epochs
        theta = (1 - current_epoch / self.remain_epoch) * math.pi

        if type(self.max_lr) == list:
            lr = []
            for maxlr , minlr in zip(self.max_lr,self.min_lr):
                lr.append(maxlr - (maxlr - minlr) * (1 + math.cos(theta)) / 2)

        else:
            lr = self.max_lr - (self.max_lr - self.min_lr) * (1 + math.cos(theta)) / 2
        return lr

    def update_lr(self,optimizer,lr):
        if type(lr) == list and type(optimizer) == list:
            for o,lr in zip(optimizer,lr):
                for param_group in o.param_groups:
                    param_group['lr'] = lr
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    def __call__(self, epoch,optimizer):

        epoch = min(epoch + 1, self.total_epochs)

        #----------------------------#
        # 热身阶段,LR线性上升
        # ---------------------------#
        if epoch <= self.warmup_epochs:
            lr = self.warmup(epoch)
        else:
        # ----------------------------#
        # 余弦退火阶段，从热身之后开始
        # ----------------------------#
            lr = self.cosine_decay(epoch)

        # ----------------------------#
        # 更新优化器的学习率
        # ----------------------------#
        if optimizer is not None:
            self.update_lr(optimizer,lr)

        return lr

class NoamLR(object):

    def __init__(self, optimizer, d_model, warmup_steps=4000, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._step = 0

    def get_lr(self):
        s = max(self._step, 1)
        return self.factor * self.d_model ** -0.5 * min(
            s ** -0.5, s * self.warmup_steps ** -1.5)

    def step(self,target_tokens,valid_tokens):
        self._step =valid_tokens / target_tokens * 100000
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def state_dict(self):
        return {'step': self._step}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']

    def __repr__(self):
        return (f'NoamLR(d_model={self.d_model}, warmup={self.warmup_steps}, '
                f'step={self._step}, lr={self.get_lr():.3e})')


def _unique_trainable(model):

    seen, params = set(), []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        params.append((name, p))
    return params


def GetOptimizer(model, optimizer_type, lr, momentum=None, weight_decay=0.0,
                 betas=None, eps=None):
    """构造优化器。
    Transformer 用法：
        GetOptimizer(model, 'adam', lr=0.0, betas=(0.9, 0.98), eps=1e-9)

    传统 CNN 用：
        GetOptimizer(model, 'adam', lr=1e-3, momentum=0.937, weight_decay=5e-4)
    """
    params = _unique_trainable(model)

    if weight_decay and weight_decay > 0:
        decay, no_decay = [], []
        for name, p in params:
            if p.ndim <= 1 or 'bias' in name or 'norm' in name.lower() or '.ln' in name:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
    else:
        groups = [{'params': [p for _, p in params]}]

    if optimizer_type == 'adam':
        if betas is None:
            betas = (momentum if momentum is not None else 0.9, 0.999)
        eps = 1e-8 if eps is None else eps
        return optim.Adam(groups, lr, betas=betas, eps=eps)

    if optimizer_type == 'sgd':
        return optim.SGD(groups, lr, momentum=momentum or 0.9, nesterov=True)

    raise ValueError(f'不支持的优化器类型：{optimizer_type}')