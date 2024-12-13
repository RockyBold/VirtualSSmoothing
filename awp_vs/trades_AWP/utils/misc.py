#############################################################################################
#  Copyed and modified from https://github.com/bearpaw/pytorch-classification/blob/master/utils/misc.py  #
#############################################################################################

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

import errno
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


class RunningMeanStd(object):
    def __init__(self, dim=1):
        self._mean = np.zeros(dim)
        self._count = 0
        self._M = np.zeros(dim)
        self.dim = dim

    def update(self, x):
        """
        :param x: [n, d]
        :return:  None
        """
        if isinstance(x, list):
            x = np.array(x)

        avg_a = self._mean
        avg_b = np.mean(x, axis=0)

        count_a = self._count
        count_b = x.shape[0]

        delta = avg_b - avg_a
        m_a = self._M
        m_b = np.var(x, axis=0) * count_b
        M2 = m_a + m_b + np.power(delta, 2) * count_a * count_b / (count_a + count_b)

        self._mean += delta * count_b / (count_a + count_b)
        self._M = M2
        self._count += count_b

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        if self._count == 1:
            return np.ones(self.dim)
        return np.sqrt(self._M / (self._count - 1))


def get_mean_and_std_modified(dataset):
    # Compute the mean and std value online
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    rms = RunningMeanStd(dim=3)

    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        inputs = inputs.detach().cpu().numpy()
        inputs = inputs.transpose((0, 2, 3, 1)).reshape(-1, 3)
        rms.update(inputs)
    return rms.mean, rms.std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adaptive_cw_loss(logits, y, num_real_classes=10, num_out_classes=0, num_v_classes=0, reduction='mean'):
    out_conf = 0
    if num_out_classes > 0:
        out_conf = logits[:, num_real_classes:num_real_classes + num_out_classes].max(dim=1)[0]
    v_conf = 0
    if num_v_classes > 0:
        st = num_real_classes + num_out_classes
        end = st + num_v_classes
        v_conf = logits[:, st:end].max(dim=1)[0]
    null_conf = 0
    num_null_classes = logits.size(1) - num_real_classes - num_out_classes - num_v_classes
    if num_null_classes > 0:
        st = num_real_classes + num_out_classes + num_v_classes
        end = st + num_null_classes
        null_conf = logits[:, st:end].max(dim=1)[0]

    indcs = torch.arange(logits.size(0))
    with torch.no_grad():
        temp_logits = logits.clone()
        temp_logits[indcs, y] = -float('inf')
        in_max_ind = temp_logits[:, :num_real_classes].max(dim=1)[1]
    in_conf = logits[indcs, in_max_ind]
    y_corr = logits[indcs, y]
    losses = in_conf - y_corr - out_conf - v_conf - null_conf

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction'.format(reduction))


def cw_loss(logits, y, reduction='mean'):
    indcs = torch.arange(logits.size(0))
    with torch.no_grad():
        temp_logits = logits.clone()
        temp_logits[indcs, y] = -float('inf')
        in_max_ind = temp_logits.max(dim=1)[1]
    logit_other = logits[indcs, in_max_ind]
    logit_corr = logits[indcs, y]
    losses = logit_other - logit_corr

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'none':
        return losses
    else:
        raise ValueError('un-supported reduction'.format(reduction))


def pgd_attack(model, x, y, attack_steps, attack_lr=0.003, attack_eps=0.3, random_init=True, random_type='gussian',
               bn_type='eval', clamp=(0, 1), attack_loss='ce', num_real_classes=10, num_v_classes=-1):
    if bn_type=='eval':
        model.eval()
    elif bn_type=='train':
        model.train()
    else:
        raise  ValueError('error bn_type: {0}'.format(bn_type))
    x_adv = x.clone().detach()
    if random_init:
        # Flag to use random initialization
        if random_type == 'gussian':
            x_adv = x_adv + 0.001 * torch.randn(x.shape, device=x.device)
        elif random_type == 'uniform':
            # x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * attack_eps
            random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-attack_steps, attack_steps).to(x_adv.device)
            x_adv = x_adv + random_noise
        else:
            raise ValueError('error random noise type: {0}'.format(random_type))

    for i in range(attack_steps):
        x_adv.requires_grad = True

        model.zero_grad()
        adv_logits = model(x_adv)

        # Untargeted attacks - gradient ascent
        if attack_loss=='ce':
            loss = F.cross_entropy(adv_logits, y)
        elif attack_loss == 'cw':
            loss = cw_loss(adv_logits, y)
        elif attack_loss == 'acw':
            assert num_real_classes > 0 and num_v_classes >= 0
            loss = adaptive_cw_loss(adv_logits, y, num_real_classes=num_real_classes, num_out_classes=0,
                                    num_v_classes=num_v_classes, reduction='mean')
        else:
            raise ValueError('un-supported attack loss'.format(attack_loss))
        loss.backward()
        grad = x_adv.grad.detach()
        grad = grad.sign()
        x_adv = x_adv.detach()
        x_adv = x_adv + attack_lr * grad

        # Projection
        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
        x_adv = torch.clamp(x_adv, *clamp)
    # prob, pred = torch.max(logits, dim=1)
    return x_adv


def eval_after_pgd(model, test_loader, attack_steps, attack_lr, attack_eps, num_real_classes):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(non_blocking=True)
        batch_y = batch_y.cuda(non_blocking=True)

        input = pgd_attack(model, batch_x, batch_y, attack_steps, attack_lr, attack_eps, random_init=False,
                           random_type='uniform')

        # compute output
        with torch.no_grad():
            output = model(input)
        _, pred = torch.max(output[:, :num_real_classes], dim=1)
        correct += (pred == batch_y).sum()
        total += batch_y.size(0)
    acc = (float(correct) / total) * 100
    return acc


def eval_after_cw(model, test_loader, attack_steps, attack_lr, attack_eps, num_real_classes, num_v_classes):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(non_blocking=True)
        batch_y = batch_y.cuda(non_blocking=True)

        input = pgd_attack(model, batch_x, batch_y, attack_steps, attack_lr, attack_eps, random_init=False,
                           random_type='uniform', attack_loss='cw', num_real_classes=num_real_classes,
                           num_v_classes=num_v_classes)

        # compute output
        with torch.no_grad():
            output = model(input)
        _, pred = torch.max(output[:, :num_real_classes], dim=1)
        correct += (pred == batch_y).sum()
        total += batch_y.size(0)
    acc = (float(correct) / total) * 100
    return acc

def eval(model, test_loader, num_real_classes):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_loader):
        input, target = data
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(input)

        _, pred = torch.max(output[:, :num_real_classes], dim=1)
        correct += (pred == target).sum()
        total += target.size(0)
    acc = (float(correct) / total) * 100
    return acc
