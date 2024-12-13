import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


def cross_entropy_soft_target(logit, y_soft):
    batch_size = logit.size()[0]
    log_prob = F.log_softmax(logit, dim=1)
    loss = -torch.sum(log_prob * y_soft) / batch_size
    return loss


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta, loss_str='trades'):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        if len(targets.size()) == 1:
            loss_ce_nat = F.cross_entropy(self.proxy(inputs_clean), targets)
        elif len(targets.size()) == 2:
            loss_ce_nat = cross_entropy_soft_target(self.proxy(inputs_clean), targets)
        else:
            raise ValueError('error dim ({0}) of targets'.format(len(targets.size())))
        loss_kl_rob = F.kl_div(F.log_softmax(self.proxy(inputs_adv), dim=1),
                               F.softmax(self.proxy(inputs_clean), dim=1),
                               reduction='batchmean')
        if loss_str=='trades':
            loss = - 1.0 * (loss_ce_nat + beta * loss_kl_rob)
        elif loss_str=='trades-plus':
            if len(targets.size()) == 1:
                loss_ce_adv = F.cross_entropy(self.proxy(inputs_adv), targets)
            elif len(targets.size()) == 2:
                loss_ce_adv = cross_entropy_soft_target(self.proxy(inputs_adv), targets)
            else:
                raise ValueError('error dim ({0}) of targets'.format(len(targets.size())))
            loss = - 1.0 * (loss_ce_nat + loss_ce_adv + beta * loss_kl_rob)
        else:
            raise ValueError('unsupported loss_str: {}'.format(loss_str))

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)




