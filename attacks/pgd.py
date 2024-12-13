import torch
import torch.nn.functional as F
from autoattack.autopgd_base import APGDAttack

def autopgd_attack(model, x, y, loss, attack_steps, attack_eps, bn_type, seed, device):
    if bn_type == 'eval':
        model.eval()
    elif bn_type == 'train':
        model.train()
    else:
        raise ValueError('error bn_type: {0}'.format(bn_type))

    apgd = APGDAttack(model, loss=loss, n_restarts=1, n_iter=attack_steps, verbose=False, eps=attack_eps, norm='Linf',
                      eot_iter=1, rho=.75, seed=seed, device=device, is_tf_model=False)
    adv_curr = apgd.perturb(x, y)
    return adv_curr


def adaptive_cw_loss(logits, y, num_in_classes=10, num_out_classes=0, num_v_classes=0, attack_v=False,
                     reduction='mean'):
    logit_v = 0
    if num_v_classes > 0 and attack_v is True:
        st = num_in_classes + num_out_classes
        end = st + num_v_classes
        logit_v = logits[:, st:end].max(dim=1)[0]

    indcs = torch.arange(logits.size(0))
    with torch.no_grad():
        temp_logits = logits.clone()
        temp_logits[indcs, y] = -float('inf')
        in_max_ind = temp_logits[:, :num_in_classes].max(dim=1)[1]
    logit_in = logits[indcs, in_max_ind]
    logit_corr = logits[indcs, y]
    losses = logit_in - logit_corr - logit_v

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
               bn_type='eval', clamp=(0, 1), loss_str='pgd-ce', num_real_classes=-1, attack_v=False):
    if bn_type == 'eval':
        model.eval()
    elif bn_type == 'train':
        model.train()
    else:
        raise ValueError('error bn_type: {0}'.format(bn_type))

    if loss_str in ['pgd-ce-static-v', 'pgd-static-v']:
        nat_logits = model(x)
        nat_max_idx_in_v = nat_logits[:, num_real_classes:].max(dim=1)[1] + num_real_classes
    u_idx = torch.arange(0, x.size(0))

    x_adv = x.clone().detach()
    if random_init:
        # Flag to use random initialization
        if random_type == 'gussian':
            x_adv = x_adv + 0.001 * torch.randn(x.shape, device=x.device)
        elif random_type == 'uniform':
            # x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * attack_eps
            random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-attack_eps, attack_eps).to(x_adv.device)
            x_adv = x_adv + random_noise
        else:
            raise ValueError('error random noise type: {0}'.format(random_type))
        x_adv = torch.clamp(x_adv, *clamp)

    for i in range(attack_steps):
        x_adv.requires_grad = True

        model.zero_grad()
        adv_logits = model(x_adv)

        # Untargeted attacks - gradient ascent
        if loss_str == 'pgd-ce':
            loss = F.cross_entropy(adv_logits, y)
        elif loss_str == 'pgd-corr':
            loss = -adv_logits[u_idx, y]
            loss = loss.mean()
        elif loss_str == 'pgd-ce-dynamic-v':
            assert adv_logits.size(1) > num_real_classes
            max_idx_in_v = adv_logits[:, num_real_classes:].max(dim=1)[1] + num_real_classes
            loss = F.cross_entropy(adv_logits, max_idx_in_v)
        elif loss_str == 'pgd-ce-static-v':
            assert adv_logits.size(1) > num_real_classes
            loss = F.cross_entropy(adv_logits, nat_max_idx_in_v)
        elif loss_str == 'pgd-dynamic-v':
            assert adv_logits.size(1) > num_real_classes
            loss = -adv_logits[:, num_real_classes:].max(dim=1)[0]
            loss = loss.mean()
        elif loss_str == 'pgd-static-v':
            assert adv_logits.size(1) > num_real_classes
            loss = -adv_logits[u_idx, nat_max_idx_in_v]
            loss = loss.mean()
        elif loss_str == 'pgd-acw':
            assert num_real_classes > 0
            num_v_classes = adv_logits.size(1) - num_real_classes
            loss = adaptive_cw_loss(adv_logits, y, num_in_classes=num_real_classes, num_v_classes=num_v_classes,
                                    attack_v=attack_v)
        elif loss_str == 'pgd-cw':
            loss = cw_loss(adv_logits, y)
        else:
            raise ValueError('un-supported adv loss:'.format(loss_str))
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


def cross_entropy_soft_target(logit, y_soft):
    batch_size = logit.size()[0]
    log_prob = F.log_softmax(logit, dim=1)
    loss = -torch.sum(log_prob * y_soft) / batch_size
    return loss


def pgd_attack_misc(model, x, y, num_in_classes, num_out_classes=0, attack_steps=10, attack_lr=0.003,
                    attack_eps=0.3, random_init=True, random_type='uniform', bn_type='eval', clamp=(0, 1),
                    loss_str='pgd-oe-in', worst_elem=False):
    if attack_eps <= 0.0:
        return x
    if bn_type == 'eval':
        model.eval()
    elif bn_type == 'train':
        model.train()
    else:
        raise ValueError('error bn_type: {0}'.format(bn_type))
    x_adv = x.clone().detach()
    if random_init:
        # Flag to use random initialization
        if random_type == 'gussian':
            x_adv = x_adv + 0.001 * torch.randn(x.shape, device=x.device)
        elif random_type == 'uniform':
            # x_adv = x_adv + (torch.rand(x.size(), dtype=x.dtype, device=x.device) - 0.5) * 2 * attack_eps
            random_noise = torch.FloatTensor(*x_adv.shape).uniform_(-attack_eps, attack_eps).to(x_adv.device)
            x_adv = x_adv + random_noise
        else:
            raise ValueError('error random noise type: {0}'.format(random_type))

    if worst_elem:
        max_in_scores = torch.zeros((len(x),)).to(x.device)
        adv_x_ret = x_adv.detach().clone()

    for i in range(attack_steps):
        x_adv.requires_grad = True
        model.zero_grad()
        adv_logits = model(x_adv)
        # Untargeted attacks - gradient ascent
        if loss_str == 'pgd-ce':
            if y is not None and len(y.size()) > 1:
                raise ValueError('please pass hard labels when using {} as adv loss'.format(loss_str))
            loss = F.cross_entropy(adv_logits, y)
        elif loss_str == 'pgd-ce-in':
            with torch.no_grad():
                nat_outputs = F.softmax(model(x), dim=1)
            loss = cross_entropy_soft_target(adv_logits[:, :num_in_classes], nat_outputs[:, :num_in_classes])
        elif loss_str == 'pgd-ce-real':
            num_real_classes = num_in_classes + num_out_classes
            with torch.no_grad():
                nat_outputs = F.softmax(model(x), dim=1)
            loss = cross_entropy_soft_target(adv_logits[:, :num_real_classes], nat_outputs[:, :num_real_classes])
        elif loss_str == 'pgd-ce-all':
            with torch.no_grad():
                nat_outputs = F.softmax(model(x), dim=1)
            loss = cross_entropy_soft_target(adv_logits, nat_outputs)
        elif loss_str == 'pgd-oe-in':
            if y is not None and len(y.size()) <= 1:
                raise ValueError('please pass soft labels when using {} as adv loss'.format(loss_str))
            # loss = OE_loss(adv_logits[:, :num_in_classes])
            loss = cross_entropy_soft_target(adv_logits[:, :num_in_classes], y[:, :num_in_classes])
        elif loss_str == 'pgd-oe-out':
            if y is not None and len(y.size()) <= 1:
                raise ValueError('please pass soft labels when using {} as adv loss'.format(loss_str))
            num_real_classes = num_in_classes + num_out_classes
            loss = cross_entropy_soft_target(adv_logits[:, num_in_classes:num_real_classes],
                                                     y[:, num_in_classes:num_real_classes])
        elif loss_str == 'pgd-oe-real':
            if y is not None and len(y.size()) <= 1:
                raise ValueError('please pass soft labels when using {} as adv loss'.format(loss_str))
            num_real_classes = num_in_classes + num_out_classes
            loss = cross_entropy_soft_target(adv_logits[:, :num_real_classes], y[:, :num_real_classes])
        elif loss_str == 'pgd-oe-all':
            if y is not None and len(y.size()) <= 1:
                raise ValueError('please pass soft labels when using {} as adv loss'.format(loss_str))
            loss = cross_entropy_soft_target(adv_logits, y)
        elif loss_str == 'pgd-in-max':
            with torch.no_grad():
                _, preds = torch.max(adv_logits[:, :num_in_classes], dim=1)
            loss = -F.cross_entropy(adv_logits, preds)
        elif loss_str == 'pgd-ce-t':
            if y is None:
                raise ValueError('target hard label should be offered when using {} as adv loss!'.format(loss_str))
            if len(y.size()) > 1:
                raise ValueError('please pass hard labels when using {} as adv loss'.format(loss_str))
            loss = -F.cross_entropy(adv_logits, y)
        else:
            raise ValueError('un-supported loss: {}'.format(loss_str))

        loss.backward()
        loss = loss.detach()  # relase video memory
        grad = x_adv.grad.detach()
        grad = grad.sign()
        x_adv = x_adv.detach()
        x_adv = x_adv + attack_lr * grad

        # Projection
        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
        x_adv = torch.clamp(x_adv, *clamp)

        if worst_elem:
            with torch.no_grad():
                model.eval()
                adv_outputs = F.softmax(model(x_adv), dim=1)
                scores, _ = torch.max(adv_outputs[:, :num_in_classes], dim=1)
                indics = scores > max_in_scores
                adv_x_ret[indics] = x_adv[indics].detach()
                max_in_scores[indics] = scores[indics]

    if worst_elem:
        return adv_x_ret

    return x_adv


def eval_pgdadv(model, test_loader, attack_steps, attack_lr, attack_eps, num_real_classes, loss_str='pgd-ce',
                norm='Linf', attack_v=False, device=torch.device("cuda")):
    model.eval()
    adv_corr = 0
    total = 0
    num_max_in_v = 0
    num_corr_max_in_v = 0
    corr_conf = []
    corr_v_conf = []
    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        u_idx = torch.arange(0, len(batch_y))
        if norm == 'Linf':
            if loss_str == 'trades':
                adv_batch_x = trades_pgd_attack(model, batch_x, attack_steps, attack_lr, attack_eps, random_init=False)
            else:
                adv_batch_x = pgd_attack(model, batch_x, batch_y, attack_steps, attack_lr, attack_eps, bn_type='eval',
                                         random_type='uniform', num_real_classes=num_real_classes, loss_str=loss_str,
                                         attack_v=attack_v, random_init=False)
        # elif norm == 'L2':
        #     adv_batch_delta = attack_pgd_pro(model, batch_x, batch_y, attack_eps, attack_lr, attack_steps, restarts=1,
        #                                      norm=norm, bn_type='eval', random_type='uniform')
        #     adv_batch_x = torch.clamp(batch_x + adv_batch_delta.detach(), min=0, max=1)
        else:
            raise ValueError('unsupported norm: {0}'.format(norm))
        # adv_batch_delta = attack_pgd_pro(model, batch_x, batch_y, attack_eps, attack_lr, attack_steps, restarts=1,
        #                                  norm=norm, bn_type='eval', random_type='uniform')
        # adv_batch_x = torch.clamp(batch_x + adv_batch_delta.detach(), min=0, max=1)

        # compute output
        with torch.no_grad():
            adv_logits = model(adv_batch_x)
            adv_conf = F.softmax(adv_logits, dim=1)
            nat_logits = model(batch_x)


        _, adv_pred = torch.max(adv_logits[:, :num_real_classes], dim=1)
        adv_corr_idx = adv_pred == batch_y
        adv_corr += adv_corr_idx.sum()
        total += batch_y.size(0)

        max_in_v_idx = adv_logits.max(dim=1)[1] >= num_real_classes
        num_max_in_v += max_in_v_idx.sum().item()
        num_corr_max_in_v += (torch.logical_and(adv_corr_idx, max_in_v_idx)).sum().item()

        nat_corr_idx = nat_logits[:, :num_real_classes].max(dim=1)[1] == batch_y
        corr_conf.append(adv_conf[u_idx, batch_y][nat_corr_idx])
        corr_v_conf.append(adv_conf[nat_corr_idx, num_real_classes:])
    corr_conf = torch.cat(corr_conf, dim=0)
    corr_v_conf = torch.cat(corr_v_conf, dim=0)
    adv_acc = (float(adv_corr) / total) * 100
    return adv_acc, num_max_in_v, num_corr_max_in_v, corr_conf, corr_v_conf


def trades_pgd_attack(model, x, attack_steps, attack_lr=0.003, attack_eps=0.3, random_init=True, clamp=(0, 1),
                      num_real_classes=10, attack_loss='trades', y_soft=None):
    model.eval()
    x_adv = x.clone().detach()
    if random_init:
        # Flag to use random initialization
        x_adv = x_adv + 0.001 * torch.randn(x.shape, device=x.device)

    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    for i in range(attack_steps):
        x_adv.requires_grad = True
        model.zero_grad()

        # attack
        logits = model(x)
        adv_logits = model(x_adv)
        logsoft_adv = F.log_softmax(adv_logits, dim=1)
        if attack_loss == 'trades':
            soft_nat = F.softmax(logits, dim=1)
            loss_kl = criterion_kl(logsoft_adv, soft_nat)
        elif attack_loss == 'trades-atlic' and y_soft is not None:
            alpha = y_soft[0][num_real_classes:].sum()
            if alpha > 0:
                soft_nat_real = F.softmax(logits[:, :num_real_classes], dim=1) * (1 - alpha)
                soft_nat = torch.zeros_like(logits)
                soft_nat[:, :num_real_classes] = soft_nat_real
                soft_nat[:, num_real_classes:] = y_soft[:, num_real_classes:]
            else:
                soft_nat = F.softmax(logits, dim=1)

            loss_kl = criterion_kl(logsoft_adv, soft_nat)
        else:
            raise ValueError('unsupported parameter combination, attack_loss:{}, y_soft: {}'
                             .format(attack_loss, y_soft))

        loss_kl.backward()
        grad = x_adv.grad.detach()
        grad = grad.sign()
        x_adv = x_adv.detach() + attack_lr * grad

        # Projection
        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
        x_adv = torch.clamp(x_adv, *clamp)
    # prob, pred = torch.max(logits, dim=1)
    return x_adv


# def generate_advdata(model, test_loader, attack_test_steps, attack_lr, attack_eps, norm='Linf'):
#     model.eval()
#     adv_x = None
#     y = None
#     for i, data in enumerate(test_loader):
#         batch_x, batch_y = data
#         batch_x = batch_x.cuda(non_blocking=True)
#         batch_y = batch_y.cuda(non_blocking=True)
#         if norm == 'Linf':
#             adv_batch_x = pgd_attack(model, batch_x, batch_y, attack_test_steps, attack_lr, attack_eps, bn_type='eval',
#                                      random_type='uniform')
#         # elif norm == 'L2':
#         #     adv_batch_delta = attack_pgd_pro(model, batch_x, batch_y, attack_eps, attack_lr, attack_test_steps,
#         #                                      restarts=1, norm=norm, bn_type='eval', random_type='uniform')
#         #     adv_batch_x = torch.clamp(batch_x + adv_batch_delta.detach(), min=0, max=1)
#         else:
#             raise ValueError('unsupported norm: {0}'.format(norm))
#         # adv_batch_delta = attack_pgd_pro(model, batch_x, batch_y, attack_eps, attack_lr, attack_test_steps, norm=norm,
#         #                                  bn_type='eval', random_type='uniform')
#         # adv_batch_x = torch.clamp((batch_x + adv_batch_delta), min=0, max=1)
#
#         if adv_x is None:
#             adv_x = adv_batch_x
#         else:
#             adv_x = torch.cat((adv_x, adv_batch_x), 0)
#
#         if y is None:
#             y = batch_y
#         else:
#             y = torch.cat((y, batch_y), 0)
#     return adv_x, y


def rslad_inner(model, teacher_logits, x_natural, step_size=0.003, epsilon=0.031, perturb_steps=10):
    # define KL-loss
    criterion_kl = torch.nn.KLDivLoss(size_average=False, reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv
