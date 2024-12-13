import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def trades_pgd_attack(model, x, attack_steps, attack_lr=0.003, attack_eps=0.3, random_init=True, clamp=(0, 1),
                      num_in_classes=10, attack_loss='trades'):
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
        if attack_loss == 'trades':
            soft_nat = F.softmax(logits, dim=1)
            logsoft_adv = F.log_softmax(adv_logits, dim=1)
            loss_kl = criterion_kl(logsoft_adv, soft_nat)
        elif attack_loss == 'trades-in':
            # soft_nat = F.softmax(logits, dim=1)
            # logsoft_adv = F.log_softmax(adv_logits, dim=1)
            # loss_kl = criterion_kl(logsoft_adv[:, :num_in_classes], soft_nat[:, :num_in_classes])

            soft_nat = F.softmax(logits[:, :num_in_classes], dim=1)
            logsoft_adv = F.log_softmax(adv_logits[:, :num_in_classes], dim=1)
            loss_kl = criterion_kl(logsoft_adv, soft_nat)

        # elif attack_loss == 'trades-atlic' and y_soft is not None:
        #     alpha = y_soft[0][num_in_classes:].sum()
        #     if alpha > 0:
        #         soft_nat_real = F.softmax(logits[:, :num_in_classes], dim=1) * (1 - alpha)
        #         soft_nat = torch.zeros_like(logits)
        #         soft_nat[:, :num_in_classes] = soft_nat_real
        #         soft_nat[:, num_in_classes:] = y_soft[:, num_in_classes:]
        #     else:
        #         soft_nat = F.softmax(logits, dim=1)
        # 
        #     loss_kl = criterion_kl(logsoft_adv, soft_nat)
        # else:
        #     raise ValueError('unsupported parameter combination, attack_loss:{}, y_soft: {}'
        #                      .format(attack_loss, y_soft))

        loss_kl.backward()
        grad = x_adv.grad.detach()
        grad = grad.sign()
        x_adv = x_adv.detach() + attack_lr * grad

        # Projection
        x_adv = x + torch.clamp(x_adv - x, min=-attack_eps, max=attack_eps)
        x_adv = torch.clamp(x_adv, *clamp)
    # prob, pred = torch.max(logits, dim=1)
    return x_adv


def trades_plus_loss(nat_logits, adv_logits, y_soft, beta=6.0, nat_ce_w=1.0, adv_ce_w=1.0, temp=1):
    if nat_ce_w == 0:
        ce_loss_nat = 0
    else:
        batch_size = nat_logits.size()[0]
        ce_loss_nat = (1.0 / batch_size) * (-torch.sum(F.log_softmax(nat_logits, dim=1) * y_soft))

    if adv_ce_w == 0:
        ce_loss_adv = 0
    else:
        batch_size = adv_logits.size()[0]
        ce_loss_adv = (1.0 / batch_size) * (-torch.sum(F.log_softmax(adv_logits, dim=1) * y_soft))

    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    kl_loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits / temp, dim=1),
                                                F.softmax(nat_logits / temp, dim=1))

    return nat_ce_w * ce_loss_nat + adv_ce_w * ce_loss_adv + beta * (kl_loss)


def trades_loss(nat_logits, adv_logits, y_soft, beta=6.0, cal_classes=-1):
    batch_size = nat_logits.size()[0]
    loss_natural = (1.0 / batch_size) * (-torch.sum(F.log_softmax(nat_logits, dim=1) * y_soft))

    # argmaxes = torch.argmax(confused_labels, dim=1)
    # loss_natural1 = F.cross_entropy(nat_logits, argmaxes)

    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    if cal_classes > 0:
        assert cal_classes <= adv_logits.size(1)
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits[:, :cal_classes], dim=1),
                                                        F.softmax(nat_logits[:, :cal_classes], dim=1))
        # loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1)[:, :cal_classes],
        #                                                 F.softmax(nat_logits, dim=1)[:, :cal_classes])
    else:
        loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(nat_logits, dim=1))

    # print('loss_natural:', loss_natural.item(), 'loss_robust:', loss_robust.item(), 'loss_natural/(beta *loss_robust):',
    #       (loss_natural / (beta * loss_robust)).item())

    return loss_natural + beta * loss_robust, loss_natural, loss_robust


def trades_atlic_loss(nat_logits, adv_logits, y_soft, beta=6.0, num_in_classes=10):
    batch_size = nat_logits.size()[0]
    loss_nat = (1.0 / batch_size) * (-torch.sum(F.log_softmax(nat_logits, dim=1) * y_soft))

    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    logsoft_adv = F.log_softmax(adv_logits, dim=1)
    # print('logsoft_adv:', logsoft_adv[0:5])

    alpha = y_soft[0][num_in_classes:].sum()
    if alpha > 0:
        soft_nat_real = F.softmax(nat_logits[:, :num_in_classes], dim=1) * (1 - alpha)
        soft_nat = torch.zeros_like(nat_logits)
        soft_nat[:, :num_in_classes] = soft_nat_real
        soft_nat[:, num_in_classes:] = y_soft[:, num_in_classes:]
        # print('soft_nat:', soft_nat[0:5])
    else:
        soft_nat = F.softmax(nat_logits, dim=1)

    loss_rob = (1.0 / batch_size) * criterion_kl(logsoft_adv, soft_nat)

    return loss_nat + beta * loss_rob


def trades_atlic_using_whole_logits_loss(nat_logits, adv_logits, y_soft, beta=6.0, num_in_classes=10):
    batch_size = nat_logits.size()[0]
    loss_nat = (1.0 / batch_size) * (-torch.sum(F.log_softmax(nat_logits, dim=1) * y_soft))

    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    logsoft_adv = F.log_softmax(adv_logits, dim=1)
    print('logsoft_adv:', logsoft_adv[0:5])

    alpha = y_soft[:, num_in_classes:].sum(dim=1)
    soft_nat_whole = F.softmax(nat_logits, dim=1)
    print('org soft_nat_whole:', soft_nat_whole[0:5])
    sum_of_soft_nat_real = soft_nat_whole[:, :num_in_classes].sum(dim=1)
    factor = ((1-alpha) / sum_of_soft_nat_real).unsqueeze(dim=1)
    soft_nat_real = soft_nat_whole[:, :num_in_classes] * factor
    print('soft_nat_real:', soft_nat_real[0:5])
    print('soft_nat_real.sum():', soft_nat_real[0:5].sum(dim=1))

    soft_nat = y_soft.clone()
    soft_nat[:, :num_in_classes] = soft_nat_real
    print('soft_nat:', soft_nat[0:5])

    loss_rob = (1.0 / batch_size) * criterion_kl(logsoft_adv, soft_nat)

    return loss_nat + beta * loss_rob



# def trades_attack_pro(model, x_natural, step_size=0.003, epsilon=0.031, perturb_steps=10, distance='Linf'):
#     model.eval()
#     batch_size = len(x_natural)
#     if distance == 'Linf':
#         x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
#         for _ in range(perturb_steps):
#             x_adv.requires_grad_()
#             with torch.enable_grad():
#                 loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
#                                    F.softmax(model(x_natural), dim=1),
#                                    reduction='sum')
#             grad = torch.autograd.grad(loss_kl, [x_adv])[0]
#             x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
#             x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
#             x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     elif distance == 'L2':
#         delta = 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
#         delta = Variable(delta.data, requires_grad=True)
#
#         # Setup optimizers
#         optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)
#
#         for _ in range(perturb_steps):
#             adv = x_natural + delta
#
#             # optimize
#             optimizer_delta.zero_grad()
#             with torch.enable_grad():
#                 loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
#                                        F.softmax(model(x_natural), dim=1),
#                                        reduction='sum')
#             loss.backward()
#             # renorming gradient
#             grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
#             delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
#             optimizer_delta.step()
#
#             # projection
#             delta.data.add_(x_natural)
#             delta.data.clamp_(0, 1).sub_(x_natural)
#             delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
#         x_adv = Variable(x_natural + delta, requires_grad=False)
#     else:
#         x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
#         x_adv = torch.clamp(x_adv, 0.0, 1.0)
#     return x_adv
