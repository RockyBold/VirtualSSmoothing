import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import os

from wideresnet import WideResNet
from preactresnet import PreActResNet18
from utils_awp import AdvWeightPerturb

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet-18', help='models: resnet-18 or wrn-34-10')
parser.add_argument('--dataset', default='cifar10', help='dataset: cifar10 or cifar100')
parser.add_argument('--l2', default=0, type=float)
parser.add_argument('--l1', default=0, type=float)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--data-dir', default='../cifar-data', type=str)
parser.add_argument('--epochs', default=200, type=int)
# parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise'])
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'none'])
parser.add_argument('--epsilon', default=0.031, type=float)
parser.add_argument('--attack-iters', default=10, type=int)
parser.add_argument('--attack-iters-test', default=20, type=int)
parser.add_argument('--restarts', default=1, type=int)
parser.add_argument('--pgd-alpha', default=0.00784, type=float)
parser.add_argument('--norm', default='Linf', type=str, choices=['Linf', 'L2'])
parser.add_argument('--fname', default='cifar_model', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--half', action='store_true')
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--chkpt-iters', default=1, type=int)
parser.add_argument('--awp-gamma', default=0.01, type=float)
parser.add_argument('--awp-warmup', default=0, type=int)
parser.add_argument('--alpha', default=0.5, type=float, help='total confidences of virtual classes')
parser.add_argument('--v_classes', default=10, type=int,
                    help='the number of additional virtual nodes in the output layer')
parser.add_argument('--add_noise', action='store_true', default=False, help='add noise to virtual classes or not')
parser.add_argument('--v_type', default='u', type=str, help='u or n')
parser.add_argument('--vs_warmup', default=0, type=int, help='apply ATLIC soft-lables after some epochs for accelerating.')
parser.add_argument('--bn_type', default='train', help='type of batch normalization during attack: train or eval')
parser.add_argument('--random_type', default='uniform', help='random type of pgd: uniform or gussian')

args = parser.parse_args()
if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.epsilon >= 1:
    args.epsilon = args.epsilon / 255
if args.pgd_alpha >= 1:
    args.pgd_alpha = args.pgd_alpha / 255

if args.dataset == 'cifar10':
    NUM_REAL_CLASSES = 10
elif args.dataset == 'cifar100':
    NUM_REAL_CLASSES = 100
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))
NUM_EXAMPLES = 50000

if not os.path.exists(args.fname):
    os.makedirs(args.fname)

use_cuda = torch.cuda.is_available()

# CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
# CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR_MEAN = (0.0, 0.0, 0.0)
CIFAR_STD = (1, 1, 1)

mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()


def normalize(X):
    return (X - mu) / std


upper_limit, lower_limit = 1, 0

# def clamp(X, lower_limit, upper_limit):
#     return torch.max(torch.min(X, upper_limit), lower_limit)


# def mixup_data(x, y, alpha=1.0):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1
#
#     batch_size = x.size()[0]
#     index = torch.randperm(batch_size).cuda()
#
#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]
#     return mixed_x, y_a, y_b, lam


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, norm, early_stop=False, mixup=False, y_a=None,
               y_b=None, lam=None, bn_type='train', random_type='uniform'):
    upper_limit, lower_limit = 1, 0

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    tmpmean = (0.0, 0.0, 0.0)
    tmpstd = (1, 1, 1)
    mu = torch.tensor(tmpmean).view(3, 1, 1).cuda()
    std = torch.tensor(tmpstd).view(3, 1, 1).cuda()
    def normalize(X):
        return (X - mu) / std

    def clamp(X, lower_limit, upper_limit):
        return torch.max(torch.min(X, upper_limit), lower_limit)

    if bn_type == 'eval':
        model.eval()
    elif bn_type == 'train':
        model.train()
    else:
        raise ValueError('error bn_type: {0}'.format(bn_type))

    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "Linf":
            if random_type == 'uniform':
                delta.uniform_(-epsilon, epsilon)
            elif random_type == 'gussian':
                delta = 0.001 * torch.randn(X.shape, device=X.device)
            else:
                raise ValueError('error random noise type: {0}'.format(random_type))
        elif norm == "L2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X + delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "Linf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "L2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)

            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X + delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X + delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    return max_delta


def add_noise_to_uniform(unif):
    assert len(unif.size()) == 2
    unif_elem = unif.float().mean()
    new_unif = unif.clone()
    new_unif.uniform_(unif_elem - 0.01 * unif_elem, unif_elem + 0.01 * unif_elem)
    factor = new_unif.sum(dim=1) / unif.sum(dim=1)
    new_unif = new_unif / factor.unsqueeze(dim=1)
    # sum=new_unif.sum(dim=1)
    return new_unif


def ls_smoothing(true_label, alpha, num_classes):
    y_ls = F.one_hot(true_label, num_classes=num_classes) * (1 - alpha) + (alpha / (num_classes))
    y_ls = y_ls.to(true_label.device)
    return y_ls


def constuct_vs_label(hard_label, alpha, num_classes, v_classes, add_noise=False, v_type='u', sub_alpha=-1):
    # one-hot
    if alpha == 0:
        return F.one_hot(hard_label, num_classes=num_classes + v_classes).to(hard_label.device)
    # standard label smoothing
    elif alpha != 0 and v_classes == 0:
        return  ls_smoothing(hard_label, alpha, num_classes)
    # with additional virtuall classes
    elif alpha != 0 and v_classes != 0:
        y_vc = torch.zeros((len(hard_label), num_classes + v_classes), device=hard_label.device)
        u = [i for i in range(len(hard_label))]
        y_vc[u, hard_label] += (1 - alpha)
        if v_type == 'u':
            if sub_alpha > 0:
                assert num_classes == v_classes
                y_vc[:, num_classes:num_classes + v_classes] = alpha * ls_smoothing(hard_label, sub_alpha, v_classes)
                if add_noise:
                    raise NotImplementedError
            else:
                temp_v_conf = alpha / v_classes
                y_vc[:, num_classes:num_classes + v_classes] = temp_v_conf
                if add_noise:
                    y_vc[:, num_classes:num_classes + v_classes] = add_noise_to_uniform(
                        y_vc[:, num_classes:num_classes + v_classes])
        elif v_type == 'n':
            v_conf = torch.randint(low=0, high=100 * v_classes, size=(len(hard_label), v_classes)).float().to(
                hard_label.device)
            v_conf = v_conf / v_conf.sum(dim=1).unsqueeze(dim=1)
            # print('v_conf.sum(dim=1):', v_conf.sum(dim=1))
            y_vc[:, num_classes:num_classes + v_classes] = alpha * v_conf
        return y_vc
    else:
        raise ValueError('error alpha:{0} or v_classes:{1}'.format(alpha, v_classes))


def cross_entropy_soft_target(logit, y_soft):
    batch_size = logit.size()[0]
    log_prob = F.log_softmax(logit, dim=1)
    loss = -torch.sum(log_prob * y_soft) / batch_size
    return loss


def eval(model, test_loader, num_org_classes):
    model.eval()
    correct = 0
    total = 0
    vnodes = 0
    for i, data in enumerate(test_loader):
        input, target = data
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        with torch.no_grad():
            output = model(input)

        _, pred = torch.max(output, dim=1)
        correct += (pred == target).sum()
        total += target.size(0)
        vnodes += (pred >= num_org_classes).sum()
    accuracy = (float(correct) / total) * 100
    return accuracy, vnodes.item()


def get_model(model_name, num_real_classes=10, num_v_classes=0):
    if model_name == 'resnet-18':
        model = PreActResNet18(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
    elif model_name == 'wrn-34-10':
        model = WideResNet(34, widen_factor=10, num_real_classes=num_real_classes, num_v_classes=num_v_classes)
    else:
        raise ValueError("Unknown model")
    return model


def eval_test_rob(model, data_loader, args, num_real_classes=10):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0
    # test_adv_vnodes = 0

    for i, (X, y) in enumerate(data_loader):
        X, y = X.cuda(), y.cuda()
        # Random initialization
        if args.attack == 'none':
            delta = torch.zeros_like(X)
        else:
            delta = attack_pgd(model, X, y, args.epsilon, args.pgd_alpha, args.attack_iters_test, args.restarts,
                               args.norm, bn_type='eval', random_type='uniform')
        delta = delta.detach()

        robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
        robust_loss = criterion(robust_output, y)
        test_robust_loss += robust_loss.item() * y.size(0)
        test_robust_acc += (robust_output[:, :num_real_classes].max(1)[1] == y).sum().item()
        # _, adv_pred = torch.max(robust_output, dim=1)
        # test_adv_vnodes += (adv_pred >= NUM_REAL_CLASSES).sum().item()
        test_n += y.size(0)

    return test_n, test_robust_loss, test_robust_acc


def eval_data(model, data_loader, num_real_classes=10):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0
    test_acc = 0
    test_n = 0
    # test_nat_vnodes = 0
    for i, (X, y) in enumerate(data_loader):
        X, y = X.cuda(), y.cuda()
        output = model(normalize(X))
        loss = criterion(output, y)
        test_loss += loss.item() * y.size(0)
        test_acc += (output[:, :num_real_classes].max(1)[1] == y).sum().item()
        test_n += y.size(0)
        # _, nat_pred = torch.max(output, dim=1)
        # test_nat_vnodes += (nat_pred >= NUM_REAL_CLASSES).sum().item()
    return test_n, test_loss, test_acc


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    epoch_lr = args.lr
    for i in range(0, len(args.schedule)):
        if epoch > args.schedule[i]:
            epoch_lr = args.lr * np.power(args.gamma, (i + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch_lr
    return epoch_lr


def main():
    print('================================================================')
    print(args)
    print('================================================================')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32),
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    num_workers = 2
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            args.data_dir, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            args.data_dir, train=False, transform=test_transform, download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            args.data_dir, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            args.data_dir, train=False, transform=test_transform, download=True)
    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    model = get_model(args.model, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes).cuda()
    proxy = get_model(args.model, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes).cuda()

    if args.l2:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params': decay, 'weight_decay': args.l2},
                  {'params': no_decay, 'weight_decay': 0}]
    else:
        params = model.parameters()

    opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=5e-4)
    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
    awp_adversary = AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)

    # if args.lr_schedule == 'superconverge':
    #     lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    #     # lr_schedule = lambda t: np.interp([t], [0, args.epochs], [0, args.lr_max])[0]
    # elif args.lr_schedule == 'piecewise':
    #     def lr_schedule(t):
    #         if t / args.epochs < 0.5:
    #             return args.lr_max
    #         elif t / args.epochs < 0.75:
    #             return args.lr_max / 10.
    #         else:
    #             return args.lr_max / 100.

    best_test_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(args.fname, f'model_{start_epoch - 1}.pth')))
        opt.load_state_dict(torch.load(os.path.join(args.fname, f'opt_{start_epoch - 1}.pth')))
        print('Resuming at epoch {start_epoch}')
        if os.path.exists(os.path.join(args.fname, f'model_best.pth')):
            best_test_robust_acc = torch.load(os.path.join(args.fname, f'model_best.pth'))['test_robust_acc']
    else:
        start_epoch = 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start_time = time.time()
        train_nat_loss = 0
        train_nat_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0

        train_nat_vnodes = 0
        train_adv_vnodes = 0

        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            if epoch >= args.vs_warmup:
                batch_y_soft = constuct_vs_label(y, args.alpha, NUM_REAL_CLASSES, args.v_classes, args.add_noise).cuda()
            else:
                batch_y_soft = F.one_hot(y, num_classes=NUM_REAL_CLASSES + args.v_classes).cuda()

            # lr = lr_schedule(epoch + (i + 1) / len(train_loader))
            lr = adjust_learning_rate(opt, epoch)
            opt.param_groups[0].update(lr=lr)

            if args.attack == 'pgd':
                delta = attack_pgd(model, X, y, args.epsilon, args.pgd_alpha, args.attack_iters, args.restarts,
                                   args.norm, bn_type=args.bn_type, random_type=args.random_type)
                delta = delta.detach()
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            X_adv = normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit))

            model.train()
            # calculate adversarial weight perturbation and perturb it
            if epoch >= args.awp_warmup:
                awp = awp_adversary.calc_awp(inputs_adv=X_adv, targets=batch_y_soft)
                awp_adversary.perturb(awp)

            robust_output = model(X_adv)
            robust_loss = cross_entropy_soft_target(robust_output, batch_y_soft)

            if args.l1:
                for name, param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1 * param.abs().sum()

            opt.zero_grad()
            robust_loss.backward()
            opt.step()

            if epoch >= args.awp_warmup:
                awp_adversary.restore(awp)

            train_robust_loss += robust_loss.item() * y.size(0)
            _, adv_pred = torch.max(robust_output[:, :NUM_REAL_CLASSES], dim=1)
            train_robust_acc += (adv_pred == y).sum().item()
            # train_adv_vnodes += (adv_pred >= NUM_REAL_CLASSES).sum().item()

            nat_output = model(normalize(X))
            nat_loss = cross_entropy_soft_target(nat_output, batch_y_soft)
            train_nat_loss += nat_loss.item() * y.size(0)
            _, nat_pred = torch.max(nat_output[:, :NUM_REAL_CLASSES], dim=1)
            train_nat_acc += (nat_pred == y).sum().item()
            # train_nat_vnodes += (nat_pred >= NUM_REAL_CLASSES).sum().item()

            train_n += y.size(0)

        train_time = time.time()
        print('================================================================')
        print('Epoch:{0}, LR:{1}'.format(epoch, lr))
        print('Train Time:{}, Train Nat Loss:{}, Train Nat Acc:{}, Train Robust Loss:{}, Train Robust Acc:{}, '
              .format(train_time - start_time, train_nat_loss / train_n, train_nat_acc / train_n,
                      train_robust_loss / train_n, train_robust_acc / train_n, ))

        st_time = time.time()
        test_n, test_robust_loss, test_robust_acc = eval_test_rob(model, test_loader, args,
                                                                  num_real_classes=NUM_REAL_CLASSES)
        test_n, test_loss, test_acc = eval_data(model, test_loader, num_real_classes=NUM_REAL_CLASSES)
        test_time = time.time()
        print('Test Time:{}, Test Loss:{}, Test Acc:{}, Test Robust Loss:{} Test Robust Acc:{}'.format(
            test_time - st_time, test_loss / test_n, test_acc / test_n, test_robust_loss / test_n,
            test_robust_acc / test_n))
        print('================================================================')

        if (epoch + 1) % args.chkpt_iters == 0 or epoch + 1 == args.epochs:
            torch.save(model.state_dict(), os.path.join(args.fname, f'model_{epoch}.pth'))
            torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))

        # save best
        if test_robust_acc / test_n > best_test_robust_acc:
            torch.save({
                'state_dict': model.state_dict(),
                'test_robust_acc': test_robust_acc / test_n,
                'test_robust_loss': test_robust_loss / test_n,
                'test_loss': test_loss / test_n,
                'test_acc': test_acc / test_n,
            }, os.path.join(args.fname, f'model_best.pth'))
            best_test_robust_acc = test_robust_acc / test_n


if __name__ == "__main__":
    main()
