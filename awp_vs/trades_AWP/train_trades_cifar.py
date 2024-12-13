from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import Bar, Logger, AverageMeter, accuracy, misc
from utils_awp import TradesAWP
from models import resnet, wideresnet

parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--model', type=str, default='wrn-34-10', help='model name: resnet-18 or wrn-34-10')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'])
parser.add_argument('--data-path', type=str, default='../data',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='Linf', type=str, choices=['Linf', 'L2'],
                    help='The threat model')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.00784, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')

parser.add_argument('--awp-gamma', default=0.005, type=float,
                    help='whether or not to add parametric noise, 0.005 for wrn-34-10, and 0.01 for resnet-18')
parser.add_argument('--awp-warmup', default=10, type=int,
                    help='We could apply AWP after some epochs for accelerating.')

parser.add_argument('--alpha', default=0.7, type=float, help='total confidences of virtual nodes')
parser.add_argument('--v_classes', default=10, type=int,
                    help='the number of additional virtual nodes in output layer')
parser.add_argument('--add_noise', action='store_true', default=False, help='add noise to virtual classes or not')
parser.add_argument('--training_method', default='trades', help='training method: trades or trades-plus')
parser.add_argument('--vs_warmup', default=20, type=int,
                    help='apply ATVC soft-labels after some epochs.')

args = parser.parse_args()
if args.epsilon >= 1:
    args.epsilon = args.epsilon / 255
    epsilon = args.epsilon
if args.step_size >= 1:
    args.step_size = args.step_size / 255
    step_size = args.step_size

if args.awp_gamma <= 0.0:
    args.awp_warmup = np.infty
if args.data == 'CIFAR100':
    NUM_REAL_CLASSES = 100
else:
    NUM_REAL_CLASSES = 10

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomCrop(32),
    # transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = getattr(datasets, args.data)(
    root=args.data_path, train=True, download=True, transform=transform_train)
testset = getattr(datasets, args.data)(
    root=args.data_path, train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def perturb_input(model, x_natural, step_size=0.003, epsilon=0.031, perturb_steps=10, distance='Linf'):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'Linf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                nat_logits = model(x_natural)
                adv_logits = model(x_adv)
                loss_kl = F.kl_div(F.log_softmax(adv_logits, dim=1), F.softmax(nat_logits, dim=1), reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'L2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            # if (grad_norms == 0).any():
            #     delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


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


def train(model, train_loader, optimizer, epoch, awp_adversary):
    batch_time = AverageMeter()
    losses = AverageMeter()
    adv_top1 = AverageMeter()
    nat_top1 = AverageMeter()
    end = time.time()

    # train_nat_vnodes = 0
    # train_adv_vnodes = 0

    print('epoch: {}'.format(epoch))
    bar = Bar('Processing', max=len(train_loader))

    for batch_idx, (data, target) in enumerate(train_loader):
        x_natural, target = data.to(device), target.to(device)
        if epoch >= args.vs_warmup:
            batch_y_soft = constuct_vs_label(target, args.alpha, NUM_REAL_CLASSES, args.v_classes,
                                                            args.add_noise).to(device)
        else:
            batch_y_soft = F.one_hot(target, num_classes=NUM_REAL_CLASSES + args.v_classes).to(device)

        # craft adversarial examples
        x_adv = perturb_input(model=model, x_natural=x_natural, step_size=args.step_size, epsilon=args.epsilon,
                              perturb_steps=args.num_steps, distance=args.norm)

        model.train()
        # calculate adversarial weight perturbation
        if epoch >= args.awp_warmup:
            awp = awp_adversary.calc_awp(inputs_adv=x_adv, inputs_clean=x_natural, targets=batch_y_soft, beta=args.beta,
                                         loss_str=args.training_method)
            awp_adversary.perturb(awp)

        optimizer.zero_grad()
        logits_adv = model(x_adv)
        loss_kl_robust = F.kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(model(x_natural), dim=1),
                               reduction='batchmean')
        # calculate natural loss and backprop
        logits = model(x_natural)
        # loss_natural = F.cross_entropy(logits, target)
        loss_ce_nat = cross_entropy_soft_target(logits, batch_y_soft)

        if args.training_method == 'trades':
            loss = loss_ce_nat + args.beta * loss_kl_robust
        elif args.training_method == 'trades-plus':
            loss_ce_rob = cross_entropy_soft_target(logits_adv, batch_y_soft)
            loss = loss_ce_nat + loss_ce_rob + args.beta * loss_kl_robust
        else:
            raise ValueError('un-supported training loss: {}'.format(args.training_method))

        adv_prec1, _ = accuracy(logits_adv[:, :NUM_REAL_CLASSES], target, topk=(1, 5))
        adv_top1.update(adv_prec1.item(), x_natural.size(0))
        nat_prec1, _ = accuracy(logits[:, :NUM_REAL_CLASSES], target, topk=(1, 5))
        nat_top1.update(nat_prec1.item(), x_natural.size(0))
        losses.update(loss.item(), x_natural.size(0))

        # update the parameters at last
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}|'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=adv_top1.avg,
        )
        bar.next()
    bar.finish()
    return losses.avg, nat_top1.avg, adv_top1.avg


def test(model, test_loader, criterion):
    global best_acc
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total: {total:}| ETA: {eta:}| Loss:{loss:.4f}| top1: {top1:.2f}'.format(
                batch=batch_idx + 1,
                size=len(test_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg)
            bar.next()
    bar.finish()
    return losses.avg, top1.avg


# def adjust_learning_rate(optimizer, epoch):
#     """decrease the learning rate"""
#     lr = args.lr
#     if epoch >= 100:
#         lr = args.lr * 0.1
#     if epoch >= 150:
#         lr = args.lr * 0.01
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    epoch_lr = args.lr
    for i in range(0, len(args.schedule)):
        if epoch >= args.schedule[i]:
            epoch_lr = args.lr * np.power(args.gamma, (i + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch_lr
    return epoch_lr


def get_model(model_name, num_real_classes=10, num_v_classes=0):
    if model_name == 'resnet-18':
        model = resnet.ResNet18(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
    elif model_name == 'wrn-34-10':
        model = wideresnet.WideResNet34(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
    else:
        raise ValueError("Unknown model")
    return model


def main():
    print(args)
    # # init model, ResNet18() can be also used here for training
    # model = nn.DataParallel(
    #     getattr(models, args.model)(num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes)).to(device)
    model = get_model(args.model, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # We use a proxy model to calculate AWP, which does not affect the statistics of BN.
    # proxy = nn.DataParallel(
    #     getattr(models, args.model)(num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes)).to(device)
    proxy = get_model(args.model, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes).to(device)
    proxy_optim = optim.SGD(proxy.parameters(), lr=args.lr)
    awp_adversary = TradesAWP(model=model, proxy=proxy, proxy_optim=proxy_optim, gamma=args.awp_gamma)

    # criterion = nn.CrossEntropyLoss()

    # logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.model)
    # logger.set_names(['Learning Rate',
    #                   'Adv Train Loss', 'Nat Train Loss', 'Nat Val Loss',
    #                   'Adv Train Acc.', 'Nat Train Acc.', 'Nat Val Acc.'
    #                   'Adv Train vnodes', 'Nat Train vnodes',
    #                   'Adv Test vnodes', 'Nat Test vnodes',
    #                   ])

    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))

    for epoch in range(args.start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        lr = adjust_learning_rate(optimizer, epoch)
        st = time.time()
        print('================================================================')
        train_loss, train_nat_acc, train_adv_acc = train(model, train_loader, optimizer, epoch, awp_adversary)
        end_t = time.time()
        print('LR:{}, EPOCH:{}'.format(lr, epoch))
        print('Training INFO: time:{0}, adv_loss:{1}, train_nat_acc:{2}, train_adv_acc:{3}'
              .format(end_t - st, train_loss, train_nat_acc, train_adv_acc))

        test_nat_acc = misc.eval(model, test_loader, NUM_REAL_CLASSES)
        print('Eval on test set: nat acc:{0}'.format(test_nat_acc))
        test_pgd_acc = misc.eval_after_pgd(model, test_loader, 20, args.step_size, args.epsilon,
                                                            NUM_REAL_CLASSES)
        print('Eval on test set: pgd acc:{0}'.format(test_pgd_acc))
        if epoch > args.schedule[0] - 10:
            test_cw_acc = misc.eval_after_cw(model, test_loader, 20, args.step_size, args.epsilon,
                                                                NUM_REAL_CLASSES, args.v_classes)
            print('Eval on test set: cw acc:{0}'.format(test_cw_acc))
        print('================================================================')

        # logger.append(
        #     [lr, adv_loss, train_loss, val_loss, adv_acc, train_acc, val_acc, train_adv_vnodes, train_nat_vnodes,
        #      test_adv_vnodes, test_nat_vnodes])

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'ours-model-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'ours-opt-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
