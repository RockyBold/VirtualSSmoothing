from __future__ import print_function
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.optim as optim
from future.backports import OrderedDict
import numpy as np
from models import wideresnet, resnet, resnet_imagenet, resnet_tiny200, mobilenet_v2
from attacks import pgd, trades
from utils import nn_util, tiny_datasets, imagenet_loader

parser = argparse.ArgumentParser(description='Adversarial Training with Virtual Smoothing Lables')
parser = argparse.ArgumentParser(description='Training Neural Networks with Virtual Smoothing Classes')

parser.add_argument('--model_name', default='resnet-18',
                    help='Model name: resnet-18, mobilenet_v2, wrn-28-10, wrn-34-10, wrn-40-4, resnet-50')

parser.add_argument('--dataset', default='cifar10',
                    help='Dataset options: svhn, cifar10, cifar100, tiny-imagenet-32x32, tiny-imagenet-64x64')

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')

parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W',
                    help='Weight decay value (default: 5e-4)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epochs', type=int, default=160, metavar='N', help='Number of epochs to train')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 155],
                    help='Epochs to decrease learning rate.')

parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate multiplier on schedule.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA training')
parser.add_argument('--attack_eps', default=0.031, type=float, help='Perturbation radius')
parser.add_argument('--attack_steps', default=10, type=int, help='Number of steps for perturbation')
parser.add_argument('--attack_lr', default=0.00784, type=float, help='Step size for perturbation')
parser.add_argument('--beta', default=6.0, type=float, help='Regularization parameter, i.e., 1/lambda in TRADES')
parser.add_argument('--nat_ce_w', default=1.0, type=float,
                    help='Regularization for natural cross-entropy loss in trades_plus_loss')
parser.add_argument('--adv_ce_w', default=0.0, type=float,
                    help='Regularization for adversarial cross-entropy loss in trades_plus_loss')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='How many batches to wait before logging training status')

parser.add_argument('--model_dir', default='dnn_models/cifar/', help='Directory to save model checkpoints')
parser.add_argument('--gpuid', type=int, default=0, help='ID of the GPU to use.')

parser.add_argument('--training_method', default='pgd', help='Training method: clean, pgd, or trades')
parser.add_argument('--bn_type', default='eval', help='Batch normalization type during attack: train or eval')
parser.add_argument('--random_type', default='gaussian', help='Random type for PGD: uniform or gaussian')
parser.add_argument('--attack_test_steps', default=20, type=int, help='Number of perturbation steps in test phase')
parser.add_argument('--resume_epoch', type=int, default=0, metavar='N', help='Epoch for resuming training')
parser.add_argument('--always_save_cpt', action='store_true', default=False,
                    help='Whether to save each epoch checkpoint')
parser.add_argument('--norm', default='Linf', type=str, choices=['Linf'],
                    help='Norm type: Linf')

parser.add_argument('--alpha', default=0.7, type=float, help='Total confidence of virtual smoothing classes')
parser.add_argument('--v_classes', default=10, type=int,
                    help='Number of virtual smoothing classes')
parser.add_argument('--add_noise', action='store_true', default=False, help='Add noise to virtual classes')
parser.add_argument('--v_type', default='u', type=str, help='Type of noise added to virtual classes: u or n')
parser.add_argument('--vs_warmup', default=0, type=int, help='Warm-up epoch for using VS labels.')
parser.add_argument('--share_bn', action='store_true', default=False,
                    help='Whether to use the same Batch Normalization for adversarial and clean examples')

parser.add_argument('--teacher_model_name', default='wrn-34-10',
                    help='Teacher model name: wrn-28-10, wrn-34-10, wrn-40-4, resnet-18, or resnet-50')
parser.add_argument('--teacher_cpt_file', default='', help='File path of the teacher model')
parser.add_argument('--distill_type', default='real',
                    help='Knowledge transfer method from the teacher: real or whole')
parser.add_argument('--teacher_v_classes', default=10, type=int,
                    help='Number of virtual classes in the teacher model')
parser.add_argument('--distill_alpha', default=1.0, type=float,
                    help='Parameter to balance the trade-off between the student model and the teacher model')
parser.add_argument('--temp', default=1.0, type=float, help='Temperature for distillation')

args = parser.parse_args()

if args.attack_lr >= 1:
    args.attack_lr = args.attack_lr / 255
if args.attack_eps >= 1:
    args.attack_eps = args.attack_eps / 255
if args.distill_alpha > 1:
    args.distill_alpha = args.distill_alpha / 6

if args.dataset == 'cifar10':
    NUM_REAL_CLASSES = 10
    NUM_EXAMPLES = 50000
elif args.dataset == 'svhn':
    NUM_REAL_CLASSES = 10
    NUM_EXAMPLES = 73257
elif args.dataset == 'cifar100':
    NUM_REAL_CLASSES = 100
    NUM_EXAMPLES = 50000
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

ADV_TRAINING_METHODS = ['pgd', 'trades', 'trades-plus', 'ard', 'ard-plus']

# settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
storage_device = torch.device("cpu")
normalizer = None

def save_cpt(model, optimizer, epoch):
    path = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, epoch))
    torch.save(model.state_dict(), path)
    path = os.path.join(args.model_dir, '{0}_cpt_epoch{1}.pt'.format(args.training_method, epoch))
    torch.save(optimizer.state_dict(), path)

def resume(model, optimizer, epoch):
    path = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, epoch))
    model.load_state_dict(torch.load(path))
    path = os.path.join(args.model_dir, '{0}_cpt_epoch{1}.pt'.format(args.training_method, epoch))
    optimizer.load_state_dict(torch.load(path))
    return model, optimizer

def get_all_test_data(test_loader):
    x_test = None
    y_test = None
    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        if x_test is None:
            x_test = batch_x
        else:
            x_test = torch.cat((x_test, batch_x), 0)

        if y_test is None:
            y_test = batch_y
        else:
            y_test = torch.cat((y_test, batch_y), 0)

    return x_test, y_test

def atuo_attack_real_logits(model, test_loader, batch_size, norm, attack_eps, num_real_classes):
    x_test, y_test = get_all_test_data(test_loader)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    model.eval()
    # load attack
    from autoattack import AutoAttack
    version = 'attack-real-logits'
    attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    adversary = AutoAttack(model, norm=norm, eps=attack_eps, version=version, attacks_to_run=attacks_to_run,
                           num_in_classes=num_real_classes, attack_vclasses=False)
    adversary.apgd_targeted.n_target_classes = num_real_classes - 1
    adversary.fab.n_target_classes = num_real_classes - 1
    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)

    return adv_complete, y_test


def atuo_attack_whole_logits(model, test_loader, batch_size, norm, attack_eps, num_real_classes, num_v_classes):
    x_test, y_test = get_all_test_data(test_loader)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    model.eval()
    # load attack
    from autoattack import AutoAttack
    version = 'attack-whole-logits'
    attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    adversary = AutoAttack(model, norm=norm, eps=attack_eps, version=version, attacks_to_run=attacks_to_run,
                           num_in_classes=num_real_classes, attack_vclasses=True)
    adversary.apgd_targeted.n_target_classes = num_real_classes - 1 + num_v_classes
    adversary.fab.n_target_classes = num_real_classes - 1 + num_v_classes

    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)

    return adv_complete, y_test


def aa_eval(model, test_loader, args):
    batch_size = args.batch_size
    norm = args.norm
    attack_eps = args.attack_eps
    v_classes = args.v_classes
    sub_alpha = args.sub_alpha
    nat_x, _ = get_all_test_data(test_loader)
    nat_x = nat_x.to(device)
    print('================================================================================================')
    model.output_real_only = False
    model.output_v_only = False

    test_cw_acc, test_cw_max_in_v, test_cw_corr_max_in_v, corr_conf, corr_v_conf = \
        pgd.eval_pgdadv(model, test_loader, args.attack_test_steps, args.attack_lr, attack_eps, NUM_REAL_CLASSES,
                        loss_str='pgd-cw', norm=norm)
    msc = corr_conf.mean()
    msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
    if corr_v_conf.size(1) > 1:
        msc_v = corr_v_conf.max(dim=1)[0].mean()
        std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
    print('Attacking the entire logits with CW, robust acc: {}, test_cw_max_in_v: {}, test_cw_corr_max_in_v: {}, msc: {}, '
          'msc_v:{}, std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(test_cw_acc, test_cw_max_in_v,
                                                                       test_cw_corr_max_in_v, msc, msc_v,
                                                                       std_corr_v_conf, mean_corr_v_conf))

    print('----------------------------------------------------------------------------------------------------')
    adv_x, y = atuo_attack_whole_logits(model, test_loader, batch_size, norm, attack_eps, NUM_REAL_CLASSES, v_classes)
    rob_acc, max_in_v, max_in_v_corr, corr_conf, corr_v_conf = nn_util.eval_from_data(model, adv_x, y, batch_size, NUM_REAL_CLASSES, align_x=nat_x)
    msc = corr_conf.mean()
    msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
    if corr_v_conf.size(1) > 1:
        msc_v = corr_v_conf.max(dim=1)[0].mean()
        std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
    print('Attacking the entire logits with Auto-Attack+, robust acc:{}, max_in_v:{}, max_in_v_corr:{}, msc:{}, msc_v:{} '
          'std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(rob_acc, max_in_v, max_in_v_corr, msc, msc_v,
                                                             std_corr_v_conf, mean_corr_v_conf))
    if v_classes > 0 and sub_alpha > 0:
        rob_in_acc, rob_v_acc, rob_in_v_acc = nn_util.eval_in_v_from_data(model, adv_x, y, batch_size, NUM_REAL_CLASSES)
        print('Using eval_in_v to statistical results, rob_in_acc:{}, rob_v_acc:{}, rob_in_v_acc:{}'
              .format(rob_in_acc, rob_v_acc, rob_in_v_acc))

    if v_classes > 0 and sub_alpha > 0:
        print('----------------------------------------------------------------------------------------------------')
        model.output_v_only = True
        adv_x, y = atuo_attack_whole_logits(model, test_loader, batch_size, norm, attack_eps, NUM_REAL_CLASSES, 0)
        rob_in_acc, rob_v_acc, rob_in_v_acc = nn_util.eval_in_v_from_data(model, adv_x, y, batch_size, NUM_REAL_CLASSES)
        model.output_v_only = False
        print('Using v_classes to defend against Auto-Attack+ (model.output_v_only = True), '
              'rob_in_acc:{}, rob_v_acc:{}, rob_in_v_acc:{}' .format(rob_in_acc, rob_v_acc, rob_in_v_acc))
    return


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    epoch_lr = args.lr
    for i in range(0, len(args.schedule)):
        if epoch > args.schedule[i]:
            epoch_lr = args.lr * np.power(args.gamma, (i + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch_lr
    return epoch_lr


def add_noise_to_uniform(unif):
    assert len(unif.size()) == 2
    unif_elem = unif.float().mean()
    new_unif = unif.clone()
    new_unif.uniform_(unif_elem - 0.01 * unif_elem, unif_elem + 0.01 * unif_elem)
    factor = new_unif.sum(dim=1) / unif.sum(dim=1)
    new_unif = new_unif / factor.unsqueeze(dim=1)
    # sum=new_unif.sum(dim=1)
    return new_unif


def label_smoothing(hard_label, alpha, num_classes):
    y_ls = F.one_hot(hard_label, num_classes=num_classes) * (1 - alpha) + (alpha / (num_classes))
    y_ls = y_ls.to(hard_label.device)
    return y_ls


def constuct_vs_label(hard_label, alpha, num_classes, v_classes, add_noise=False, v_type='u'):
    # one-hot
    if alpha == 0:
        return F.one_hot(hard_label, num_classes=num_classes + v_classes).to(hard_label.device)
    # standard label smoothing
    elif alpha != 0 and v_classes == 0:
        return  label_smoothing(hard_label, alpha, num_classes)
    # with additional virtuall smoothing classes
    elif alpha != 0 and v_classes != 0:
        y_vc = torch.zeros((len(hard_label), num_classes + v_classes), device=hard_label.device)
        u = [i for i in range(len(hard_label))]
        y_vc[u, hard_label] += (1 - alpha)
        if v_type == 'u':
            temp_v_conf = alpha / v_classes
            y_vc[:, num_classes:num_classes + v_classes] = temp_v_conf
            if add_noise:
                y_vc[:, num_classes:num_classes + v_classes] = add_noise_to_uniform(
                    y_vc[:, num_classes:num_classes + v_classes])
        elif v_type == 'n':
            v_conf = torch.randint(low=0, high=100 * v_classes, size=(len(hard_label), v_classes)).float().to(
                hard_label.device)
            v_conf = v_conf / v_conf.sum(dim=1).unsqueeze(dim=1)
            y_vc[:, num_classes:num_classes + v_classes] = alpha * v_conf
        return y_vc
    else:
        raise ValueError('alpha:{0}, v_classes:{1}'.format(alpha, v_classes))


def train(model, train_loader, optimizer, test_loader):
    
    start_epoch = 1
    if args.resume_epoch > 0:
        print('try to resume from epoch', args.resume_epoch)
        model, optimizer = resume(model, optimizer, args.resume_epoch)
        start_epoch = args.resume_epoch + 1

    if args.teacher_cpt_file != '':
        teach_model = get_model(args.teacher_model_name, num_real_classes=NUM_REAL_CLASSES,
                                num_v_classes=args.teacher_v_classes, normalizer=normalizer, dataset=args.dataset).to(device)
        teach_model.load_state_dict(torch.load(args.teacher_cpt_file))
        teach_model.eval()
        print('successfully loaded teach_model from {}'.format(args.teacher_cpt_file))
        KL_loss = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        epoch_lr = adjust_learning_rate(optimizer, epoch)

        total = 0
        train_nat_correct = 0
        train_adv_correct = 0
        train_nat_max_in_v = 0
        train_pgd_max_in_v = 0
        train_nat_corr_max_in_v = 0
        train_pgd_corr_max_in_v = 0
        for i, data in enumerate(train_loader):
            nat_batch_x, batch_y = data
            if use_cuda:
                nat_batch_x = nat_batch_x.cuda(non_blocking=True)
                batch_y = batch_y.cuda(non_blocking=True)
            if args.teacher_cpt_file != '':
                with torch.no_grad():
                    nat_teach_logits = teach_model(nat_batch_x)
            if epoch >= args.vs_warmup:
                batch_y_soft = constuct_vs_label(batch_y, args.alpha, NUM_REAL_CLASSES, args.v_classes, args.add_noise,
                                                 v_type=args.v_type, sub_alpha=args.sub_alpha)
            else:
                batch_y_soft = F.one_hot(batch_y, num_classes=NUM_REAL_CLASSES + args.v_classes).to(batch_y.device)
            if use_cuda:
                batch_y_soft = batch_y_soft.cuda(non_blocking=True)
                
            if args.training_method == 'pgd':
                adv_batch_x = pgd.pgd_attack(model, nat_batch_x, batch_y, attack_steps=args.attack_steps,
                                                 attack_lr=args.attack_lr, attack_eps=args.attack_eps,
                                                 random_type=args.random_type, bn_type=args.bn_type)
                model.train()
                adv_logits = model(adv_batch_x)
                loss = nn_util.cross_entropy_soft_target(adv_logits, batch_y_soft)
            elif args.training_method == 'trades':
                adv_batch_x = trades.trades_pgd_attack(model, nat_batch_x, args.attack_steps, args.attack_lr, args.attack_eps)
                model.train()
                if not args.share_bn:
                    nat_logits = model(nat_batch_x)
                    adv_logits = model(adv_batch_x)
                else:
                    cat_x = torch.cat((nat_batch_x, adv_batch_x), dim=0)
                    cat_logits = model(cat_x)
                    nat_logits = cat_logits[:len(nat_batch_x)]
                    adv_logits = cat_logits[len(nat_batch_x):]
                loss, loss_ce, loss_kl = trades.trades_loss(nat_logits, adv_logits, batch_y_soft, args.beta)
            elif args.training_method == 'trades-plus':
                adv_batch_x = trades.trades_pgd_attack(model, nat_batch_x, args.attack_steps, args.attack_lr,
                                                       args.attack_eps)
                model.train()
                if not args.share_bn:
                    nat_logits = model(nat_batch_x)
                    adv_logits = model(adv_batch_x)
                else:
                    cat_x = torch.cat((nat_batch_x, adv_batch_x), dim=0)
                    cat_logits = model(cat_x)
                    nat_logits = cat_logits[:len(nat_batch_x)]
                    adv_logits = cat_logits[len(nat_batch_x):]
                loss = trades.trades_plus_loss(nat_logits, adv_logits, batch_y_soft, args.beta, args.nat_ce_w,
                                                 args.adv_ce_w, args.temp)
            elif args.training_method == 'ard':
                adv_batch_x = trades.trades_pgd_attack(model, nat_batch_x, args.attack_steps, args.attack_lr, args.attack_eps)
                model.train()
                batch_y_onehot = F.one_hot(batch_y, num_classes=NUM_REAL_CLASSES + args.v_classes).to(batch_y.device)
                if not args.share_bn:
                    nat_logits = model(nat_batch_x)
                    adv_logits = model(adv_batch_x)
                else:
                    cat_x = torch.cat((nat_batch_x, adv_batch_x), dim=0)
                    cat_logits = model(cat_x)
                    nat_logits = cat_logits[:len(nat_batch_x)]
                    adv_logits = cat_logits[len(nat_batch_x):]
                if args.distill_type == 'real':
                    # assert args.vs_warmup >= args.epochs or args.alpha == 0
                    kl_loss = KL_loss(F.log_softmax(adv_logits[:, :NUM_REAL_CLASSES] / args.temp, dim=1),
                                      F.softmax(nat_teach_logits[:, :NUM_REAL_CLASSES] / args.temp, dim=1))
                    stu_nat_ce_loss = nn_util.cross_entropy_soft_target(nat_logits[:, :NUM_REAL_CLASSES],
                                                                    batch_y_onehot[:, :NUM_REAL_CLASSES])
                else:
                    kl_loss = KL_loss(F.log_softmax(adv_logits / args.temp, dim=1),
                                      F.softmax(nat_teach_logits / args.temp, dim=1))
                    stu_nat_ce_loss = nn_util.cross_entropy_soft_target(nat_logits, batch_y_onehot)
                loss = (1.0 - args.distill_alpha) * stu_nat_ce_loss \
                       + args.distill_alpha * args.temp * args.temp * kl_loss
            elif args.training_method == 'ard-plus':
                adv_batch_x = trades.trades_pgd_attack(model, nat_batch_x, args.attack_steps, args.attack_lr, args.attack_eps)
                model.train()
                if not args.share_bn:
                    nat_logits = model(nat_batch_x)
                    adv_logits = model(adv_batch_x)
                else:
                    cat_x = torch.cat((nat_batch_x, adv_batch_x), dim=0)
                    cat_logits = model(cat_x)
                    nat_logits = cat_logits[:len(nat_batch_x)]
                    adv_logits = cat_logits[len(nat_batch_x):]
                if args.distill_type == 'real':
                    assert args.vs_warmup >= args.epochs or args.alpha == 0
                    kl_loss = KL_loss(F.log_softmax(adv_logits[:, :NUM_REAL_CLASSES] / args.temp, dim=1),
                                      F.softmax(nat_teach_logits[:, :NUM_REAL_CLASSES] / args.temp, dim=1))
                    stu_nat_ce_loss = nn_util.cross_entropy_soft_target(nat_logits[:, :NUM_REAL_CLASSES],
                                                                        batch_y_soft[:, :NUM_REAL_CLASSES])
                    stu_adv_ce_loss = nn_util.cross_entropy_soft_target(adv_logits[:, :NUM_REAL_CLASSES],
                                                                        batch_y_soft[:, :NUM_REAL_CLASSES])
                else:
                    kl_loss = KL_loss(F.log_softmax(adv_logits / args.temp, dim=1),
                                      F.softmax(nat_teach_logits / args.temp, dim=1))
                    stu_nat_ce_loss = nn_util.cross_entropy_soft_target(nat_logits, batch_y_soft)
                    stu_adv_ce_loss = nn_util.cross_entropy_soft_target(adv_logits, batch_y_soft)
                loss = (1.0 - args.distill_alpha) * (stu_nat_ce_loss + stu_adv_ce_loss) \
                       + args.distill_alpha * args.temp * args.temp * kl_loss
            elif args.training_method == 'clean':
                model.train()
                nat_logits = model(nat_batch_x)
                loss = nn_util.cross_entropy_soft_target(nat_logits, batch_y_soft)
            else:
                raise ValueError('unsupported training method: {0}'.format(args.training_method))

            # compute output
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record acc
            model.eval()
            with torch.no_grad():
                nat_logits = model(nat_batch_x)
                _, nat_pred = torch.max(nat_logits[:, :NUM_REAL_CLASSES], dim=1)
                nat_correct_indices = nat_pred == batch_y
                train_nat_correct += nat_correct_indices.sum().item()
                _, nat_whole_preds = torch.max(nat_logits, dim=1)
                max_in_v_indices = nat_whole_preds >= NUM_REAL_CLASSES
                train_nat_max_in_v += max_in_v_indices.sum().item()
                train_nat_corr_max_in_v += (torch.logical_and(nat_correct_indices, max_in_v_indices)).sum().item()

                if args.training_method in ADV_TRAINING_METHODS:
                    adv_logits = model(adv_batch_x)
                    _, adv_pred = torch.max(adv_logits[:, :NUM_REAL_CLASSES], dim=1)
                    adv_correct_indices = adv_pred == batch_y
                    train_adv_correct += adv_correct_indices.sum().item()
                    _, adv_whole_preds = torch.max(adv_logits, dim=1)
                    adv_max_in_v_indices = adv_whole_preds >= NUM_REAL_CLASSES
                    train_pgd_max_in_v += adv_max_in_v_indices.sum().item()
                    train_pgd_corr_max_in_v += (torch.logical_and(adv_correct_indices, adv_max_in_v_indices)).sum().item()

            batch_size = len(batch_y)
            total += batch_size
            if i % args.log_interval == 0 or (i + 1) * nat_batch_x.size()[0] >= NUM_EXAMPLES:
                processed_ratio = 100. * ((i + 1) * batch_size) / NUM_EXAMPLES
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(epoch, (i + 1) * batch_size, NUM_EXAMPLES,
                                                                              processed_ratio, loss.item()))

        end_time = time.time()
        batch_time = end_time - start_time

        train_nat_acc = (float(train_nat_correct) / total) * 100
        train_pgd_acc = (float(train_adv_correct) / total) * 100
        message = 'Epoch {}, Time {}, LR: {}, Loss: {}, ' \
                  'Training nat acc: {}, train_nat_max_in_v: {}, train_nat_corr_max_in_v: {},' \
                  'Training adv acc: {}, train_pgd_max_in_v: {}, train_pgd_corr_max_in_v: {},' \
            .format(epoch, batch_time, epoch_lr, loss.item(), train_nat_acc, train_nat_max_in_v,
                    train_nat_corr_max_in_v, train_pgd_acc, train_pgd_max_in_v, train_pgd_corr_max_in_v)
        print(message)

        # Evaluation
        test_nat_acc, test_nat_max_in_v, test_nat_corr_max_in_v, corr_conf, corr_v_conf \
            = nn_util.eval(model, test_loader, NUM_REAL_CLASSES)
        msc = corr_conf.mean()
        msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
        if corr_v_conf.size(1) > 1:
            msc_v = corr_v_conf.max(dim=1)[0].mean()
            std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
        print('test nat acc: {}, test_nat_max_in_v: {}, test_nat_corr_max_in_v: {}, msc: {}, msc_v:{}, '
              'std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(test_nat_acc, test_nat_max_in_v,
                                                                 test_nat_corr_max_in_v, msc, msc_v, std_corr_v_conf,
                                                                 mean_corr_v_conf))

        # model.output_real_only = False
        test_pgd_acc, test_pgd_max_in_v, test_pgd_corr_max_in_v, corr_conf, corr_v_conf = \
            pgd.eval_pgdadv(model, test_loader, args.attack_test_steps, args.attack_lr, args.attack_eps,
                            NUM_REAL_CLASSES, norm=args.norm)
        msc = corr_conf.mean()
        msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
        if corr_v_conf.size(1) > 1:
            msc_v = corr_v_conf.max(dim=1)[0].mean()
            std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
        print('Under PGD-CE ATTACK, test pgd acc: {}, test_pgd_max_in_v: {}, test_pgd_corr_max_in_v: {}, msc: {}, '
              'msc_v: {}, std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(test_pgd_acc, test_pgd_max_in_v,
                                                                 test_pgd_corr_max_in_v, msc, msc_v, std_corr_v_conf,
                                                                 mean_corr_v_conf))
        test_cw_acc, test_cw_max_in_v, test_cw_corr_max_in_v = -1., 0, 0
        if epoch > args.schedule[0] - 10:
            test_cw_acc, test_cw_max_in_v, test_cw_corr_max_in_v, corr_conf, corr_v_conf = \
                pgd.eval_pgdadv(model, test_loader, args.attack_test_steps, args.attack_lr, args.attack_eps,
                                NUM_REAL_CLASSES, loss_str='pgd-acw', norm=args.norm, attack_v=True)
            msc = corr_conf.mean()
            msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
            if corr_v_conf.size(1) > 1:
                msc_v = corr_v_conf.max(dim=1)[0].mean()
                std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
            print('Under PGD-ACW ATTACK, test cw acc: {}, test_cw_max_in_v: {}, test_cw_corr_max_in_v: {}, msc: {}, '
                  'msc_v: {}, std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(test_cw_acc, test_cw_max_in_v,
                                                                     test_cw_corr_max_in_v, msc, msc_v, std_corr_v_conf,
                                                                     mean_corr_v_conf))
        print('================================================================')
        save_cpt(model, optimizer, epoch)

    return loss.item(), (test_nat_acc, test_nat_max_in_v), (test_pgd_acc, test_pgd_max_in_v)


def get_model(model_name, num_real_classes, num_v_classes, normalizer=None, dataset='cifar10'):
    size_3x32x32 = ['svhn', 'cifar10', 'cifar100', 'tiny-imagenet-32x32']
    size_3x64x64 = ['tiny-imagenet-64x64']
    size_3x224x224 = ['imagenet']
    if dataset in size_3x32x32:
        if model_name == 'wrn-34-10':
            return wideresnet.WideResNet(depth=34, widen_factor=10, num_real_classes=num_real_classes,
                                         num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'wrn-28-10':
            return wideresnet.WideResNet(depth=28, widen_factor=10, num_real_classes=num_real_classes,
                                         num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'wrn-40-4':
            return wideresnet.WideResNet(depth=40, widen_factor=4, num_real_classes=num_real_classes,
                                         num_v_classes=num_v_classes, normalizer=normalizer)
        elif model_name == 'resnet-18':
            return resnet.ResNet18(num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                                   normalizer=normalizer)
        elif model_name == 'resnet-50':
            return resnet.ResNet50(num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                                   normalizer=normalizer)
        elif model_name == 'mobilenet_v2':
            return mobilenet_v2.mobilenet_v2(num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                                             normalizer=normalizer)
        else:
            raise ValueError('un-supported model: {0}', model_name)
    elif dataset in size_3x64x64:
        if model_name == 'resnet-18':
            return resnet_tiny200.resnet18(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-50':
            return resnet_tiny200.resnet50(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        else:
            raise ValueError('un-supported model: {0}', model_name)
    elif dataset in size_3x224x224:
        if model_name == 'resnet-18':
            return resnet_imagenet.resnet18(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-50':
            return resnet_imagenet.resnet50(num_real_classes=num_real_classes, num_v_classes=num_v_classes)
        else:
            raise ValueError('un-supported model: {0}', model_name)
    else:
        raise ValueError('un-supported dataset: {0}', dataset)


def main():
    # setup data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'cifar10':
        transform_train = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            # T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        transform_test = T.Compose([T.ToTensor()])

        dataloader = torchvision.datasets.CIFAR10
        train_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar10/', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar10/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        transform_train = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            # T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        transform_test = T.Compose([T.ToTensor()])

        dataloader = torchvision.datasets.CIFAR100
        train_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar100/', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar100/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'svhn':
        # normalizer = None
        transform_train = T.Compose([
            # T.RandomCrop([54, 54]),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        svhn_train = torchvision.datasets.SVHN(root='../../datasets/svhn/', download=True, transform=transform_train,
                                               split='train')
        svhn_test = torchvision.datasets.SVHN(root='../../datasets/svhn/', download=True, transform=transform_test,
                                              split='test')
        train_loader = torch.utils.data.DataLoader(dataset=svhn_train, batch_size=args.batch_size, shuffle=True,
                                                   **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
    elif args.dataset == 'tiny-imagenet-32x32':
        data_dir = '../../datasets/tiny-imagenet-200/'
        normalize = T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = T.Compose([
            T.RandomResizedCrop(32),
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            # T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        # transform_train = T.Compose([T.RandomResizedCrop(32), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        transform_test = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            normalize
        ])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'tiny-imagenet-64x64':
        data_dir = '../../datasets/tiny-imagenet-200/'
        normalize = T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        transform_train = T.Compose([
            T.Pad(8, padding_mode='reflect'),
            T.RandomCrop(64),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        # transform_train = T.Compose([T.RandomResizedCrop(32), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        transform_test = T.Compose([
            T.ToTensor(),
            normalize
        ])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'imagenet':
        imagenet_root = '../../datasets/'
        train_loader, test_loader = imagenet_loader.data_loader(imagenet_root, batch_size=args.batch_size)
    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))

    cudnn.benchmark = True

    # init model, Net() can be also used here for training
    model = get_model(args.model_name, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes,
                      normalizer=normalizer, dataset=args.dataset).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print('========================================================================================================')
    print('args:', args)
    print('========================================================================================================')

    train(model, train_loader, optimizer, test_loader)


if __name__ == '__main__':
    main()
