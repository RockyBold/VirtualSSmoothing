from __future__ import print_function
import os
import torch
import argparse

import torchvision
import torchvision.transforms as T
from torchvision import datasets
from utils import nn_util
from models import wideresnet, resnet, resnet_imagenet, resnet_tiny200, mobilenet_v2
import torch.nn.functional as F

from attacks import pgd

parser = argparse.ArgumentParser(description='PyTorch Robustness Evaluation')
parser.add_argument('--model_name', default='resnet-18',
                    help='Model name: mobilenet_v2, wrn-28-10, wrn-34-10, wrn-40-4, '
                         'resnet-18, resnet-50, resnext-50, resnext-101, resnext-29_2x64d, resnext-29_32x4d, densenet-121')

parser.add_argument('--dataset', default='cifar10',
                    help='Dataset: svhn, cifar10, cifar100, tiny-imagenet-32x32, tiny-imagenet-64x64')

parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA training')
parser.add_argument('--gpuid', type=int, default=0, help='The ID of GPU.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 1)')
parser.add_argument('--model_file', default='./dnn_models/dataset/model.pt', help='file path of the model')
parser.add_argument('--v_classes', default=10, type=int,
                    help='The number of virtual smoothing classes')
parser.add_argument('--alpha', default=0., type=float, help='Total confidence of virtual smoothing classes')
parser.add_argument('--attack_eps', default=0.031, type=float,
                    help='max allowed perturbation (if attack_eps >= 1, attack_eps/255 will be used)')
parser.add_argument('--attack_lr', default=0.00784, type=float, help='perturb step size')
parser.add_argument('--attack_test_steps', default=20, type=int, help='perturb number of steps in test phase')
parser.add_argument('--norm', default='Linf', type=str, choices=['Linf'])


args = parser.parse_args()

if args.attack_lr >= 1:
    args.attack_lr = args.attack_lr / 255
if args.attack_eps >= 1:
    args.attack_eps = args.attack_eps / 255

if args.dataset == 'cifar10':
    NUM_REAL_CLASSES = 10
    NUM_EXAMPLES = 50000
elif args.dataset == 'svhn':
    NUM_REAL_CLASSES = 10
    NUM_EXAMPLES = 73257
elif args.dataset == 'cifar100':
    NUM_REAL_CLASSES = 100
    NUM_EXAMPLES = 50000
elif 'tiny-imagenet' in args.dataset:
    NUM_REAL_CLASSES = 200
    # NUM_EXAMPLES = 50000
elif args.dataset == 'imagenet':
    NUM_REAL_CLASSES = 1000
    # NUM_EXAMPLES = 50000
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# settings
# if not os.path.exists(args.model_dir):
#     os.makedirs(args.model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
storage_device = torch.device("cpu")
normalizer = None


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

def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


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


def standard_aa_eval(model, test_loader, args):
    batch_size = args.batch_size
    norm = args.norm
    attack_eps = args.attack_eps
    nat_x, nat_y = get_all_test_data(test_loader)
    nat_x = nat_x.to(device)
    nat_y = nat_y.to(device)

    print('================================================================================================')
    print('Run standard Auto-Attack...')
    model.output_real_only = False

    model.eval()
    # load attack
    from autoattack import AutoAttack
    version = 'standard-aa'
    attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    adversary = AutoAttack(model, norm=norm, eps=attack_eps, version=version, attacks_to_run=attacks_to_run, num_in_classes=NUM_REAL_CLASSES, attack_vclasses=True)
    adversary.apgd_targeted.n_target_classes = 9
    adversary.fab.n_target_classes = 9
    with torch.no_grad():
        adv_x = adversary.run_standard_evaluation(nat_x, nat_y, bs=batch_size)
    rob_acc, max_in_v, max_in_v_corr, corr_conf, corr_v_conf = nn_util.eval_from_data(model, adv_x, nat_y, batch_size, NUM_REAL_CLASSES, align_x=nat_x)
    msc = corr_conf.mean()
    msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
    if corr_v_conf.size(1) > 1:
        msc_v = corr_v_conf.max(dim=1)[0].mean()
        std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
    print('under standard Auto-Attack, robust acc:{}, max_in_v:{}, max_in_v_corr:{}, msc:{}, msc_v:{} '
          'std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(rob_acc, max_in_v, max_in_v_corr, msc, msc_v,
                                                             std_corr_v_conf, mean_corr_v_conf))
    return rob_acc


def aa_eval_real_part(model, test_loader, args):
    model.output_real_only = True
    batch_size = args.batch_size
    norm = args.norm
    attack_eps = args.attack_eps
    nat_x, _ = get_all_test_data(test_loader)
    nat_x = nat_x.to(device)
    print('================================================================================================')
    test_pgd_acc, _, _, corr_conf, _ = pgd.eval_pgdadv(model, test_loader, args.attack_test_steps, args.attack_lr,
                                                       attack_eps, NUM_REAL_CLASSES,
                                                       loss_str='pgd-ce', norm=norm, attack_v=False)
    msc = corr_conf.mean()
    print('attacking the real part with pgd-ce, robust acc: {}, mean of corr conf: {}'.format(test_pgd_acc, msc))

    print('----------------------------------------------------------------------------------------------------')
    test_cw_acc, _, _, corr_conf, _ = pgd.eval_pgdadv(model, test_loader, args.attack_test_steps, args.attack_lr,
                                                      attack_eps, NUM_REAL_CLASSES,
                                                      loss_str='pgd-cw', norm=norm)
    msc = corr_conf.mean()
    print('attacking the real part with pgd-cw, robust acc: {}, mean of corr conf: {}'.format(test_cw_acc, msc))

    print('----------------------------------------------------------------------------------------------------')
    print('Attacking the real part of the logits...')
    adv_x, y = atuo_attack_real_logits(model, test_loader, batch_size, norm, attack_eps, NUM_REAL_CLASSES)
    rob_acc, _, _, corr_conf, _ = nn_util.eval_from_data(model, adv_x, y, batch_size, NUM_REAL_CLASSES, align_x=nat_x)
    msc = corr_conf.mean()
    print('attacking the real part with auto-attack, robust acc:{}, msc: {}'.format(rob_acc, msc))
    model.output_real_only = False
    return


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
    print('Attacking the entire logits with CW, robust acc: {}, test_cw_max_in_v: {}, test_cw_corr_max_in_v: {}, '
          'msc: {},  msc_v:{}, std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(test_cw_acc, test_cw_max_in_v,
                                                                     test_cw_corr_max_in_v, msc, msc_v,
                                                                     std_corr_v_conf, mean_corr_v_conf))

    print('----------------------------------------------------------------------------------------------------')
    adv_x, y = atuo_attack_whole_logits(model, test_loader, batch_size, norm, attack_eps, NUM_REAL_CLASSES, v_classes)
    rob_acc, max_in_v, max_in_v_corr, corr_conf, corr_v_conf = nn_util.eval_from_data(model, adv_x, y, batch_size,
                                                                                      NUM_REAL_CLASSES, align_x=nat_x)
    msc = corr_conf.mean()
    msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
    if corr_v_conf.size(1) > 1:
        msc_v = corr_v_conf.max(dim=1)[0].mean()
        std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
    print('Attacking the entire logits with Auto-Attack+, robust acc:{}, max_in_v:{}, max_in_v_corr:{}, msc:{}, '
        'msc_v:{}, std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(rob_acc, max_in_v, max_in_v_corr, msc, msc_v,
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
              'rob_in_acc:{}, rob_v_acc:{}, rob_in_v_acc:{}'.format(rob_in_acc, rob_v_acc, rob_in_v_acc))
    return

def test_real_v(model, test_loader, args):
    from autoattack import AutoAttack
    batch_size = args.batch_size
    norm = args.norm
    attack_eps = args.attack_eps
    v_classes = args.v_classes

    x_test, y_test = get_all_test_data(test_loader)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    print('================================================================================================')
    test_nat_acc, test_nat_max_in_v, test_nat_corr_max_in_v, corr_conf, corr_v_conf = nn_util.eval(model, test_loader,
                                                                                                   NUM_REAL_CLASSES)
    msc = corr_conf.mean()
    msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
    if corr_v_conf.size(1) > 1:
        msc_v = corr_v_conf.max(dim=1)[0].mean()
        std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
    print('test nat acc: {}, test_nat_max_in_v: {}, test_nat_corr_max_in_v: {}, msc: {}, msc_v: {}, '
          'std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(test_nat_acc, test_nat_max_in_v,
                                                             test_nat_corr_max_in_v, msc, msc_v, std_corr_v_conf,
                                                             mean_corr_v_conf))
    print()
    model.eval()

    loss_strs=['pgd-ce', 'pgd-corr', 'pgd-ce-static-v', 'pgd-ce-dynamic-v', 'pgd-static-v', 'pgd-dynamic-v']
    for loss_str in loss_strs:
        print('-------------------------------------------------------------------------')
        rob_acc, max_in_v, max_in_v_corr, corr_conf, corr_v_conf = \
            pgd.eval_pgdadv(model, test_loader, args.attack_test_steps, args.attack_lr, args.attack_eps,
                            NUM_REAL_CLASSES, loss_str=loss_str, norm=args.norm)
        msc = corr_conf.mean()
        msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
        if corr_v_conf.size(1) > 1:
            msc_v = corr_v_conf.max(dim=1)[0].mean()
            std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
        print('Under attack with {}, rob acc: {}, max_in_v: {}, max_in_v_corr: {}, msc: {}, msc_v: {}, '
              'std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(loss_str, rob_acc, max_in_v, max_in_v_corr, msc, msc_v,
                                                                 std_corr_v_conf, mean_corr_v_conf))
        print()


    print('---------------------------------apgd-ce-t-real----------------------------------------')
    version = 'apgd-ce-t-real'
    attacks_to_run = ['apgd-ce-t']
    adversary = AutoAttack(model, norm=norm, eps=attack_eps, version=version, attacks_to_run=attacks_to_run,
                           num_in_classes=NUM_REAL_CLASSES, attack_vclasses=True)
    adversary.apgd_targeted.n_target_classes = NUM_REAL_CLASSES - 1
    adversary.apgd_targeted.target_in_real_only = True
    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
    rob_acc, max_in_v, max_in_v_corr, corr_conf, corr_v_conf \
        = nn_util.eval_from_data(model, adv_complete, y_test, batch_size, NUM_REAL_CLASSES, align_x=x_test)
    msc = corr_conf.mean()
    msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
    if corr_v_conf.size(1) > 1:
        msc_v = corr_v_conf.max(dim=1)[0].mean()
        std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
    print('With targets in real classes, robust acc: {}, max_in_v: {}, max_in_v_corr: {}, msc: {}, '
          'msc_v:{}, std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(rob_acc, max_in_v, max_in_v_corr, msc, msc_v,
                                                                       std_corr_v_conf, mean_corr_v_conf))
    print()

    print('---------------------------------apgd-dlr-t-real----------------------------------------')
    version = 'apgd-dlr-t-real'
    attacks_to_run = ['apgd-t']
    adversary = AutoAttack(model, norm=norm, eps=attack_eps, version=version, attacks_to_run=attacks_to_run,
                           num_in_classes=NUM_REAL_CLASSES, attack_vclasses=True)
    adversary.apgd_targeted.n_target_classes = NUM_REAL_CLASSES - 1
    adversary.apgd_targeted.target_in_real_only = True
    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
    rob_acc, max_in_v, max_in_v_corr, corr_conf, corr_v_conf \
        = nn_util.eval_from_data(model, adv_complete, y_test, batch_size, NUM_REAL_CLASSES, align_x=x_test)
    msc = corr_conf.mean()
    msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
    if corr_v_conf.size(1) > 1:
        msc_v = corr_v_conf.max(dim=1)[0].mean()
        std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
    print('With targets in real classes, robust acc: {}, max_in_v: {}, max_in_v_corr: {}, msc: {}, msc_v:{}, '
          'std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(rob_acc, max_in_v, max_in_v_corr, msc, msc_v,
                                                                       std_corr_v_conf, mean_corr_v_conf))
    print()

    if v_classes > 0:
        print('---------------------------------apgd-ce-t-v----------------------------------------')
        version = 'apgd-ce-t-v'
        attacks_to_run = ['apgd-ce-t']
        adversary = AutoAttack(model, norm=norm, eps=attack_eps, version=version, attacks_to_run=attacks_to_run,
                               num_in_classes=NUM_REAL_CLASSES, attack_vclasses=True)
        adversary.apgd_targeted.target_in_v_only = True
        adversary.apgd_targeted.n_target_classes = v_classes - 1
        with torch.no_grad():
            adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
        rob_acc, max_in_v, max_in_v_corr, corr_conf, corr_v_conf \
            = nn_util.eval_from_data(model, adv_complete, y_test, batch_size, NUM_REAL_CLASSES, align_x=x_test)
        msc = corr_conf.mean()
        msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
        if corr_v_conf.size(1) > 1:
            msc_v = corr_v_conf.max(dim=1)[0].mean()
            std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
        print('robust acc: {}, test_cw_max_in_v: {}, test_cw_corr_max_in_v: {}, msc: {}, msc_v:{}, '
              'std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(rob_acc, max_in_v, max_in_v_corr, msc, msc_v,
                                                                         std_corr_v_conf, mean_corr_v_conf))
        print()

        print('---------------------------------apgd-dlr-t-v----------------------------------------')
        version = 'apgd-dlr-t-v'
        attacks_to_run = ['apgd-t']
        adversary = AutoAttack(model, norm=norm, eps=attack_eps, version=version, attacks_to_run=attacks_to_run,
                               num_in_classes=NUM_REAL_CLASSES, attack_vclasses=True)
        adversary.apgd_targeted.target_in_v_only = True
        adversary.apgd_targeted.n_target_classes = v_classes - 1
        with torch.no_grad():
            adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)
        rob_acc, max_in_v, max_in_v_corr, corr_conf, corr_v_conf \
            = nn_util.eval_from_data(model, adv_complete, y_test, batch_size, NUM_REAL_CLASSES, align_x=x_test)
        msc = corr_conf.mean()
        msc_v, std_corr_v_conf, mean_corr_v_conf = 0., 0., 0.
        if corr_v_conf.size(1) > 1:
            msc_v = corr_v_conf.max(dim=1)[0].mean()
            std_corr_v_conf, mean_corr_v_conf = torch.std_mean(corr_v_conf, unbiased=False)
        print('robust acc: {}, test_cw_max_in_v: {}, test_cw_corr_max_in_v: {}, msc: {}, msc_v:{}, '
              'std_corr_v_conf: {}, mean_corr_v_conf: {}'.format(rob_acc, max_in_v, max_in_v_corr, msc, msc_v,
                                                                         std_corr_v_conf, mean_corr_v_conf))
        print()

    return


def main():
    # setup data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        transform_test = T.Compose([T.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar10', train=False, download=True, transform=transform_test), batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        dataloader = datasets.CIFAR100
        transform_test = T.Compose([T.ToTensor()])
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar100', train=False, download=True, transform=transform_test), batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'svhn':
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        svhn_test = torchvision.datasets.SVHN(root='../../datasets/svhn', download=True, transform=transform_test, split='test')
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
    else:
        raise ValueError('error dataset: {0}'.format(args.dataset))

    print('================================================================')
    print('args:', args)
    print('================================================================')
    model = get_model(args.model_name, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes,
                      normalizer=normalizer, dataset=args.dataset).to(device)
    cpt = filter_state_dict(torch.load(args.model_file))
    model.load_state_dict(cpt)

    # standard_aa_eval(model, test_loader, args)
    # aa_eval_real_part(model, test_loader, args)
    aa_eval(model, test_loader, args)


if __name__ == '__main__':
    main()
