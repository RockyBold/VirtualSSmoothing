import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn import manifold
import torchvision.transforms as T
import argparse
from atlic.models import wideresnet, resnet, resnet_imagenet, resnet_tiny200, mobilenet_v2
from torch.backends import cudnn
from atlic.utils import nn_util, tiny_datasets, imagenet_loader
from matplotlib.colors import ListedColormap

parser = argparse.ArgumentParser(description='Visualization for TSNE')
parser.add_argument('--model_name', default='mobilenet_v2',
                    help='model name, resnet-18, mobilenet_v2, wrn-28-10, wrn-34-10, wrn-40-4, resnet-18 or resnet-50')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset: svhn, cifar10, cifar100, tiny-imagenet-32x32, tiny-imagenet-64x64 or imagenet')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model_file', default='./dnn_models/dataset/model.pt', help='file path of model')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--v_classes', default=10, type=int,
                    help='the number of label-irrelevant nodes in last layer')

args = parser.parse_args()

args.model_name = 'mobilenet_v2'
args.model_file = './dnn_models/clean_mob-v2_vclasses0_epoch173.pt'
prefix='st'

# args.model_name = 'resnet-18'
# args.model_file = './dnn_models/clean_resnet18-vcls0-alpha0_epoch200.pt'
# prefix='st'

# args.model_name = 'resnet-18'
# # args.model_file = './dnn_models/clean_resnet18-vcls0-alpha0.3_epoch188.pt'
# # args.model_file = './dnn_models/clean_resnet18-vcls0-alpha0.5_epoch199.pt'
# # args.model_file = './dnn_models/clean_resnet18-vcls0-alpha0.8_epoch185.pt'
# args.model_file = './dnn_models/clean_resnet18-vcls0-alpha0.95_epoch160.pt'
# # args.model_file = './dnn_models/clean_resnet18-vcls0-alpha0.99_epoch154.pt'
# prefix='ls-0.95'
# args.v_classes = 0

# args.model_name = 'resnet-18'
# args.model_file = './dnn_models/clean_resnet18-vcls10-alpha0.5_epoch193.pt'
# prefix='vs-0.5'
# args.v_classes = 10

if args.dataset == 'cifar10':
    NUM_REAL_CLASSES = 10
elif args.dataset == 'svhn':
    NUM_REAL_CLASSES = 10
elif args.dataset == 'cifar100':
    NUM_REAL_CLASSES = 100
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(1)
device = torch.device("cuda" if use_cuda else "cpu")
storage_device = torch.device("cpu")
normalizer = None

# digits = datasets.load_digits(n_class=6)
# X, y = digits.data, digits.target
# n_samples, n_features = X.shape
#
# '''显示原始数据'''
# n = 20  # 每行20个数字，每列20个数字
# img = np.zeros((10 * n, 10 * n))
# for i in range(n):
#     ix = 10 * i + 1
#     for j in range(n):
#         iy = 10 * j + 1
#         img[ix:ix + 8, iy:iy + 8] = X[i * n + j].reshape((8, 8))
# plt.figure(figsize=(8, 8))
# plt.imshow(img, cmap=plt.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
#
# '''t-SNE'''
# tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
# X_tsne = tsne.fit_transform(X)
#
# print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
#
# '''嵌入空间可视化'''
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
# plt.figure(figsize=(8, 8))
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xticks([])
# plt.yticks([])
# plt.show()

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


def visualize_tsne(X, y, label_ind=[], prefix=''):
    label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                  4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                  8: 'ship', 9: 'truck'}

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # plt.figure(figsize=(8, 8))
    # for i in range(X_norm.shape[0]):
    #     color = plt.cm.Set1(y[i])
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=color, fontdict={'weight': 'bold', 'size': 9})
    # plt.legend()
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig('{}--{}-{}-digit.jpg'.format(prefix, args.model_name, args.v_classes))
    # plt.show()

    # import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 8))
    classes = ['airp-\nlane', 'auto-\nmobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    row_colors = ['#FFB6C1', '#DF21D6', '#880EBE', '#FB2E05', '#2CD43C', '#757C8B', '#F4A460', '#4B04FC', '#800000', '#DD5823']
    colors = ListedColormap(row_colors)
    if len(label_ind) > 0:
        tem_classes = []
        for i in label_ind:
            tem_classes.append(classes[i])
        classes = tem_classes

    # scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap=colors)
    scatter = plt.scatter(X_norm[:, 0], X_norm[:, 1], c=y, cmap=colors, linewidths=np.zeros(X_norm.shape[0])+0.2, marker='o')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, bbox_to_anchor=(1, 0), loc=3, prop = {'size':20})
    plt.xticks([])
    plt.yticks([])
    plt.savefig('{}--{}-{}.jpg'.format(prefix, args.model_name, args.v_classes), bbox_inches = 'tight')
    plt.show()


def get_correct_logits(test_loader, model, device=torch.device('cuda')):
    logits = []
    y = []
    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        if use_cuda:
            batch_x = batch_x.cuda(non_blocking=True)
            batch_y = batch_y.cuda(non_blocking=True)
        print('batch {}'.format(i+1))

        with torch.no_grad():
            batch_logits = model(batch_x)
        pred = batch_logits[:, :NUM_REAL_CLASSES].max(dim=1)[1]
        corr_idx = pred == batch_y
        # logits.append(batch_logits[corr_idx])
        # cls_idx = batch_y < 3
        # idx = torch.logical_and(corr_idx, cls_idx)
        logits.append(batch_logits[:, :NUM_REAL_CLASSES][corr_idx])
        y.append(batch_y[corr_idx])
        # if i > 2:
        #     break

    logits = torch.cat(logits)
    y = torch.cat(y)
    return logits, y


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


def temp_func():
    print('---------------------------vary alpha-------------------------------------------')
    res18_acc = np.array([94.69, 95.21, 95.3, 95.12, 95.20, 95.27, 94.98])
    print('cifar10 res18_acc: mean-std:({}, {})'.format(res18_acc.mean(), res18_acc.std()))
    resx29_acc = np.array([94.44, 94.76, 95.19, 94.88, 95.16, 95.27, 94.91])
    print('cifar10 resx29_acc: mean-std:({}, {})'.format(resx29_acc.mean(), resx29_acc.std()))
    cf100_res18_acc = np.array([76.29, 76.94, 77.06, 76.99, 77.67, 78.05, 77.16])
    print('cifar100 cf100_res18_acc: mean-std:({}, {})'.format(cf100_res18_acc.mean(), cf100_res18_acc.std()))
    cf100_resx29_acc = np.array([76.59, 77.86, 77.23, 77.72, 78.58, 79.06, 79.84])
    print('cifar100 cf100_resx29_acc: mean-std:({}, {})'.format(cf100_resx29_acc.mean(), cf100_resx29_acc.std()))
    print()
    print()

    res18_aa = np.array([49.28, 49.26, 49.58, 50.03, 48.88])
    print('cifar10 res18_aa: mean-std:({}, {})'.format(res18_aa.mean(), res18_aa.std()))
    wrn34_aa = np.array([52.22, 52.34, 52.29, 52.70, 53.16, 52.16])
    print('cifar10 wrn34_aa: mean-std:({}, {})'.format(wrn34_aa.mean(), wrn34_aa.std()))
    cf100_res18_aa = np.array([25.78, 26.26, 26.75, 25.92])
    print('cifar100 cf100_res18_aa: mean-std:({}, {})'.format(cf100_res18_aa.mean(), cf100_res18_aa.std()))
    cf100_wrn34_aa = np.array([28.37, 28.25, 28.56, 28.63])
    print('cifar100 cf100_wrn34_aa: mean-std:({}, {})'.format(cf100_wrn34_aa.mean(), cf100_wrn34_aa.std()))
    print()
    print()

    print('---------------------------vary V-------------------------------------------')
    res18_acc = np.array([94.95, 95.03, 95.3, 95.29, 95.03])
    print('cifar10 res18_acc: mean-std:({}, {})'.format(res18_acc.mean(), res18_acc.std()))
    resx29_acc = np.array([94.39, 94.83, 95.27, 95.27, 95.49])
    print('cifar10 resx29_acc: mean-std:({}, {})'.format(resx29_acc.mean(), resx29_acc.std()))
    print()
    res18_aa = np.array([49.14, 50.03, 49.44, 48.99])
    print('cifar10 res18_aa: mean-std:({}, {})'.format(res18_aa.mean(), res18_aa.std()))
    wrn34_aa = np.array([52.39, 53.16, 52.55, 52.9])
    print('cifar10 wrn34_aa: mean-std:({}, {})'.format(wrn34_aa.mean(), wrn34_aa.std()))


if __name__ == '__main__':

    # import matplotlib.pyplot as plt
    # from matplotlib.colors import ListedColormap
    #
    # x = [1, 3, 4, 6, 7, 9]
    # y = [0, 0, 5, 8, 8, 8]
    # classes = ['A', 'B', 'C']
    # values = [0, 0, 1, 2, 2, 2]
    # colors = ListedColormap(['r', 'b', 'g'])
    # scatter = plt.scatter(x, y, c=values, cmap=colors)
    # plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    # plt.show()

    # setup data loader
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    if args.dataset == 'cifar10':
        transform_test = T.Compose([T.ToTensor()])
        dataloader = torchvision.datasets.CIFAR10
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar10/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        transform_test = T.Compose([T.ToTensor()])
        dataloader = torchvision.datasets.CIFAR100
        test_loader = torch.utils.data.DataLoader(
            dataloader('../../datasets/cifar100/', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'svhn':
        # normalizer = None
        transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        svhn_test = torchvision.datasets.SVHN(root='../../datasets/svhn/', download=True, transform=transform_test,
                                              split='test')
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
    elif args.dataset == 'tiny-imagenet-32x32':
        data_dir = '../../datasets/tiny-imagenet-200/'
        normalize = T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        # transform_train = T.Compose([T.RandomResizedCrop(32), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        transform_test = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            normalize
        ])
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'tiny-imagenet-64x64':
        data_dir = '../../datasets/tiny-imagenet-200/'
        normalize = T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        # transform_train = T.Compose([T.RandomResizedCrop(32), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
        transform_test = T.Compose([
            T.ToTensor(),
            normalize
        ])
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'imagenet':
        imagenet_root = '../../datasets/'
        _, test_loader = imagenet_loader.data_loader(imagenet_root, batch_size=args.batch_size)
    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))
    cudnn.benchmark = True

    # init model, Net() can be also used here for training
    model = get_model(args.model_name, num_real_classes=NUM_REAL_CLASSES, num_v_classes=args.v_classes,
                      normalizer=normalizer, dataset=args.dataset).to(device)
    cpt = filter_state_dict(torch.load(args.model_file, map_location=torch.device('cpu')))
    model.load_state_dict(cpt)

    logits, y = get_correct_logits(test_loader, model, device)
    ind = torch.logical_or(y == 5, y == 7)
    ind = torch.logical_or(ind, y == 9)
    print('sum:', torch.sum(ind))
    logits = logits[ind].cpu().numpy()
    y = y[ind].cpu().numpy()
    prefix = prefix + '-579'
    visualize_tsne(logits, y, label_ind=[5, 7, 9], prefix=prefix)
    pass
