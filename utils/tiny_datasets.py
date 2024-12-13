import os
import numpy as np
import torch
# from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
# from PIL import Image


# class CIFAR10(datasets.CIFAR10):
#     def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
#         super(CIFAR10, self).__init__(root, train=train, transform=transform,
#                                       target_transform=target_transform, download=download)
#
#         # unify the interface
#         if not hasattr(self, 'data'):  # torch <= 0.4.1
#             if self.train:
#                 self.data, self.targets = self.train_data, self.train_labels
#             else:
#                 self.data, self.targets = self.test_data, self.test_labels
#
#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, index
#
#     @property
#     def num_classes(self):
#         return 10
#
#
# class CIFAR100(datasets.CIFAR100):
#     def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
#         super(CIFAR100, self).__init__(root, train=train, transform=transform,
#                                        target_transform=target_transform, download=download)
#
#         # unify the interface
#         if not hasattr(self, 'data'):  # torch <= 0.4.1
#             if self.train:
#                 self.data, self.targets = self.train_data, self.train_labels
#             else:
#                 self.data, self.targets = self.test_data, self.test_labels
#
#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]
#
#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
#         img = Image.fromarray(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, index
#
#     @property
#     def num_classes(self):
#         return 100


def arrange_val_set(target_folder = '../../../datasets/tiny-imagenet-200/val/'):
    import glob
    import os
    from shutil import move
    from os import rmdir

    # target_folder = '../datasets/tiny-imagenet-200/val/'

    val_dict = {}
    with open(target_folder + '/val_annotations.txt', 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            val_dict[split_line[0]] = split_line[1]

    paths = glob.glob(target_folder + 'images/*')
    for path in paths:
        file = path.split(os.sep)[-1]
        folder = val_dict[file]
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')

    for path in paths:
        file = path.split(os.sep)[-1]
        folder = val_dict[file]
        dest = target_folder + str(folder) + '/images/' + str(file)
        move(path, dest)

    rmdir(target_folder + '/images')


def tiny_loader(batch_size, data_dir='../../../datasets/tiny-imagenet-200/', transform_train=None,
                transform_test=None):
    import torchvision.transforms as T
    normalize = T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    if transform_train is None:
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
    if transform_test is None:
        transform_test = T.Compose([
            T.Resize(32),
            T.ToTensor(),
            normalize
        ])
    trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader

if __name__ == '__main__':
    arrange_val_set()
