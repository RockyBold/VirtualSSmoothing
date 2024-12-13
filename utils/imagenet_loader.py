import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

'''
1). pre-process training data:
mkdir ILSVRC2012_img_train && mv ILSVRC2012_img_train.tar ILSVRC2012_img_train/ && cd ILSVRC2012_img_train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

2). pre-process testing data:
mkdir ILSVRC2012_img_val && mv ILSVRC2012_img_val.tar ILSVRC2012_img_val/ && cd ILSVRC2012_img_val && tar -xvf ILSVRC2012_img_val.tar
sh valprep.sh

'''


def data_loader(root, batch_size=256, workers=4, pin_memory=True, randaug=False, num_ops=2, magnitude=9):
    traindir = os.path.join(root, 'ILSVRC2012_img_train')
    valdir = os.path.join(root, 'ILSVRC2012_img_val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if randaug:
        from torchvision.transforms import v2
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                v2.RandAugment(num_ops=num_ops, magnitude=magnitude),
                transforms.ToTensor(),
                normalize
            ])
        )
    else:
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader
