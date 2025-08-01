import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_real_classes=10, num_v_classes=0, normalizer=None, output_real_only=False,
                 output_v_only=False, add_real_v=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_real_classes + num_v_classes)
        self.normalizer = normalizer
        self.num_real_classes = num_real_classes
        self.num_v_classes = num_v_classes
        self.output_real_only = output_real_only
        self.output_v_only = output_v_only
        assert (self.output_real_only & self.output_v_only) == False
        self.add_real_v = add_real_v
        if self.add_real_v:
            assert self.num_real_classes == self.num_v_classes


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        assert (self.output_real_only & self.output_v_only) == False
        if self.normalizer is not None:
            x = x.clone()
            x[:, 0, :, :] = (x[:, 0, :, :] - self.normalizer.mean[0]) / self.normalizer.std[0]
            x[:, 1, :, :] = (x[:, 1, :, :] - self.normalizer.mean[1]) / self.normalizer.std[1]
            x[:, 2, :, :] = (x[:, 2, :, :] - self.normalizer.mean[2]) / self.normalizer.std[2]

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.output_v_only:
            out = out[:, self.num_real_classes:self.num_real_classes + self.num_v_classes]
        else:
            if self.output_real_only:
                out = out[:, :self.num_real_classes]
            else:
                if self.add_real_v:
                    assert self.num_real_classes == self.num_v_classes
                    out = out[:, :self.num_real_classes] + out[:, self.num_real_classes:self.num_real_classes + self.num_v_classes]
                else:
                    out = out
        return out



def ResNet18(num_real_classes=10, num_v_classes=0, normalizer=None, output_real_only=False, output_v_only=False,
             add_real_v=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                  normalizer=normalizer, output_real_only=output_real_only, output_v_only=output_v_only,
                  add_real_v=add_real_v)
                 # change the BasicBlock array to [2,1,1,1] or anyother.


def ResNet34(num_real_classes=10, num_v_classes=0, normalizer=None, output_real_only=False, output_v_only=False,
             add_real_v=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                  normalizer=normalizer, output_real_only=output_real_only, output_v_only=output_v_only,
                  add_real_v=add_real_v)
                 # change the BasicBlock array to [1,2,2,1] or anyother.


def ResNet50(num_real_classes=10, num_v_classes=0, normalizer=None, output_real_only=False, output_v_only=False,
             add_real_v=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                  normalizer=normalizer, output_real_only=output_real_only, output_v_only=output_v_only,
                  add_real_v=add_real_v)


def ResNet101(num_real_classes=10, num_v_classes=0, normalizer=None, output_real_only=False, output_v_only=False,
              add_real_v=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                  normalizer=normalizer, output_real_only=output_real_only, output_v_only=output_v_only,
                  add_real_v=add_real_v)


def ResNet152(num_real_classes=10, num_v_classes=0, normalizer=None, output_real_only=False, output_v_only=False,
              add_real_v=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_real_classes=num_real_classes, num_v_classes=num_v_classes,
                  normalizer=normalizer, output_real_only=output_real_only, output_v_only=output_v_only,
                  add_real_v=add_real_v)


if __name__ == '__main__':
    from torchsummary import summary
    net = ResNet50(num_real_classes=10)
    summary(net, (3, 64, 64))
