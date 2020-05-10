# -*- coding: utf-8 -*-
import torch.nn as nn
from .common import SeLayer, conv1x1_block, conv3x3_block, get_activation
from torchvision.models.resnet import ResNet

__all__ = ['se_resnet18', 'se_resnet34', 'se_resnet50', 'se_resnet101', 'se_resnet152']


class SeBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SeBasicBlock, self).__init__()
        self.conv1 = conv3x3_block(in_channels, out_channels, stride=stride, activation_name='relu')
        self.conv2 = conv3x3_block(out_channels, out_channels, stride=1, activation_name=None)
        self.se = SeLayer(out_channels, reduction=reduction)
        self.relu = get_activation('relu')
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class SeBottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SeBottleBlock, self).__init__()
        self.conv1 = conv1x1_block(in_channels, out_channels, activation_name='relu')
        self.conv2 = conv3x3_block(out_channels, out_channels, activation_name='relu')
        self.conv3 = conv1x1_block(out_channels, out_channels * self.expansion, activation_name=None)
        self.se = SeLayer(out_channels * self.expansion, reduction=reduction)
        self.relu = get_activation('relu')
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def se_resnet18(num_classes=1000):
    model = ResNet(SeBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model


def se_resnet34(num_classes=1000):
    model = ResNet(SeBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


def se_resnet50(num_classes=1000):
    model = ResNet(SeBottleBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


def se_resnet101(num_classes=1000):
    model = ResNet(SeBottleBlock, [3, 4, 23, 3], num_classes=num_classes)
    return model


def se_resnet152(num_classes=1000):
    model = ResNet(SeBottleBlock, [3, 8, 36, 3], num_classes=num_classes)
    return model
