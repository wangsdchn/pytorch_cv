# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .common import conv1x1_pre_block, conv3x3_pre_block, conv7x7_block, get_pooling_func


class DenseUnit(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, dropout_rate=0.5):
        super(DenseUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        mid_channels = growth_rate * bn_size
        self.conv1 = conv1x1_pre_block(in_channels=in_channels, out_channels=mid_channels)
        self.conv2 = conv3x3_pre_block(in_channels=mid_channels, out_channels=growth_rate)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = torch.cat((residual, x), dim=1)
        return x


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.conv = conv1x1_pre_block(in_channels=in_channels, out_channels=out_channels)
        self.pool = get_pooling_func(pooling_name='avg')

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, bn_size=4, dropout_rate=0.5):
        super(DenseBlock, self).__init__()
        self.denselayers = nn.ModuleList()
        num_channels = n = in_channels
        for i in range(num_layers):
            num_channels = num_channels + i * growth_rate
            layer = DenseUnit(
                in_channels=num_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=dropout_rate
            )
            self.denselayers.append(layer)

    def forward(self, x):
        features = [x]
        for denselayer in self.denselayers:
            new_feature = denselayer(*features)
            features.append(new_feature)
        x = torch.cat(features, dim=1)
        return x


class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5, num_init_feature=24, bn_size=4,
                 drop_rate=0, num_classes=10):
        super(DenseNet, self).__init__()
        assert 0 < compression <= 1
        self.densenet = nn.Sequential()
        conv1 = conv7x7_block(in_channels=3, out_channels=num_init_feature, pooling_name='max')
        self.densenet.add_module('conv1', conv1)
        num_features = num_init_feature
        for i, num_layers in enumerate(block_config):
            denseblock = DenseBlock(
                in_channels=num_features,
                num_layers=num_layers,
                growth_rate=growth_rate,
                bn_size=bn_size,
                dropout_rate=drop_rate
            )
            self.densenet.add_module('denseblock%d' % (i + 1), denseblock)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                transblock = TransitionBlock(
                    in_channels=num_features,
                    out_channels=int(num_features * compression)
                )
                self.densenet.add_module('trandition%d' % (i + 1), transblock)
                num_features = int(num_features * compression)
        self.densenet.add_module('normal_last', nn.BatchNorm2d(num_features))
        self.densenet.add_module('normal_relu', nn.ReLU(inplace=True))
        self.gap = get_pooling_func(pooling_name='gap')
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.densenet(x)
        x = self.gap(x)
        x = self.linear(x)
        return x
