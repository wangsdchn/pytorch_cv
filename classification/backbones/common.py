"""
common modules in Pytorch
"""

__all__ = ['conv1x1', 'conv3x3', 'conv1x1_block', 'conv3x3_block', 'SeLayer', 'conv1x1_pre_block', 'conv3x3_pre_block',
           'get_pooling_func', 'get_activation']

import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias
    )


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1, groups=1, bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )


def conv3x3_dw(channels, stride, use_bias=False):
    return nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        groups=channels,
        bias=use_bias
    )


class Swish(nn.Module):
    """
    f(x) = x · sigmoid(x)
    """

    def forward(self, x):
        return x.mul_(F.sigmoid(x))


class HSwish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class HSigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


def get_activation(activation_name='relu'):
    if activation_name == 'sigmoid':
        return nn.Sigmoid()
    elif activation_name == "relu":
        return nn.ReLU(inplace=True)
    elif activation_name == 'relu6':
        return nn.ReLU6(inplace=True)
    elif activation_name == 'swish':
        return Swish()
    elif activation_name == 'hswish':
        return HSwish()
    elif activation_name == 'hsigmoid':
        return HSigmoid()
    else:
        return None


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, dilation=1, use_bias=False,
                 use_bn=True, bn_eps=1e-5, activation_name='relu', pooling_name=None):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            dilation=dilation,
            bias=use_bias
        )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps
            )
        self.activation = get_activation(activation_name)
        self.pool = get_pooling_func(pooling_name)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        if self.pool:
            x = self.pool(x)
        return x


def conv1x1_block(in_channels, out_channels, kernel_size=1, stride=1, groups=1, use_bias=False, use_bn=True,
                  bn_eps=1e-5,
                  activation_name='relu'):
    return ConvBlock(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,
                     use_bias=use_bias, use_bn=use_bn, bn_eps=bn_eps, activation_name=activation_name)


def conv3x3_block(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, use_bias=False,
                  use_bn=True, bn_eps=1e-5, activation_name='relu'):
    return ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                     stride=stride, groups=groups, use_bias=use_bias, use_bn=use_bn, bn_eps=bn_eps,
                     activation_name=activation_name)


def conv7x7_block(in_channels, out_channels, kernel_size=7, padding=3, dilation=1, stride=2, groups=1,
                  use_bias=False, use_bn=True, bn_eps=1e-5, activation_name='relu', pooling_name='max'):
    return ConvBlock(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                     stride=stride, groups=groups, use_bias=use_bias, use_bn=use_bn, bn_eps=bn_eps,
                     activation_name=activation_name, pooling_name=pooling_name)


def conv3x3_dw_block(channels, kernel_size=3, padding=1, dilation=1, use_bias=False, bn_eps=1e-5,
                     activation_name='relu'):
    return ConvBlock(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_eps=bn_eps,
        activation_name=activation_name
    )


class SeLayer(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SeLayer, self).__init__()
        mid_channels = in_channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1_block(in_channels, mid_channels, activation_name='relu'),
            conv1x1_block(mid_channels, in_channels, activation_name='hsigmoid')
        )

    def forward(self, x):
        return x * self.se(x)


class PreConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, dilation=1, use_bias=False,
                 use_bn=True, bn_eps=1e-5, activation_name='relu'):
        super(PreConvBlock, self).__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps
            )
        self.activation = get_activation(activation_name)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            dilation=dilation,
            bias=use_bias
        )

    def forward(self, x):
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        x = self.conv(x)
        return x


def conv1x1_pre_block(in_channels, out_channels, kernel_size=1, stride=1, use_bias=False, use_bn=True,
                      bn_eps=1e-5,
                      activation_name='relu'):
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        use_bias=use_bias,
        bn_eps=bn_eps,
        use_bn=use_bn,
        activation_name=activation_name)


def conv3x3_pre_block(in_channels, out_channels, kernel_size=3, stride=1, use_bias=False, use_bn=True,
                      bn_eps=1e-5, padding=1, dilation=1,
                      activation_name='relu'):
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        use_bias=use_bias,
        bn_eps=bn_eps,
        use_bn=use_bn,
        activation_name=activation_name)


def get_pooling_func(pooling_name, kernal_size=2, stride=2, padding=0, out_size=1):
    if pooling_name == 'max':
        return nn.MaxPool2d(kernel_size=kernal_size, stride=stride, padding=padding)
    elif pooling_name == 'avg':
        return nn.AvgPool2d(kernel_size=kernal_size, stride=stride, padding=padding)
    elif pooling_name == 'gap':
        return nn.AdaptiveAvgPool2d(output_size=out_size)
    elif pooling_name == 'gmp':
        return nn.AdaptiveMaxPool2d(output_size=out_size)
    else:
        return None
