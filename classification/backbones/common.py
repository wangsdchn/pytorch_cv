"""
common modules in Pytorch
"""

__all__ = ['conv1x1', 'conv3x3', 'conv1x1_block', 'conv3x3_block']

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
    return nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, groups=channels,
                     bias=use_bias)


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
                 use_bn=True, bn_eps=1e-5, activation_name='relu'):
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

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
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
