"""
common modules in Pytorch
"""

__all__ = ['conv1x1', 'conv3x3', 'conv1x1_block', 'conv3x3_block']

import torch.nn as nn


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kerner_size=1,
        stride=stride,
        groups=groups,
        bias=bias
    )


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1, groups=1, bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kerner_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias
    )

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    else:
        return None
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, dilation=1, bias=False,
                 use_bn=True, bn_eps=1e5, activation='relu'):
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
            bias=bias
        )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(
                num_features=out_channels,
                eps=bn_eps
            )
        self.activate = activation

