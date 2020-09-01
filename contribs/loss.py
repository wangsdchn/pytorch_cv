from torch import nn


def create_criterion(args):
    if args.loss_type == 'FocalLoss':
        criterion = None
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    return criterion
