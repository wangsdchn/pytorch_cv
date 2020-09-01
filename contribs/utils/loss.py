from torch import nn


def create_criterion(args):
    if args.loss_type == 'FocalLoss':
        criterion = None
    elif args.loss_type == 'MSELoss':
        criterion = nn.MSELoss()
    elif args.loss_type == 'L1Loss':
        criterion = nn.L1Loss()
    elif args.loss_type == 'SmoothL1Loss':
        criterion = nn.SmoothL1Loss()
    elif args.loss_type == 'CTCLoss':
        criterion = nn.CTCLoss()
    elif args.loss_type == 'KLDivLoss': # 衡量两个分布之间的差异
        criterion = nn.KLDivLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion
