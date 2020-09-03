from torch import nn


def create_criterion(args):  # default: cross_entropy_loss
    """
    FocalLoss: FL(pt) = -alpha*(1-pt)^gamma*log(pt)
    L2 = |f(x)-y|^2
    L1 = |f(x)-y|
    SmoothL1Loss(x) = 0.5*x^2 if |x|<1 else |x|-0.5
    CE(pt) = -y*log(pt)
    """
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
    elif args.loss_type == 'KLDivLoss':  # 衡量两个分布之间的差异
        criterion = nn.KLDivLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion
