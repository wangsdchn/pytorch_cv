import torch


def meshgrid(w, h, row_major=True):
    """
    :param w:
    :param h:
    :param row_major: bool, row_major or col major
    :return: tensor, meshgrid, size: [w*h, 2]

    Example:
        meshgrid(2,3)
        0, 0
        1, 0
        2, 0
        0, 1
        1, 1
        2, 1
    """
    a = torch.arange(0, w)
    b = torch.arange(0, h)
    xx = a.repeat(h).view(-1, 1)
    yy = b.view(-1, 1).repeat(1, w).view(-1, 1)
    return torch.cat([xx, yy], 1) if row_major else torch.cat([yy, xx], 1)


def change_box_order(boxes, order):
    assert order in ['xyxy2xywh', 'xywh2xyxy']
    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == 'xyxy2xywh':
        return torch.cat([(b + a) / 2, (b - a) / 2], 1)
    else:
        return torch.cat([a - b / 2, a + b / 2], 1)


def boxes_iou(box1, box2, order='xyxy'):
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')
    # N = box1.size(0)
    # M = box2.size(0)
    # lt = torch.max(box1[:, :2].view(N, 1, 2).expand(N, M, 2), box2[:, :2].view(1, M, 2).expand(N, M, 2))
    # rb = torch.min(box1[:, 2:].unsqueeze(1).expand(N, M, 2), box2[:, 2:].unsqueeze(0).expand(N, M, 2))
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2], None在指定位置增加维度
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    wh = (rb - lt + 1).clamp(min=0)
    inner = wh[:, :, 0] * wh[:, :, 1]
    area1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)
    # ious = inner / (area1.unsqueeze(1).expand(N, M) + area2.unsqueeze(0).expand(N, M) - inner)
    print(area1.size(), area2.size())
    ious = inner / (area1[:, None] + area2 - inner)
    return ious


def nms_hard(boxes, scores, thresh=0.5):
    """
    :param boxes: tensor, [N, 4]
    :param scores: tensor, [N,]
    :param thresh:
    :return: tensor, selected index
    """
    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    areas = (x1 - x0 + 1) * (y1 - y0 + 1)
    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        xx0 = x0[order[1:]].clamp(min=x0[i])
        yy0 = y0[order[1:]].clamp(min=y0[i])
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        w = (xx1 - xx0 + 1).clamp(min=0)
        h = (yy1 - yy0 + 1).clamp(min=0)
        inter = w * h
        overlap = inter / (areas[1:] + areas[i] - inter)
        ids = (overlap <= thresh).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.tensor(keep, dtype=torch.long)


def boxes_nms(boxes, scores, thresh=0.5, nms_type='default'):
    if nms_type == 'soft':
        boxes_keep = []
    elif nms_type == 'diou':
        boxes_keep = []
    else:   # hard
        boxes_keep = nms_hard(boxes, scores, thresh)
    return boxes_keep
