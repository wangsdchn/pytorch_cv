import numpy as np


def nms(bboxes, iou_thresh):
    """
    :param bboxes: [[x0,y0,x1,y1,score]]
    :param iou_thresh
    :return: bboxes index remain
    """
    x0 = bboxes[:, 0]
    y0 = bboxes[:, 1]
    x1 = bboxes[:, 2]
    y1 = bboxes[:, 3]
    score = bboxes[:, 4]
    area = (y1 - y0 + 1) * (x1 - x0 + 1)

    keep = []
    index = score.argsort()[::-1]
    while index.size > 0:
        keep.append(index[0])
        x_lt = np.maximum(x0[index[0]], x0[index[1:]])
        y_lt = np.maximum(y0[index[0]], y0[index[1:]])
        x_rb = np.minimum(x1[index[0]], x1[index[1:]])
        y_rb = np.minimum(y1[index[0]], y1[index[1:]])
        h = np.maximum(0, y_rb - y_lt + 1)
        w = np.maximum(0, x_rb - x_lt + 1)
        overlap = w * h
        ious = overlap / (area[index[0]] + area[index[1:]] - overlap)
        index = index[np.where(ious < iou_thresh)[0] + 1]
    return keep


if __name__ == '__main__':
    bboxes = np.array([[12, 15, 26, 29, 0.5], [18, 26, 30, 36, 0.3]])
    idx_keep = nms(bboxes, 0.5)
    print(idx_keep)
    print(bboxes[idx_keep])
