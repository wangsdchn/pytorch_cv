import os
import numpy as np
from contribs.utils.nms import nms

if __name__ == '__main__':
    bboxes = np.array([[12, 15, 26, 29, 0.5], [18, 26, 30, 36, 0.3]])
    idx_keep = nms(bboxes, 0.5)
    print(idx_keep)
    print(bboxes[idx_keep])