import torch
import math


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


class DataEncoder:
    def __init__(self):
        self.anchor_areas = [32 * 32., 64 * 64., 128 * 128., 256 * 256., 512 * 512.]
        self.aspect_ratios = [1 / 2., 1., 2.]
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        """
        Compute anchor width and height for each feature map
        Return:
            anchor_wh: [num_fms, anchors_per_cell, 2]
        """
        anchor_wh = []
        for area in self.anchor_areas:
            for ar in self.aspect_ratios:
                h = math.sqrt(area / ar)
                w = ar * h
                for sr in self.scale_ratios:
                    anchor_h = h * sr
                    anchor_w = w * sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        """ Create anchor boxes for each feature map
        Args:
            input_size: tensor, model input size: (w, h)
        Return:
            boxes: list, anchor boxes for each feature map, size: [#fms * #anchors, 4]
                    #anchors = fmw * fmh * #anchor_per_cell
        """
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2., i+3)).ceil() for i in range(num_fms)]
        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fmw, fmh = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fmw, fmh) + 0.5

    def encode(self, boxes, labels, input_size):
        """Encode target bounding boxes and class labels
        Box coder(Faster RCNN):
            tx = (x - anchor_x) / anchor_w
            ty = (y - anchor_y) / anchor_h
            tw = log(w / anchor_w)
            th = log(h / anchor_h)

        Args:
            boxes: tensor, [xmin, ymin, xmax, ymax], size: [#objs, 4]
            labels: tensor, size: [#objs,]
            input_size: int/tuple, model input size of (w, h)

        Return:
            loc_targets: tensor, encoded bboxes, size: [#anchors, 4]
            cls_targets: tensor, encoded labels, size: [#anchors,]
        """
        input_size = torch.tensor([input_size, input_size]) if isinstance(input_size, int) else torch.tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
