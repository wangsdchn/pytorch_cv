import torch
import math
from utils import meshgrid, boxes_nms, boxes_iou, change_box_order


class DataEncoder:
    def __init__(self, input_size=600, cls_thresh=0.5, nms_thresh=0.5):
        scale = (1. * input_size / 600) ** 2
        self.input_size = input_size
        self.CLS_THRESH = cls_thresh
        self.NMS_THRESH = nms_thresh
        self.anchor_areas = [32 * 32. * scale, 64 * 64. * scale, 128 * 128. * scale, 256 * 256. * scale,
                             512 * 512. * scale]
        self.aspect_ratios = [1 / 2., 1., 2.]
        self.scale_ratios = [1., pow(2, 1 / 3.), pow(2, 2 / 3.)]
        self.anchor_boxes = self._get_anchor_boxes()

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

    def _get_anchor_boxes(self):
        """ Create anchor boxes for each feature map
        Args:
            input_size: tensor, model input size: (w, h)
        Return:
            boxes: list, anchor boxes for each feature map, size: [#fms * #anchors, 4]
                    #anchors = fmw * fmh * #anchor_per_cell
        """
        input_size = torch.tensor([self.input_size, self.input_size])
        self.anchor_wh = self._get_anchor_wh()
        num_fms = len(self.anchor_areas)
        fm_sizes = [(input_size / pow(2., i + 3)).ceil() for i in range(num_fms)]
        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            grid_size = input_size / fm_size
            fmw, fmh = int(fm_size[0]), int(fm_size[1])
            grid = meshgrid(fmw, fmh)
            xy = grid.float() + 0.5
            xy = (xy * grid_size).view(fmw, fmh, 1, 2).expand(fmw, fmh, 9, 2)
            wh = self.anchor_wh[i].view(1, 1, 9, 2).expand(fmw, fmh, 9, 2)
            box = torch.cat([xy, wh], 3)
            boxes.append(box.view(-1, 4))
        return torch.cat(boxes, 0)

    def encode(self, boxes, labels):
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
        boxes = boxes.float()
        boxes = change_box_order(boxes, 'xyxy2xywh')
        ious = boxes_iou(self.anchor_boxes, boxes, 'xywh')
        max_ious, max_ids = ious.max(1)  # 一个anchor只对应一个gt
        boxes = boxes[max_ids]
        loc_xy = (boxes[:, :2] - self.anchor_boxes[:, :2]) / self.anchor_boxes[:, :2]
        loc_wh = torch.log(boxes[:, 2:] / self.anchor_boxes[:, 2:])
        loc_target = torch.cat([loc_xy, loc_wh], dim=1)
        cls_target = 1 + labels[max_ids]

        cls_target[max_ious < 0.5] = 0
        ignore = (max_ious > 0.4) & (max_ious < 0.5)
        cls_target[ignore] = -1
        return loc_target, cls_target

    def decode(self, loc_pred, cls_pred):
        loc_xy = loc_pred[:, :2]
        loc_wh = loc_pred[:, 2:]
        xy = loc_xy * self.anchor_boxes[:, :2] + self.anchor_boxes[:, :2]
        wh = loc_wh.exp() * self.anchor_boxes[:, 2:]
        boxes = torch.cat([xy - wh / 2, xy + wh / 2], 1)  # [#anchors, 4]
        scores, preds = cls_pred.sigmoid().max(1)
        ids = scores > self.CLS_THRESH
        ids = ids.nonzero().squeeze()
        keep = boxes_nms(boxes[ids], scores[ids], thresh=self.NMS_THRESH, nms_type='default')
        return torch.cat([boxes[keep], scores[keep], preds[keep]], 1)


if __name__ == '__main__':
    boxes = torch.tensor([[2, 5, 9, 10], [20, 30, 70, 90]])
    labels = torch.tensor([1, 2])
    input_size = 100
    datacoder = DataEncoder(input_size=input_size)
    datacoder.encode(boxes, labels)
