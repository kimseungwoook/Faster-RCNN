import torch

def box_iou(boxes1, boxes2):
    """
    Parameters:
        boxse1 : Anchor box --> [N, 4]
        boxse2 : GT box --> [M, 4]

    :return:
        IoU : Anchor box 와 GT box 의 IoU
            [N, M]
    """

    boxes1_min = boxes1[:, None, :2]
    boxes2_min = boxes2[None, :, :2]
    boxes1_max = boxes1[:, None, 2:]
    boxes2_max = boxes2[None, :, 2:]

    # N, M, 2
    inter_min = torch.maximum(boxes1_min, boxes2_min)
    # N, M, 2
    inter_max = torch.minimum(boxes1_max, boxes2_max)

    inter_w = torch.clamp(inter_max[:, :, :1] - inter_min[:, :, :1], min=0)
    inter_h = torch.clamp(inter_max[:, :, 1:] - inter_min[:, :, 1:], min=0)

    # (N, M)
    inter_area = (inter_w * inter_h).squeeze(-1)

    boxes1_w = boxes1[:, 2] - boxes1[:, 0]
    boxes2_w = boxes2[:, 2] - boxes2[:, 0]
    boxes1_h = boxes1[:, 3] - boxes1[:, 1]
    boxes2_h = boxes2[:, 3] - boxes2[:, 1]

    # boxes1_area [N], boxes2_area[M]
    boxes1_area = (boxes1_w * boxes1_h).unsqueeze(1)
    boxes2_area = (boxes2_w * boxes2_h).unsqueeze(0)

    # unio_area = [N, M]
    union_area = boxes1_area + boxes2_area - inter_area

    IoU = inter_area / union_area.clamp(min=1e-6)

    return IoU

def encode_boxes(anchors, gt_boxes):
    """
    anchor box를 GT_box와의 좌표를 통해 좌표를 예측
    GT box는 anchor_box와 매칭된 gt_box를 주기 떄문에 [N, 4] 형태이다.
        param anchors: [N, 4]
        param gt_boxes: [N, 4]

    return:
        regression_targets : [N, 4]
    """
    # anchor = [N, 4]
    anchor_center_x = (anchors[:, 2] + anchors[:, 0]) / 2
    anchor_center_y = (anchors[:, 3] + anchors[:, 1]) / 2
    # GT = [M, 4]
    gt_center_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
    gt_center_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

    tx = (gt_center_x - anchor_center_x) / (anchors[:, 2] - anchors[:, 0])
    ty = (gt_center_y - anchor_center_y) / (anchors[:, 3] - anchors[:, 1])
    tw = torch.log((gt_boxes[:, 2] - gt_boxes[:, 0]) / (anchors[:, 2] - anchors[:, 0]))
    th = torch.log((gt_boxes[:, 3] - gt_boxes[:, 1]) / (anchors[:, 3] - anchors[:, 1]))

    stack = torch.stack([tx, ty, tw, th], dim=1)

    return stack

def decode_boxes():
    pass