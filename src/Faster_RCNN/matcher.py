
import torch
from src.Faster_RCNN.box_ops import box_iou


class AnchorMatcher:
    def __init__(self, pos_iou=0.7, neg_iou=0.3):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou

    def match(self, anchors, gt_boxes):
        """
        Args:
            anchors: [N, 4]
            gt_boxes: [M, 4]

        Returns:
            labels: [N]
                1  : Positive
                0  : Negative
                -1 : Ignore

            matched_gt_indices: [N]
                각 Anchor의 최대 IoU GT 인덱스
        """

        num_anchors = anchors.shape[0]

        labels = torch.full(
            (num_anchors,),
            -1,
            dtype=torch.int64,
            device=anchors.device
        )

        # GT가 없는 경우 모든 Anchor를 Negative로 처리
        if gt_boxes.shape[0] == 0:
            labels[:] = 0
            matched_gt_indices = torch.full(
                (num_anchors,), -1,
                dtype=torch.int64,
                device=anchors.device
            )
            return labels, matched_gt_indices

        # Anchor가 없는 경우
        if num_anchors == 0:
            return labels, torch.empty(
                (0,), dtype=torch.int64,
                device=anchors.device
            )

        # [N, M]
        iou_matrix = box_iou(anchors, gt_boxes)

        # [N], [N]
        max_iou, matched_gt_indices = iou_matrix.max(dim=1)

        labels[max_iou >= self.pos_iou] = 1
        labels[max_iou < self.neg_iou] = 0

        # GT별 최대 IoU Anchor 찾기
        # [M]
        best_anchor_indices = iou_matrix.argmax(dim=0)

        # 0.7 미만이어도 GT별 최고 IoU Anchor는 Positive
        labels[best_anchor_indices] = 1

        return labels, matched_gt_indices