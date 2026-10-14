# GT와 Anchor를 비교하여 Positive/Negative/Ignore 할당
from .box_ops import calculate_iou

class AnchorMatcher:
    def __init__(self, positive_threshold, negative_threshold):
        self.pos_threshold = positive_threshold
        self.neg_threshold = negative_threshold

    def match(self, anchors, gt_boxes, valid_flags):
        """
        Anchor Box와 GT Box의 IoU를 계산하여 label을 할당

        Args :
            anchors : Anchor box의 좌표 리스트 [xmin, ymin, xmax, ymax]
            gt_boxes : GT Bounding Box 좌표 리스트
            valid_flags : 각 Anchor의 이미지 경계 내의 유효 여부
        Return :
            labels : Anchor별 Label (1:positive, 0:negative, -1:Ignore)
            max_IoUs : 각 Anchor의 최대 IoU
            matched_gt_indices : 각 앵커에 매칭된 GT 인덱스 (-1 : 매칭 x)
        """
        labels = []
        max_IoUs = []
        matched_gt_indices = []

        for anchor_idx, (valid, anchor_box) in enumerate(zip(valid_flags, anchors)):
            # 각 Anchor box의 최대 IoU 계산 -> max_IoU의 값에 따라 positive, negative, Ignore 배정
            max_iou = 0
            matched_gt_idx = -1

            if valid:
                for gt_idx, gt_box in enumerate(gt_boxes):
                    IoU = calculate_iou(anchor_box, gt_box)

                    if IoU > max_iou:
                        max_iou = IoU
                        matched_gt_idx = gt_idx

                # 1 = positive, 0 = negative, -1 = Ignore
                if max_iou >= self.pos_threshold:
                    labels.append(1)
                elif max_iou <= self.neg_threshold:
                    labels.append(0)
                else:
                    labels.append(-1)
            else:
                labels.append(-1)

            max_IoUs.append(max_iou)
            matched_gt_indices.append(matched_gt_idx)

        for gt_idx, gt_box in enumerate(gt_boxes):
            # 각 GT에서 최대 IoU를 가진 유효한 Anchor들을 Positive로 강제 지정
            # 단 모든 IoU가 0인 경우에는 강제 지정 x
            best_iou = -1.0
            best_anchor_indices = []

            for anchor_idx, (anchor, valid) in enumerate(zip(anchors, valid_flags)):
                if not valid:
                    continue

                iou = calculate_iou(anchor, gt_box)

                if iou > best_iou:
                    best_iou = iou
                    best_anchor_indices = [anchor_idx]
                elif iou == best_iou:
                    best_anchor_indices.append(anchor_idx)

            if best_iou > 0:
                for best_idx in best_anchor_indices:
                    labels[best_idx] = 1

        return labels, max_IoUs, matched_gt_indices