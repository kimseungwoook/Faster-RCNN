
from src.Faster_RCNN.matcher import AnchorMatcher
from src.Faster_RCNN.box_ops import box_iou
import torch


if __name__ == "__main__":
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    matcher = AnchorMatcher(pos_iou=0.7, neg_iou=0.3)

    # =====================================
    # TEST 1. IoU Threshold 분류
    # =====================================
    anchors = torch.tensor([
        [0., 0., 100., 100.],   # IoU = 1.0
        [0., 0., 50., 100.],    # IoU = 0.5
        [0., 0., 20., 100.]     # IoU = 0.2
    ], device=device)

    gt_boxes = torch.tensor([
        [0., 0., 100., 100.]
    ], device=device)

    labels, matched_gt_indices = matcher.match(anchors, gt_boxes)

    expected_labels = torch.tensor([1, -1, 0], device=device)
    expected_indices = torch.tensor([0, 0, 0], device=device)

    print("=== TEST 1: IoU Threshold 분류 ===")
    print(f"Labels : {labels}")
    print(f"Matched GT Indices : {matched_gt_indices}")

    assert torch.equal(labels, expected_labels)
    assert torch.equal(matched_gt_indices, expected_indices)
    print("TEST 1 통과!\n")

    # =====================================
    # TEST 2. GT별 최대 IoU Positive 보장
    # =====================================
    anchors = torch.tensor([
        [0., 0., 60., 100.],      # IoU = 0.6
        [0., 0., 40., 100.],      # IoU = 0.4
        [200., 200., 300., 300.] # IoU = 0.0
    ], device=device)

    gt_boxes = torch.tensor([
        [0., 0., 100., 100.]
    ], device=device)

    labels, matched_gt_indices = matcher.match(anchors, gt_boxes)

    expected_labels = torch.tensor([1, -1, 0], device=device)

    print("=== TEST 2: GT별 최대 IoU Positive 보장 ===")
    print(f"IoU Matrix :\n{box_iou(anchors, gt_boxes)}")
    print(f"Labels : {labels}")

    assert torch.equal(labels, expected_labels)
    print("TEST 2 통과!\n")

    # =====================================
    # TEST 3. 여러 GT Box 매칭
    # =====================================
    anchors = torch.tensor([
        [0., 0., 100., 100.],
        [200., 200., 260., 300.],
        [0., 0., 50., 100.],
        [400., 400., 500., 500.]
    ], device=device)

    gt_boxes = torch.tensor([
        [0., 0., 100., 100.],
        [200., 200., 300., 300.]
    ], device=device)

    labels, matched_gt_indices = matcher.match(anchors, gt_boxes)

    expected_labels = torch.tensor([1, 1, -1, 0], device=device)
    expected_indices = torch.tensor([0, 1, 0, 0], device=device)

    print("=== TEST 3: 여러 GT Box 매칭 ===")
    print(f"IoU Matrix :\n{box_iou(anchors, gt_boxes)}")
    print(f"Labels : {labels}")
    print(f"Matched GT Indices : {matched_gt_indices}")

    assert labels.shape == (4,)
    assert matched_gt_indices.shape == (4,)
    assert torch.equal(labels, expected_labels)
    assert torch.equal(matched_gt_indices, expected_indices)
    print("TEST 3 통과!\n")

    # =====================================
    # TEST 4. GT Box가 없는 경우
    # =====================================
    anchors = torch.tensor([
        [0., 0., 100., 100.],
        [200., 200., 300., 300.]
    ], device=device)

    gt_boxes = torch.empty((0, 4), device=device)

    labels, matched_gt_indices = matcher.match(anchors, gt_boxes)

    expected_labels = torch.tensor([0, 0], device=device)
    expected_indices = torch.tensor([-1, -1], device=device)

    print("=== TEST 4: GT Box 없음 ===")
    print(f"Labels : {labels}")
    print(f"Matched GT Indices : {matched_gt_indices}")

    assert torch.equal(labels, expected_labels)
    assert torch.equal(matched_gt_indices, expected_indices)
    print("TEST 4 통과!\n")

    # =====================================
    # TEST 5. Anchor가 없는 경우
    # =====================================
    anchors = torch.empty((0, 4), device=device)

    gt_boxes = torch.tensor([
        [0., 0., 100., 100.]
    ], device=device)

    labels, matched_gt_indices = matcher.match(anchors, gt_boxes)

    print("=== TEST 5: Anchor 없음 ===")
    print(f"Labels Shape : {labels.shape}")
    print(f"Matched GT Indices Shape : {matched_gt_indices.shape}")

    assert labels.shape == (0,)
    assert matched_gt_indices.shape == (0,)
    print("TEST 5 통과!\n")

    print("모든 AnchorMatcher 테스트 통과!")
