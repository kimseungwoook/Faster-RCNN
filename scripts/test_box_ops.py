from src.Faster_RCNN.box_ops import box_iou, encode_boxes
import torch
import math

if __name__ == "__main__":
    # ===== 두 box가 일부 겹치는 경우 =====
    box1 = torch.tensor([[0., 0., 100., 100.]])
    box2 = torch.tensor([[50., 50., 150., 150.]])

    iou = box_iou(box1, box2)
    expected_iou = torch.tensor([[0.142857]])

    print("=== TEST 1: 두 box가 일부 겹치는 상황 검증 ===")
    print(f"IoU : {iou}")
    print(f"IoU : {iou.shape}")
    assert torch.allclose(iou, expected_iou)
    print("Test1 통과")

    # ===== 두 box가 겹치지 않는 경우 =====
    box3 = torch.tensor([[0., 0., 100., 100.]])
    box4 = torch.tensor([[150., 150., 250., 250.]])

    iou = box_iou(box3, box4)
    expected_iou = torch.tensor([[0.]])

    print("=== TEST 2: 두 box가 겹치지 않는 상황 검증 ===")
    print(f"IoU : {iou}")
    print(f"IoU : {iou.shape}")
    assert torch.allclose(iou, expected_iou)
    print("Test2 통과")

    # ===== 두 box가 완전히 일치하는 경우 =====
    box5 = torch.tensor([[0., 0., 100., 100.]])
    box6 = torch.tensor([[0., 0., 100., 100.]])

    iou = box_iou(box5, box6)
    expected_iou = torch.tensor([[1.]])
    print("=== TEST 3: 두 box가 완전히 겹치는 상황 검증 ===")
    print(f"IoU : {iou}")
    print(f"IoU : {iou.shape}")
    assert torch.allclose(iou, expected_iou)
    print("Test3 통과")


    # ===== 여러 box 비교하는 경우 =====
    box7 = torch.tensor([
        [0., 0., 100., 100.],
        [50., 50., 150., 150.]
    ])
    box8 = torch.tensor([
        [0., 0., 100., 100.],
        [200., 200., 300., 300.]
    ])

    iou = box_iou(box7, box8)
    expected_iou = torch.tensor([
        [1.0, 0.0],
        [1/7, 0.0]
    ])
    print("=== TEST 4: 여러 box가 비교하는 상황 검증 ===")
    print(f"IoU : {iou}")
    print(f"IoU : {iou.shape}")
    assert iou.shape == (2, 2)
    assert torch.allclose(iou, expected_iou)
    print("Test4 통과")


    # ===== Encoding Test =====
    anchors = torch.tensor([
        [0., 0., 100., 100.]
    ])
    gt_boxes = torch.tensor([
        [10., 20., 130., 120.]
    ])

    # 반환 값 = [N, 4]
    encoded = encode_boxes(anchors, gt_boxes)
    expected_tensor = torch.tensor([
        [0.2000, 0.2000, math.log(1.2), 0.0000]
    ])
    print(f"Encode Boxes : {encoded}")
    print(f"Encode Shape : {encoded.shape}")
    assert encoded.shape == (1, 4)
    assert torch.allclose(encoded, expected_tensor)
    print("Encoding Test 통과")