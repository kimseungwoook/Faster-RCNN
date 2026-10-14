from src.Faster_RCNN.matcher import AnchorMatcher

if __name__ == "__main__":
    anchors = [
        [0, 0, 100, 100],
        [200, 200, 300, 300]
    ]

    gt_boxes = [
        [0, 0, 80, 80]
    ]

    valid_flags = [True, True]
    matcher = AnchorMatcher(
        positive_threshold=0.7,
        negative_threshold=0.3
    )

    labels, max_ious, matched_gt_indices = matcher.match(
        anchors, gt_boxes, valid_flags
    )

    print("===== Anchor Matching 결과 =====")

    for i in range(len(anchors)):
        print(
            f"Anchor {i}: "
            f"Label={labels[i]}, "
            f"Max IoU={max_ious[i]:.4f}, "
            f"Matched GT={matched_gt_indices[i]}"
        )

    print("\n===== 길이 검증 =====")

    print(f"Anchors: {len(anchors)}")
    print(f"Labels: {len(labels)}")
    print(f"Max IoUs: {len(max_ious)}")
    print(f"Matched GT indices: {len(matched_gt_indices)}")

    # Anchor 개수와 반환 결과의 길이 검증
    assert len(anchors) == len(labels)
    assert len(anchors) == len(max_ious)
    assert len(anchors) == len(matched_gt_indices)

    # 각 Anchor의 Label 검증
    assert labels == [1, 0]

    # IoU 검증
    assert abs(max_ious[0] - 0.64) < 1e-6
    assert max_ious[1] == 0.0

    # GT 매칭 인덱스 검증
    assert matched_gt_indices == [0, -1]

    print("모든 테스트 통과!")