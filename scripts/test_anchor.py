
import torch
from src.Faster_RCNN.anchor import AnchorGenerator


if __name__ == "__main__":

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    feature_map = torch.randn(1, 512, 20, 31, device=device)

    # ----------------------------------
    # TEST 1. Anchor Shape 검증
    # ----------------------------------
    generator = AnchorGenerator(
        scale=[64, 128, 256],
        ratio=[0.5, 1.0, 2.0],
        stride=16
    )

    anchors = generator.generate_anchors(feature_map)

    print("=== TEST 1: Shape 검증 ===")
    print(f"Anchor Shape : {anchors.shape}")

    assert anchors.shape == (5580, 4)
    print("Shape 검증 성공!\n")

    # ----------------------------------
    # TEST 2. Anchor 좌표 검증
    # ----------------------------------
    generator = AnchorGenerator(
        scale=[64],
        ratio=[1.0],
        stride=16
    )

    base_anchors = generator.generate_base_anchors()
    anchors = generator.generate_anchors(feature_map)

    expected_base = torch.tensor(
        [-32., -32., 32., 32.]
    )

    expected_anchor = torch.tensor(
        [-24., -24., 40., 40.],
        device=anchors.device
    )

    print("=== TEST 2: 좌표 검증 ===")
    print(f"Base Anchor : {base_anchors[0]}")
    print(f"First Anchor : {anchors[0]}")

    assert torch.allclose(base_anchors[0], expected_base)
    assert torch.allclose(anchors[0], expected_anchor)
    assert anchors.shape == (620, 4)

    print("좌표 검증 성공!\n")

    # ----------------------------------
    # TEST 3. Device 검증
    # ----------------------------------
    print("=== TEST 3: Device 검증 ===")
    print(f"Feature Map Device : {feature_map.device}")
    print(f"Anchor Device : {anchors.device}")

    assert anchors.device == feature_map.device
    print("Device 검증 성공!\n")

    print("모든 Anchor Generator 테스트 통과!")
