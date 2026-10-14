from src.Faster_RCNN.anchor import AnchorGenerator

if __name__ == "__main__":
    scale = [64, 128, 256]
    ratio = [0.5, 1.0, 2.0]
    subsampling_ratio = 16

    feature_h = 50
    feature_w = 50
    image_h = 800
    image_w = 800

    generator = AnchorGenerator(scale, ratio, subsampling_ratio)
    anchors, valid_flags = generator.generate(feature_h, feature_w, image_h, image_w)

    # 생성된 Anchor 개수 검증
    expected_count = feature_h * feature_w * len(scale) * len(ratio)

    assert len(anchors) == expected_count

    # Anchor와 valid_flags의 개수가 동일한지 검증
    assert len(anchors) == len(valid_flags)

    print(f"anchor 개수 : {len(anchors)}")
    print(anchors[0])
    print(anchors[1])
    print(anchors[2])

    if len(anchors) == sum(valid_flags):
        print(f"이미지 내에 앵커 박스가 모두 들어옵니다! 개수 : {len(anchors)}")
    else:
        print(f"이미지 내에 앵커 박스가 모두 들어오지 않습니다! \n"
              f"valid_flag 개수 : {sum(valid_flags)}\n"
              f"anchor 개수 : {len(anchors)}")