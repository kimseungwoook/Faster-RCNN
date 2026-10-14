import math

class AnchorGenerator:
    def __init__(self, scales, ratios, subsampling_ratio):
        """
        Args:
            scales : Anchor box의 기준 크기 (sqrt(width * height))
            ratios : Anchor box의 높이/너비 비율 (height / width)
            subsampling_ratio : Feature map의 stride
        """
        self.scales = scales
        self.ratios = ratios
        self.subsampling_ratio = subsampling_ratio

    def generate(self, feature_h, feature_w, image_h, image_w):
        """
        Feature map의 각 위치에서 Anchor box를 생성

        Args:
            feature_h : Feature map의 높이
            feature_w : Feature map의 너비
            image_h : Image의 높이
            image_w : Image의 너비

        Returns:
            anchors : 생성된 Anchor box의 좌표 리스트
                        [xmin, ymin, xmax, yamx]
            valid_flags : Anchor Box가 이미지 경계 내에 포함되지 않는 Anchor box = False
                          아니면 True
        """
        anchors = []
        valid_flags = []

        for y in range(feature_h):
            for x in range(feature_w):
                # Feature map의 좌표를 원본 이미지의 좌표계로 변환
                center_x = (x + 0.5) * self.subsampling_ratio
                center_y = (y + 0.5) * self.subsampling_ratio

                for scale in self.scales:
                    for ratio in self.ratios:
                        # scale을 기준으로 면적을 유지하면서
                        # ratio에 따라 Anchor box 가로, 세로 결정
                        anchor_w = scale / math.sqrt(ratio)
                        anchor_h = ratio * anchor_w

                        # Anchor 중심좌표를 기준으로 네 경계 좌표 계산
                        xmin = center_x - (anchor_w * 0.5)
                        ymin = center_y - (anchor_h * 0.5)
                        xmax = center_x + (anchor_w * 0.5)
                        ymax = center_y + (anchor_h * 0.5)

                        if xmin >= 0 and xmax <= image_w and ymin >= 0 and ymax <= image_h:
                            valid_flags.append(True)
                        else:
                            valid_flags.append(False)
                        anchors.append([xmin, ymin, xmax, ymax])

        return anchors, valid_flags
