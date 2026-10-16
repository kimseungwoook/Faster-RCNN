import torch
import math

class AnchorGenerator:
    def __init__(self, scale, ratio, stride):
        self.scale = scale
        self.ratio = ratio
        self.stride = stride

    def generate_base_anchors(self):
        """
        0,0 을 기준으로 Anchor box들의 xmin, ymin, xmax, ymax의 값을 구함

        Returns:
            [xmin, ymin, xmax, ymax]
        """
        base_anchors = []

        for scale in self.scale:
            for ratio in self.ratio:
                in_w = scale / math.sqrt(ratio)
                in_h = scale * math.sqrt(ratio)

                xmin = -in_w / 2
                ymin = -in_h / 2
                xmax = +in_w / 2
                ymax = +in_h / 2

                base_anchors.append([xmin, ymin, xmax, ymax])

        base_anchors_tensor = torch.tensor(base_anchors, dtype=torch.float32)

        return base_anchors_tensor

    def generate_anchors(self, feature_map):
        feature_map_h = feature_map.shape[2]
        feature_map_w = feature_map.shape[3]

        x = (torch.arange(feature_map_w, device=feature_map.device) + 0.5) * self.stride
        y = (torch.arange(feature_map_h, device=feature_map.device) + 0.5) * self.stride

        # grid_x = 20, 31, grid_y = 20, 31
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        # 중심 좌표
        center = torch.stack([grid_x, grid_y, grid_x, grid_y], dim=2)
        center = center.reshape(-1, 4)

        base_anchors = self.generate_base_anchors().to(device=feature_map.device)

        center = center.unsqueeze(1)
        base_anchors = base_anchors.unsqueeze(0)

        broadcasting_anchors = center + base_anchors
        anchors = broadcasting_anchors.reshape(-1, 4)

        return anchors
