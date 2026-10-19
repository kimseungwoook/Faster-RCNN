
import torch


class BalancedPositiveNegativeSampler:
    def __init__(self, batch_size_per_image=256,
                 positive_fraction=0.5):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def sample(self, labels):
        """
        Args:
            labels: [N]
                1  : Positive
                0  : Negative
                -1 : Ignore

        Returns:
            pos_mask: [N] (bool)
            neg_mask: [N] (bool)
        """

        # 1. Positive / Negative Anchor 인덱스
        positive_indices = torch.where(labels == 1)[0]
        negative_indices = torch.where(labels == 0)[0]

        # 2. Sampling 개수 결정
        num_pos = min(
            int(self.batch_size_per_image * self.positive_fraction),
            positive_indices.numel()
        )

        num_neg = min(
            self.batch_size_per_image - num_pos,
            negative_indices.numel()
        )

        # 3. 무작위로 Anchor 선택
        pos_perm = torch.randperm(
            positive_indices.numel(),
            device=labels.device
        )[:num_pos]

        neg_perm = torch.randperm(
            negative_indices.numel(),
            device=labels.device
        )[:num_neg]

        selected_pos = positive_indices[pos_perm]
        selected_neg = negative_indices[neg_perm]

        # 4. Boolean Mask 생성
        pos_mask = torch.zeros(
            labels.shape[0],
            dtype=torch.bool,
            device=labels.device
        )

        neg_mask = torch.zeros(
            labels.shape[0],
            dtype=torch.bool,
            device=labels.device
        )

        # 5. 선택된 Anchor만 True로 설정
        pos_mask[selected_pos] = True
        neg_mask[selected_neg] = True

        return pos_mask, neg_mask
