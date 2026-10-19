
import torch
from src.Faster_RCNN.sampler import BalancedPositiveNegativeSampler


if __name__ == "__main__":
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # TEST 1. 기본 Sampling
    sampler = BalancedPositiveNegativeSampler(
        batch_size_per_image=4,
        positive_fraction=0.5
    )

    labels = torch.tensor(
        [1, 1, 0, 0, 0, -1],
        device=device
    )

    pos_mask, neg_mask = sampler.sample(labels)

    assert pos_mask.sum().item() == 2
    assert neg_mask.sum().item() == 2
    assert torch.all(labels[pos_mask] == 1)
    assert torch.all(labels[neg_mask] == 0)
    assert not (pos_mask & neg_mask).any().item()

    print("TEST 1: 기본 Sampling 통과")

    # TEST 2. Positive Anchor가 부족한 경우
    sampler = BalancedPositiveNegativeSampler(
        batch_size_per_image=256,
        positive_fraction=0.5
    )

    labels = torch.tensor(
        [1] * 50 + [0] * 500 + [-1] * 20,
        device=device
    )

    pos_mask, neg_mask = sampler.sample(labels)

    assert pos_mask.sum().item() == 50
    assert neg_mask.sum().item() == 206

    print("TEST 2: Positive 부족 상황 통과")

    # TEST 3. Positive / Negative 충분한 경우
    labels = torch.tensor(
        [1] * 200 + [0] * 300,
        device=device
    )

    pos_mask, neg_mask = sampler.sample(labels)

    assert pos_mask.sum().item() == 128
    assert neg_mask.sum().item() == 128

    print("TEST 3: 256개 Sampling 통과")

    # TEST 4. 모든 Anchor가 Ignore인 경우
    labels = torch.tensor([-1, -1, -1], device=device)

    pos_mask, neg_mask = sampler.sample(labels)

    assert pos_mask.sum().item() == 0
    assert neg_mask.sum().item() == 0

    print("TEST 4: Ignore 처리 통과")

    print("\n모든 Sampler 테스트 통과!")
