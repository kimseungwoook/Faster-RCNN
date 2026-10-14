from torch.utils.data import DataLoader
from src.Faster_RCNN.dataset import VOCDataset

def collate_fn(batch):
     images, targets = zip(*batch)
     return list(images), list(targets)

if __name__ == "__main__":
    batch_size = 2
    root_dir = r"C:\Users\admin\source\Faster R-CNN\Data\VOCdevkit\VOC2007"

    dataset = VOCDataset(root_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)

    for images, targets in dataloader:
        # Batch 크기 검증
        assert len(images) == batch_size

        # Image와 Target 개수가 동일한지 검증
        assert len(images) == len(targets)

        print(f"Batch 이미지 개수: {len(images)}")

        for idx, (image, target) in enumerate(zip(images, targets)):
            print(f"Image {idx} shape: {image.shape}")
            print(f"Boxes {idx} shape: {target['boxes'].shape}")
            print(f"Labels {idx} shape: {target['labels'].shape}")

        break