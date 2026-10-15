from src.Faster_RCNN.dataset import VOCDataset
from src.Faster_RCNN.backbone import VGG16Backbone
import torch
from torchvision.transforms.functional import normalize

if __name__ == "__main__":
    root_dir = r"C:\Users\admin\source\Faster R-CNN\Data\VOCdevkit\VOC2007"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_dataset = VOCDataset(root_dir, "train")
    image, target = train_dataset[0]
    image = normalize(image, mean, std).unsqueeze(0)

    print(f"이미지 Shape : {image.shape}")
    print(f"Boxes Shape : {target['boxes'].shape}")
    print(f"Labels Shape : {target['labels'].shape}")

    backbone = VGG16Backbone()
    backbone.eval()

    with torch.no_grad():
        output = backbone(image)

    print(f"vgg16 backbone output shape : {output.shape}")
