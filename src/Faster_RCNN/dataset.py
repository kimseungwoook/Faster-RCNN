from pathlib import Path
from torch.utils.data import Dataset
import cv2
import xml.etree.ElementTree as ET
import torch

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class VOCDataset(Dataset):
    def __init__(self, root_dir):
        """
            PASCAL VOC 데이터셋의 경로와 이미지 ID를 초기화

            Args:
                root_dir : VOC2007의 최상위 경로
        """
        self.root_dir = Path(root_dir)
        self.jpeg_dir = self.root_dir / "JPEGImages"
        self.annotation_dir = self.root_dir / "Annotations"
        self.imageSets_main = self.root_dir / "ImageSets/Main"
        self.image_ids = []
        self.class_to_idx = {
            name : idx + 1
            for idx, name in enumerate(VOC_CLASSES)
        }

        if not self.jpeg_dir.is_dir():
            raise FileNotFoundError(f"JPEGImages not found in {self.jpeg_dir}")
        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(f"Annotations not found in {self.annotation_dir}")
        if not self.imageSets_main.is_dir():
            raise FileNotFoundError(f"ImageSets not found in {self.imageSets_main}")

        train_path = self.imageSets_main / "train.txt"
        with open(train_path, "r") as f:
            for line in f:
                self.image_ids.append(line.strip())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        이미지 한 장과 이미지의 객체 정보를 불러오는 함수

        Args:
            idx : 가져올 데이터의 인덱스
        Returns:
            image_tensor : 이미지를 tensor 형태로 변환하여 반환 (3, H, W)
            target : { boxes: 바운딩 박스의 좌표 텐서 형태로 변환,
                       labels: 해당 객체가 VOC_CLASSES 객체 번호를 텐서 형태로 변환 }
        """
        image_id = self.image_ids[idx]
        image_path = self.jpeg_dir / f"{image_id}.jpg"
        annotation_path = self.annotation_dir / f"{image_id}.xml"

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없다. {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 텐서 형태 변환 및 float32, 0.0 ~ 1.0
        image_tensor = torch.from_numpy(image_rgb)
        image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0

        boxes = []
        class_names = []
        labels = []

        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            class_names.append(name)

            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

        for cls_name in class_names:
            labels.append(self.class_to_idx[cls_name])


        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes" : boxes_tensor,
            "labels" : labels_tensor
        }

        return image_tensor, target