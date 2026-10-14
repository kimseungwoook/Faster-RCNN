import cv2
from src.Faster_RCNN.dataset import VOCDataset

if __name__ == "__main__":
    root_dir = r"C:\Users\admin\source\Faster R-CNN\Data\VOCdevkit\VOC2007"
    dataset = VOCDataset(root_dir)

    image_tensor, targets = dataset[100]

    # 현재 이미지는 torch.tensor 형태 -> numpy로 변환
    image = image_tensor.permute(1, 2, 0).cpu().numpy()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_bgr = (image_bgr * 255).astype("uint8")

    boxes = targets.get("boxes", [])

    for xmin, ymin, xmax, ymax in boxes:
        xmin, ymin, xmax, ymax = map(int, (xmin, ymin, xmax, ymax))
        cv2.rectangle(image_bgr, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)

    cv2.imshow("image", image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()