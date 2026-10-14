# Faster R-CNN: Object Detection from Fundamentals to Practice

## 1. Project Overview

본 프로젝트는 객체 탐지(Object Detection) 알고리즘의 내부 동작 원리를 이해하고, Python 및 PyTorch 기반의 직접 구현 능력을 향상시키기 위해 진행하는 개인 학습 프로젝트입니다.

Faster R-CNN의 전체 구조를 처음부터 구현하는 대신, 객체 탐지의 핵심 구성 요소인 **Anchor Generator, IoU 계산, Anchor Matcher**를 직접 구현하여 알고리즘의 원리를 학습합니다.

또한 PASCAL VOC 2007 데이터셋을 직접 처리하여 이미지 및 Bounding Box 정보를 모델 입력에 적합한 Tensor 형태로 변환하는 과정을 구현합니다.

**&#x20;Faster R-CNN 모델**을 활용해 객체이후 **Torchvision에서 제공하는** 탐지 모델의 학습(Training), 추론(Inference), 성능 평가(Evaluation) 및 성능 개선 실험을 수행할 계획입니다.

최종적으로 객체 탐지 알고리즘에 대한 이해와 실제 모델 개발 경험을 바탕으로 Computer Vision 및 Robot Vision 분야에서 활용할 수 있는 소프트웨어 구현 역량을 확보하는 것을 목표로 합니다.

## 2. Development Environment

본 프로젝트는 이전 Penn-Fudan 객체 탐지 학습 프로젝트에서 생성한 Conda 가상환경 `pennfudan`을 재사용하여 개발하였습니다.

### Development Tools

- **OS:** Windows
- **IDE:** PyCharm
- **Environment:** Conda (`pennfudan`)
- **Language:** Python 3.11.16
- **GPU Acceleration:** CUDA Available

### Main Libraries

| Library | Version | Purpose |
|---|---|---|
| PyTorch | 2.11.0+cu128 | Tensor 연산 및 딥러닝 모델 구현 |
| Torchvision | 0.26.0+cu128 | Faster R-CNN 모델 학습 및 추론 (예정) |
| OpenCV | 5.0.0 | 이미지 로딩, 색상 변환 및 시각화 |
| NumPy | 2.4.6 | 이미지 배열 및 데이터 처리 |

### Dataset

- **PASCAL VOC 2007**
- 20개의 객체 클래스를 포함한 객체 탐지 데이터셋
- JPEG 이미지와 XML Annotation을 활용하여 Bounding Box 및 Class Label 처리

**Note:** 기존 Conda 환경을 재사용하므로, 설치된 전체 패키지가 아닌 프로젝트에 필요한 주요 라이브러리만 기록하였습니다.

## 3. Project Structure

프로젝트는 객체 탐지 핵심 알고리즘을 구현하는 `src/`와 각 기능을 테스트하는 `scripts/`로 구분하여 구성하였습니다.

### Source Code

| File | Description |
|---|---|
| `anchor.py` | Feature Map의 각 위치에서 Scale과 Ratio를 이용한 Anchor Box 생성 |
| `box_ops.py` | 두 Bounding Box 사이의 IoU 계산 |
| `matcher.py` | Anchor와 GT Box를 비교하여 Positive, Negative, Ignore Label 할당 |
| `dataset.py` | PASCAL VOC 이미지 및 XML Annotation을 PyTorch Tensor로 변환 |

### Test Scripts

| File                      | Description |
|---------------------------|---|
| `01_test_anchor.py`       | Anchor 생성 및 Valid Anchor 검증 |
| `02_test_matcher.py`      | Anchor Matching 결과 및 IoU 검증 |
| `03_test_dataloader.py`   | DataLoader의 Batch 구성 및 Tensor Shape 확인 |
| `04_test_visualize_gt.py` | 이미지에 GT Bounding Box를 그려 시각화 |

### Dataset Directory

`Data/` 폴더에는 PASCAL VOC 2007 데이터셋이 저장되어 있습니다.

- `JPEGImages/`: 원본 이미지
- `Annotations/`: 객체의 Bounding Box와 Class Label을 포함한 XML 파일
- `ImageSets/Main/`: Train/Validation 이미지 ID 목록

**Note:** 데이터셋은 `.gitignore`를 통해 GitHub 업로드 대상에서 제외하였습니다.