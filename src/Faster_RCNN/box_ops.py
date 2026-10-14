def calculate_iou(box1, box2):
    """
    두 Bounding box 사이의 IoU를 계산

    Args:
        box1 : 첫 번째 Bounding box [xmin, ymin, xmax, ymax]
        box2 : 두 번째 Bounding box [xmin, ymin, xmax, ymax]

    Returns:
        IoU : 두 Bounding box의 Intersection Over Union 계산
              (0.0 ~ 1.0)
    """
    x_min_b1, y_min_b1, x_max_b1, y_max_b1 = box1
    x_min_b2, y_min_b2, x_max_b2, y_max_b2 = box2

    x_min_inter = max(x_min_b1, x_min_b2)
    y_min_inter = max(y_min_b1, y_min_b2)
    x_max_inter = min(x_max_b1, x_max_b2)
    y_max_inter = min(y_max_b1, y_max_b2)

    if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
        return 0.0

    intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

    union_area = ((x_max_b1 - x_min_b1) * (y_max_b1 - y_min_b1)) + ((x_max_b2 - x_min_b2) * (y_max_b2 - y_min_b2)) - intersection_area

    IoU = intersection_area / union_area

    return IoU