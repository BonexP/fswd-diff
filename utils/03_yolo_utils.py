"""
03_yolo_utils.py

功能：
读取YOLO标注，返回类别 + bbox

YOLO格式：
class x_center y_center width height （归一化）

输出：
[x1, y1, x2, y2]

作者：你的论文项目
"""
import os


def yolo_to_bbox(label_path, img_width, img_height):
    bboxes = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, x, y, w, h = map(float, line.strip().split())

        x1 = (x - w / 2) * img_width
        y1 = (y - h / 2) * img_height
        x2 = (x + w / 2) * img_width
        y2 = (y + h / 2) * img_height

        bboxes.append([x1, y1, x2, y2])

    return bboxes


def load_yolo_labels(label_path, img_width, img_height):
    results = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, x, y, w, h = map(float, line.strip().split())

        x1 = (x - w / 2) * img_width
        y1 = (y - h / 2) * img_height
        x2 = (x + w / 2) * img_width
        y2 = (y + h / 2) * img_height

        results.append({
            "class": int(cls),
            "bbox": [x1, y1, x2, y2]
        })

    return results


def save_yolo_label(output_path, bbox, img_width, img_height, cls):
    x1, y1, x2, y2 = bbox

    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    # Ensure parent directory exists before writing label file.
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"{cls} {x_center} {y_center} {w} {h}\n")