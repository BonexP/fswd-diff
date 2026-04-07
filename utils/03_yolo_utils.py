"""
03_yolo_utils.py

功能：
读取YOLO格式标注文件，并转换为bbox

YOLO格式：
class x_center y_center width height （归一化）

输出：
[x1, y1, x2, y2]

作者：你的论文项目
"""

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