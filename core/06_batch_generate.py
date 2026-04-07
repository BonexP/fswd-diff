"""
06_batch_generate.py

功能：
批量生成缺陷图像

流程：
遍历数据集 → 读取bbox → 生成 → 保存

作者：你的论文项目
"""

import os
from utils.03_yolo_utils import yolo_to_bbox
from utils.04_image_utils import load_image, save_image
from core.05_inpaint import generate_defect


def batch_generate(pipe, image_dir, label_dir, output_dir, prompt):

    for filename in os.listdir(image_dir):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        image = load_image(image_path)
        w, h = image.size

        bboxes = yolo_to_bbox(label_path, w, h)

        for i, bbox in enumerate(bboxes):
            result, mask = generate_defect(pipe, image, bbox, prompt)

            save_image(result, f"{output_dir}/{filename}_gen_{i}.png")
            save_image(mask, f"{output_dir}/{filename}_mask_{i}.png")