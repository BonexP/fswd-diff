"""
06_batch_generate.py

功能：
批量生成缺陷图像

流程：
遍历数据集 → 读取bbox → 生成 → 保存

作者：你的论文项目
"""

import os
import importlib

yolo_to_bbox = importlib.import_module("utils.03_yolo_utils").yolo_to_bbox
image_utils = importlib.import_module("utils.04_image_utils")
load_image = image_utils.load_image
save_image = image_utils.save_image
generate_defect = importlib.import_module("core.05_inpaint").generate_defect


def batch_generate(pipe, image_dir, label_dir, output_dir, prompt):

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        image = load_image(image_path)
        w, h = image.size

        bboxes = yolo_to_bbox(label_path, w, h)

        for i, bbox in enumerate(bboxes):
            result_path = f"{output_dir}/{filename}_gen_{i}.png"
            mask_path = f"{output_dir}/{filename}_mask_{i}.png"

            # Skip if both outputs already exist for this bbox index.
            if os.path.exists(result_path) and os.path.exists(mask_path):
                continue

            result, mask = generate_defect(pipe, image, bbox, prompt)

            save_image(result, result_path)
            save_image(mask, mask_path)