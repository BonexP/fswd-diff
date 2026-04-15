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

load_yolo_labels = importlib.import_module("utils.03_yolo_utils").load_yolo_labels
save_yolo_label=importlib.import_module("utils.03_yolo_utils").save_yolo_label
image_utils = importlib.import_module("utils.04_image_utils")
load_image = image_utils.load_image
save_image = image_utils.save_image
generate_defect = importlib.import_module("core.05_inpaint").generate_defect


def batch_generate(
    pipe,
    image_dir,
    label_dir,
    output_dir,
    target_class,   # ⭐ 新增：目标类别
    prompt
):
    images_out_dir = os.path.join(output_dir, "images")
    masks_out_dir = os.path.join(output_dir, "masks")
    labels_out_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(masks_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if not filename.endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace(".jpg", ".txt"))

        image = load_image(image_path)
        w, h = image.size

        labels = load_yolo_labels(label_path, w, h)

        for i, item in enumerate(labels):

            cls = item["class"]
            bbox = item["bbox"]

            result_path = f"{output_dir}/{filename}_gen_{i}.png"
            mask_path = f"{output_dir}/{filename}_mask_{i}.png"

            # ⭐ 只增强目标类别
            if cls != target_class:
                continue

            # Skip if both outputs already exist for this bbox index.
            if os.path.exists(result_path) and os.path.exists(mask_path):
                continue
            result, mask = generate_defect(pipe, image, bbox, prompt)

            # 保存图像
            img_out_path = f"{output_dir}/images/{filename}_gen_{i}.jpg"
            mask_out_path = f"{output_dir}/masks/{filename}_mask_{i}.png"
            label_out_path = f"{output_dir}/labels/{filename}_gen_{i}.txt"

            save_image(result, img_out_path)
            save_image(mask, mask_out_path)

            # ⭐ 保存YOLO标签
            save_yolo_label(label_out_path, bbox, w, h, cls)