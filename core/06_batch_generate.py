"""
06_batch_generate.py（升级版）

功能：
批量生成缺陷 + 自动更新bbox + 完整YOLO标签

作者：你的论文项目（增强版）
"""

import os
import importlib

load_yolo_labels = importlib.import_module("utils.03_yolo_utils").load_yolo_labels
save_yolo_labels = importlib.import_module("utils.03_yolo_utils").save_yolo_labels
image_utils = importlib.import_module("utils.04_image_utils")
load_image = image_utils.load_image
save_image = image_utils.save_image
mask_to_bbox = importlib.import_module("utils.05_mask_to_bbox").mask_to_bbox
generate_defect = importlib.import_module("core.05_inpaint").generate_defect


def batch_generate(
    pipe,
    image_dir,
    label_dir,
    output_dir,
    target_class,
    prompt
):
    images_out_dir = os.path.join(output_dir, "images")
    labels_out_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_out_dir, exist_ok=True)
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
            if cls != target_class:
                continue

            img_out_path = os.path.join(images_out_dir, f"{filename}_gen_{i}.jpg")
            label_out_path = os.path.join(labels_out_dir, f"{filename}_gen_{i}.txt")

            # 输出已存在时跳过，避免重复生成
            if os.path.exists(img_out_path) and os.path.exists(label_out_path):
                print(f"[SKIP] 已存在输出，跳过 {filename} idx={i}")
                continue

            result, mask = generate_defect(pipe, image, bbox, prompt)

            new_bbox = mask_to_bbox(mask)
            if new_bbox is None:
                print(f"[WARNING] mask为空，跳过 {filename}")
                continue

            new_labels = []
            for j, other in enumerate(labels):
                new_item = other.copy()

                if j == i:
                    new_item["bbox"] = new_bbox

                new_labels.append(new_item)

            save_image(result, img_out_path)
            save_yolo_labels(label_out_path, new_labels, w, h)
