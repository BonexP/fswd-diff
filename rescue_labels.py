"""
rescue_labels.py

功能：
修复生成数据的YOLO标签：
- 补回原始标签
- 替换对应的隧道bbox

作者：论文工程修复工具
"""

import os


# ========================
# 工具函数
# ========================

def load_yolo_txt(path):
    labels = []
    with open(path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            cls = int(parts[0])
            bbox = list(map(float, parts[1:]))
            labels.append({"class": cls, "bbox": bbox})
    return labels


def save_yolo_txt(path, labels):
    with open(path, "w") as f:
        for item in labels:
            cls = item["class"]
            bbox = item["bbox"]
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")


# ========================
# IoU计算（核心）
# ========================

def yolo_to_xyxy(bbox):
    x, y, w, h = bbox
    x1 = x - w/2
    y1 = y - h/2
    x2 = x + w/2
    y2 = y + h/2
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter / union


# ========================
# 核心修复逻辑
# ========================

def rescue_one(original_labels, gen_label, target_class=2):

    gen_bbox = gen_label[0]["bbox"]
    gen_xyxy = yolo_to_xyxy(gen_bbox)

    best_iou = -1
    best_idx = -1

    # 找最匹配的隧道
    for i, item in enumerate(original_labels):
        if item["class"] != target_class:
            continue

        orig_xyxy = yolo_to_xyxy(item["bbox"])
        iou = compute_iou(gen_xyxy, orig_xyxy)

        if iou > best_iou:
            best_iou = iou
            best_idx = i

    # 复制标签
    new_labels = []

    for i, item in enumerate(original_labels):
        if i == best_idx:
            # 替换bbox
            new_labels.append({
                "class": target_class,
                "bbox": gen_bbox
            })
        else:
            new_labels.append(item)

    return new_labels


# ========================
# 主函数
# ========================

def rescue_all(
    gen_label_dir,
    original_label_dir,
    output_dir,
    target_class=2
):

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(gen_label_dir):

        if not filename.endswith(".txt"):
            continue

        # 解析原图名
        base_name = filename.split(".jpg_gen_")[0] + ".txt"

        gen_path = os.path.join(gen_label_dir, filename)
        orig_path = os.path.join(original_label_dir, base_name)

        if not os.path.exists(orig_path):
            print(f"[WARN] 找不到原始标签: {base_name}")
            continue

        gen_label = load_yolo_txt(gen_path)
        orig_labels = load_yolo_txt(orig_path)

        new_labels = rescue_one(orig_labels, gen_label, target_class)

        out_path = os.path.join(output_dir, filename)
        save_yolo_txt(out_path, new_labels)

        print(f"[OK] 修复: {filename}")


# ========================
# 运行入口
# ========================

if __name__ == "__main__":

    rescue_all(
        gen_label_dir="data/outputs/labels",        # 生成标签
        original_label_dir="data/labels",           # 原始标签
        output_dir="data/outputs/labels_fixed",     # 输出
        target_class=2                              # 隧道类
    )