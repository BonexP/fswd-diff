"""
02_bbox_to_mask.py

功能：
将YOLO或通用bbox转换为mask图像

说明：
mask中：
- 白色区域 = 生成区域
- 黑色区域 = 保持不变

作者：你的论文项目
"""

from PIL import Image, ImageDraw


def bbox_to_mask(image, bbox, expand_ratio=0.2):
    w, h = image.size
    mask = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(mask)

    x1, y1, x2, y2 = bbox

    dw = (x2 - x1) * expand_ratio
    dh = (y2 - y1) * expand_ratio

    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(w, x2 + dw)
    y2 = min(h, y2 + dh)

    draw.rectangle([x1, y1, x2, y2], fill="white")

    return mask