"""
05_mask_to_bbox.py

功能：
从mask图像中提取bbox（自动标注生成区域）

原理：
- 找到mask中白色区域
- 计算最小外接矩形

作者：你的论文项目（升级版）
"""

import numpy as np


def mask_to_bbox(mask_image):
    """
    输入：
        mask_image: PIL.Image (白=生成区域)

    输出：
        bbox: [x1, y1, x2, y2]
    """

    mask = np.array(mask_image.convert("L"))

    ys, xs = np.where(mask > 128)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    return [x1, y1, x2, y2]
