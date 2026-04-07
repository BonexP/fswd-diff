"""
04_image_utils.py

功能：
图像读取与保存辅助函数

作者：你的论文项目
"""

from PIL import Image
import os


def load_image(path):
    return Image.open(path).convert("RGB")


def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)