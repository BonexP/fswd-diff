"""
run.py

功能：
项目主入口

说明：
运行该文件即可执行整个生成流程

作者：你的论文项目
"""

from models.01_load_model import load_model
from core.06_batch_generate import batch_generate

if __name__ == "__main__":

    pipe = load_model()

    prompt = "elongated tunnel defect on friction stir welding surface, realistic metal texture"

    batch_generate(
        pipe,
        image_dir="data/images",
        label_dir="data/labels",
        output_dir="data/outputs",
        prompt=prompt
    )