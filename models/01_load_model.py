"""
01_load_model.py

功能：
加载 Stable Diffusion Inpainting 模型

说明：
该模块负责初始化扩散模型，仅需加载一次，
供后续生成模块调用。

作者：你的论文项目
"""

import torch
from diffusers import StableDiffusionInpaintPipeline


def load_model():
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16
    ).to("cuda")

    return pipe