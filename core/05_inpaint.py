"""
05_inpaint.py

功能：
执行单张图像的缺陷生成（Inpainting）

流程：
image + bbox → mask → diffusion → 新图

作者：你的论文项目
"""

from utils.02_bbox_to_mask import bbox_to_mask


def generate_defect(pipe, image, bbox, prompt):
    mask = bbox_to_mask(image, bbox)

    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=30,
        guidance_scale=7.5,
        strength=0.85
    ).images[0]

    return result, mask