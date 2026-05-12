"""
run_soft.py

Soft-mask entrypoint for the inpainting generation workflow.
The original run.py hard-mask workflow is intentionally left unchanged.
"""

import argparse
import importlib

load_model = importlib.import_module("models.01_load_model").load_model
batch_generate_soft = importlib.import_module(
    "core.06_batch_generate"
).batch_generate_soft

PROMPT = (
    "elongated tunnel defect along the welding seam, "
    "narrow and continuous cavity, "
    "dark interior, "
    "realistic metal surface, "
    "industrial defect, "
    "high detail"
)
TARGET_CLASS = 2
BLUR_RADIUS = 8
STRENGTH = 0.85
IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
OUTPUT_DIR = "data/outputs_soft_r8"


def parse_args():
    parser = argparse.ArgumentParser(description="Run soft-mask defect generation.")
    parser.add_argument("--image-dir", default=IMAGE_DIR)
    parser.add_argument("--label-dir", default=LABEL_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--target-class", type=int, default=TARGET_CLASS)
    parser.add_argument("--blur-radius", type=float, default=BLUR_RADIUS)
    parser.add_argument("--strength", type=float, default=STRENGTH)
    parser.add_argument("--prompt", default=PROMPT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipe = load_model()

    batch_generate_soft(
        pipe,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        target_class=args.target_class,
        prompt=args.prompt,
        blur_radius=args.blur_radius,
        strength=args.strength
    )
