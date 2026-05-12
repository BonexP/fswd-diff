# Soft Mask 并行实验说明

## 1. 改动目标

本次改动将 soft mask 作为一条与原 hard binary mask 并行的生成路径加入项目。原有 hard-mask 流程保持不变，仍然通过 `run.py` 运行；soft-mask 流程通过新增的 `run_soft.py` 运行。

这样可以在不破坏已有 baseline 的前提下，对比不同 mask transition 对生成质量和下游检测性能的影响。

---

## 2. 当前代码路径

### 2.1 Hard binary mask 路径

```text
run.py
  → core/06_batch_generate.py::batch_generate
  → core/05_inpaint.py::generate_defect
  → utils/02_bbox_2_mask.py::bbox_to_mask
```

该路径使用二元椭圆 mask：

- 白色区域：需要 inpainting 的生成区域；
- 黑色区域：保持原图不变；
- 输出目录默认是 `data/outputs/`。

### 2.2 Soft mask 路径

```text
run_soft.py
  → core/06_batch_generate.py::batch_generate_soft
  → core/05_inpaint.py::generate_defect_soft
  → utils/02_bbox_2_mask.py::bbox_to_soft_mask
```

该路径先调用原有 `bbox_to_mask(...)` 得到 hard mask，再执行 Gaussian Blur：

```python
soft_mask = hard_mask.convert("L").filter(
    ImageFilter.GaussianBlur(radius=blur_radius)
)
```

因此 soft mask 不是替换原 mask 函数，而是在原 hard mask 之上新增羽化版本。

---

## 3. 文件与接口

### 3.1 `utils/02_bbox_2_mask.py`

- `bbox_to_mask(image, bbox, expand_ratio=0.2)`
  - 原有函数，保持 hard binary mask 行为。
  - 当前仍生成压扁椭圆 mask。

- `bbox_to_soft_mask(image, bbox, expand_ratio=0.2, blur_radius=8)`
  - 新增函数。
  - 内部调用 `bbox_to_mask(...)`。
  - 返回灰度模式 soft mask。

### 3.2 `core/05_inpaint.py`

- `generate_defect(pipe, image, bbox, prompt)`
  - 原有 hard-mask 单图生成函数。
  - 默认 `strength=0.85`。

- `generate_defect_soft(pipe, image, bbox, prompt, blur_radius=8, strength=0.85)`
  - 新增 soft-mask 单图生成函数。
  - prompt、steps、guidance 与 hard-mask 版本保持一致。
  - 可通过参数调整 `blur_radius` 和 `strength`。

### 3.3 `core/06_batch_generate.py`

- `batch_generate(...)`
  - 原有 hard-mask 批量生成函数。

- `batch_generate_soft(..., blur_radius=8, strength=0.85)`
  - 新增 soft-mask 批量生成函数。
  - 生成图保存到 `output_dir/images/`。
  - 标签保存到 `output_dir/labels/`。
  - soft mask 保存到 `output_dir/masks/`。

标签 bbox 仍然使用 hard mask 计算，而不是对 soft mask 阈值化后计算。这样可以避免 blur 半径改变 YOLO 标注框大小，使 hard/soft 实验之间的标签定义保持一致。

### 3.4 `run_soft.py`

soft mask 独立入口，默认值如下：

```python
TARGET_CLASS = 2
BLUR_RADIUS = 8
STRENGTH = 0.85
IMAGE_DIR = "data/images"
LABEL_DIR = "data/labels"
OUTPUT_DIR = "data/outputs_soft_r8"
```

支持命令行覆盖：

```bash
python run_soft.py \
  --image-dir data/images \
  --label-dir data/labels \
  --output-dir data/outputs_soft_r8 \
  --target-class 2 \
  --blur-radius 8 \
  --strength 0.85
```

---

## 4. 推荐实验组织方式

建议将 hard mask 和不同 soft mask 参数写入不同输出目录，避免互相覆盖：

```bash
python run.py

python run_soft.py \
  --blur-radius 4 \
  --output-dir data/outputs_soft_r4

python run_soft.py \
  --blur-radius 8 \
  --output-dir data/outputs_soft_r8
```

推荐的第一轮对照：

| 实验名 | 入口 | 输出目录 | 说明 |
|---|---|---|---|
| hard mask baseline | `run.py` | `data/outputs/` | 原二元椭圆 mask |
| soft mask r4 | `run_soft.py --blur-radius 4` | `data/outputs_soft_r4/` | 轻度羽化 |
| soft mask r8 | `run_soft.py --blur-radius 8` | `data/outputs_soft_r8/` | 默认羽化 |

如果后续要做 strength sweep，建议在输出目录中同时记录 strength，例如：

```text
data/outputs_soft_r8_s065/
data/outputs_soft_r8_s075/
data/outputs_soft_r8_s085/
```

---

## 5. 评估方式

`metrics/eval_fid.py` 的默认路径保持不变，因此原命令仍然可用：

```bash
python metrics/eval_fid.py --class 2 --patch
```

新增可选参数用于评估不同实验目录：

- `--real-dir`
- `--label-dir`
- `--fake-dir`
- `--fake-label-dir`
- `--output-csv`
- `--note`

评估 soft mask r8 的示例：

```bash
python metrics/eval_fid.py \
  --class 2 \
  --patch \
  --fake-dir data/outputs_soft_r8/images \
  --fake-label-dir data/outputs_soft_r8/labels \
  --note class_2_soft_r8
```

如果真实图像目录不在默认的 `data/images/`，需要显式传入：

```bash
python metrics/eval_fid.py \
  --real-dir /path/to/images \
  --label-dir data/labels \
  --class 2 \
  --patch \
  --fake-dir data/outputs_soft_r8/images \
  --fake-label-dir data/outputs_soft_r8/labels \
  --note class_2_soft_r8
```

---

## 6. 实验记录建议

建议每组实验至少记录：

- mask 类型：hard / soft；
- `blur_radius`；
- `strength`；
- prompt；
- 生成样本数；
- patch FID / KID；
- 下游检测的 mAP、Recall、F1、class 2 AP；
- 人工观察到的边缘自然性与缺陷形态问题。

论文消融表可以按如下方式组织：

| 方法 | blur radius | strength | Patch FID | KID | class 2 AP | Recall |
|---|---:|---:|---:|---:|---:|---:|
| hard ellipse mask | - | 0.85 | - | - | - | - |
| soft ellipse mask | 4 | 0.85 | - | - | - | - |
| soft ellipse mask | 8 | 0.85 | - | - | - | - |

---

## 7. 注意事项

- `run.py` 是 hard-mask baseline 入口，不应被 soft-mask 实验覆盖。
- `run_soft.py` 默认输出到 `data/outputs_soft_r8/`，用于避免覆盖 `data/outputs/`。
- 当前批量生成逻辑与原项目一致，只处理 `.jpg` 文件。
- soft mask 的 `masks/` 输出用于检查实际送入 diffusion 的羽化掩膜；YOLO 标签仍基于 hard mask bbox。
- 本地代码仓库可以只负责代码和文档提交，真实 diffusion 生成与检测训练可在包含数据和运行库的远端机器执行。
