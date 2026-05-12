## 项目简介

本项目用于基于扩散模型（Stable Diffusion Inpainting）对搅拌摩擦焊（FSW）缺陷数据进行生成式增强。

### 核心功能

- bbox → mask
- hard binary mask 引导缺陷生成
- soft mask 羽化掩膜引导缺陷生成
- 自动批量扩充 YOLO 训练数据集
- 支持 FID / KID 生成质量评估

## 项目结构

- `data/`：原始数据与生成数据
- `models/`：扩散模型加载
- `utils/`：bbox、mask、YOLO 标签与图像工具函数
- `core/`：inpainting 与批量生成核心逻辑
- `metrics/`：FID / KID 评估脚本
- `doc/`：实验方法与评估说明文档
- `run.py`：原 hard binary mask 生成入口
- `run_soft.py`：soft mask 生成入口

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 准备数据

```text
data/images/  原始图像
data/labels/  YOLO 格式标签
```

3. 运行原 hard binary mask 流程

```bash
python run.py
```

4. 运行 soft mask 流程

```bash
python run_soft.py --blur-radius 8 --output-dir data/outputs_soft_r8
```

`run_soft.py` 默认使用：

- `target_class=2`
- `blur_radius=8`
- `strength=0.85`
- `output_dir=data/outputs_soft_r8`

这些默认值可以通过命令行参数覆盖，例如：

```bash
python run_soft.py --image-dir data/images --label-dir data/labels --target-class 2 --blur-radius 4 --strength 0.85 --output-dir data/outputs_soft_r4
```

## 输出目录

原 hard binary mask 流程默认输出到：

```text
data/outputs/
├── images/
└── labels/
```

soft mask 流程建议输出到独立目录，避免覆盖 hard-mask baseline：

```text
data/outputs_soft_r8/
├── images/
├── labels/
└── masks/
```

其中 `masks/` 保存实际送入 diffusion 的 soft mask，便于人工检查羽化边界。

## 评估示例

原默认评估方式保持不变：

```bash
python metrics/eval_fid.py --class 2 --patch
```

评估 soft mask 实验时，可指定生成结果目录与备注：

```bash
python metrics/eval_fid.py --class 2 --patch --fake-dir data/outputs_soft_r8/images --fake-label-dir data/outputs_soft_r8/labels --note class_2_soft_r8
```

更多 soft mask 设计与实验建议见：

- `doc/SoftMask并行实验说明.md`

## 作者

用于硕士论文：FSW 缺陷检测与生成增强。
