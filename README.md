## 项目简介

本项目用于基于扩散模型（Stable Diffusion Inpainting）对搅拌摩擦焊（FSW）缺陷数据进行生成式增强。

### 核心功能

- bbox → mask
- 掩膜引导缺陷生成
- 自动批量扩充数据集
- 支持YOLO训练数据生成

## 项目结构

- data/：原始数据与生成数据
- models/：模型加载
- utils/：工具函数
- core/：核心逻辑
- run.py：主运行入口

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 准备数据

images 放入 data/images/

labels（YOLO格式）放入 data/labels/

3. 运行

```bash
python run.py
```

## 输出

生成图像保存在：

data/outputs/

## 作者

用于硕士论文：FSW缺陷检测与生成增强