"""
eval_fid.py

功能：
自动计算 FID 与 KID 指标，并保存结果到 CSV

说明：
- 统计真实图 vs 生成图的样本数
- 支持按目标类别筛选（避免类别混淆导致的高 FID）
- 计算 FID（Fréchet Inception Distance）
- 计算 KID（Kernel Inception Distance，对小样本更稳）
- 结果保存到 metrics/eval_log.csv，便于横向对比与论文复现

使用：
  python metrics/eval_fid.py              # 全量对比
  python metrics/eval_fid.py --class 2   # 仅对比 class 2 的真实图 vs 生成图

作者：你的论文项目
"""

import os
import csv
import sys
import shutil
from datetime import datetime
from pathlib import Path

def count_images(directory):
    """统计目录下的图片数量"""
    count = 0
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                count += 1
    return count


def filter_images_by_class(image_dir, label_dir, target_class, output_dir):
    """
    从 image_dir 中筛选出包含 target_class 的图像。
    如果一张图里有多个 target_class 实例，则复制多份到 output_dir，
    让真实集按实例数对齐生成集的采样口径。
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    image_count = 0
    instance_count = 0

    # 遍历标签文件
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(label_dir, label_file)

        # 读取标签文件，统计 target_class 实例数
        target_instances = 0
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(parts[0]) == target_class:
                        target_instances += 1
        except Exception as e:
            print(f"  ⚠️  读取标签文件失败: {label_file} - {e}")
            continue

        if target_instances == 0:
            continue

        # 找到对应的图像文件
        stem = Path(label_file).stem
        source_path = None
        source_ext = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_file = f"{stem}{ext}"
            image_path = os.path.join(image_dir, image_file)
            if os.path.exists(image_path):
                source_path = image_path
                source_ext = ext
                break

        if source_path is None:
            print(f"  ⚠️  未找到对应图像: {stem}")
            continue

        image_count += 1
        for idx in range(target_instances):
            output_name = f"{stem}__cls{target_class}_{idx}{source_ext}"
            output_path = os.path.join(output_dir, output_name)
            shutil.copy(source_path, output_path)
            instance_count += 1

    return image_count, instance_count


def compute_metrics(real_dir, fake_dir, target_class=None, label_dir=None):
    """计算 FID 与 KID"""
    print("\n" + "="*60)
    print("📊 FID / KID 评估")
    print("="*60)

    # 如果指定了目标类别，则筛选同类别的真实图
    if target_class is not None and label_dir is not None:
        print(f"\n🔍 按类别 {target_class} 筛选真实图...")
        temp_real_dir = "metrics/.temp_real_class"
        real_image_count, real_count_filtered = filter_images_by_class(
            real_dir, label_dir, target_class, temp_real_dir
        )
        print(f"  已筛选出 {real_image_count} 张包含类别 {target_class} 的真实图")
        print(f"  已按实例对齐扩展为 {real_count_filtered} 个真实样本")
        real_dir = temp_real_dir

    # 统计样本数
    real_count = count_images(real_dir)
    fake_count = count_images(fake_dir)

    print(f"\n✓ 真实图目录: {real_dir}")
    print(f"  样本数: {real_count}")
    print(f"\n✓ 生成图目录: {fake_dir}")
    print(f"  样本数: {fake_count}")

    if real_count == 0 or fake_count == 0:
        print("\n❌ 错误：至少一个目录为空，无法计算指标")
        print(f"   真实图: {real_count}, 生成图: {fake_count}")
        return None, None, real_count, fake_count

    print(f"\n⏳ 计算中... (样本数: {real_count} vs {fake_count})")

    try:
        from cleanfid import fid

        # 计算 FID
        print("\n  计算 FID...")
        fid_score = fid.compute_fid(real_dir, fake_dir)
        print(f"  FID = {fid_score:.4f}")

        # 计算 KID
        print("  计算 KID...")
        kid_score = fid.compute_kid(real_dir, fake_dir)
        print(f"  KID = {kid_score:.4f}")

        return fid_score, kid_score, real_count, fake_count

    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        return None, None, real_count, fake_count

    finally:
        # 清理临时目录
        if target_class is not None and label_dir is not None:
            temp_real_dir = "metrics/.temp_real_class"
            if os.path.exists(temp_real_dir):
                shutil.rmtree(temp_real_dir)


def save_result(fid_score, kid_score, real_count, fake_count, output_csv="metrics/eval_log.csv", note=""):
    """保存结果到 CSV"""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 检查文件是否存在
    file_exists = os.path.exists(output_csv)

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 如果是新文件，写入表头
        if not file_exists:
            writer.writerow([
                "时间戳",
                "FID",
                "KID",
                "真实样本数",
                "生成样本数",
                "备注"
            ])

        # 写入数据行
        fid_str = f"{fid_score:.4f}" if fid_score is not None else "ERROR"
        kid_str = f"{kid_score:.4f}" if kid_score is not None else "ERROR"
        writer.writerow([
            timestamp,
            fid_str,
            kid_str,
            real_count,
            fake_count,
            note
        ])

    print(f"\n✓ 结果已保存: {output_csv}")
    print_csv_tail(output_csv, lines=3)


def print_csv_tail(filepath, lines=5):
    """打印 CSV 最后几行（带表头）"""
    if not os.path.exists(filepath):
        return

    with open(filepath, "r", encoding="utf-8") as f:
        rows = f.readlines()

    print("\n📋 最近评估记录:")
    print("  " + "-" * 70)

    # 打印表头（第一行）
    if rows:
        print("  " + rows[0].strip())
        print("  " + "-" * 70)
        # 打印最后 lines 行数据
        for row in rows[-lines:]:
            print("  " + row.strip())

    print("  " + "-" * 70)


if __name__ == "__main__":
    # 配置路径（可根据需要修改）
    REAL_DIR = "data/images"
    LABEL_DIR = "data/labels"
    FAKE_DIR = "data/outputs/images"
    OUTPUT_CSV = "metrics/eval_log.csv"
    TARGET_CLASS = None  # 默认对比全量，可通过 --class 参数指定

    # 解析命令行参数
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--class" and i + 1 < len(sys.argv) - 1:
                TARGET_CLASS = int(sys.argv[i + 2])
                print(f"ℹ️  已指定目标类别: {TARGET_CLASS}")

    # 验证路径
    if not os.path.isdir(REAL_DIR):
        print(f"❌ 错误：真实图目录不存在: {REAL_DIR}")
        exit(1)

    if not os.path.isdir(FAKE_DIR):
        print(f"❌ 错误：生成图目录不存在: {FAKE_DIR}")
        exit(1)

    if TARGET_CLASS is not None and not os.path.isdir(LABEL_DIR):
        print(f"❌ 错误：标签目录不存在: {LABEL_DIR}")
        exit(1)

    # 计算指标
    fid_score, kid_score, real_count, fake_count = compute_metrics(
        REAL_DIR, FAKE_DIR,
        target_class=TARGET_CLASS,
        label_dir=LABEL_DIR
    )

    # 保存结果
    if fid_score is not None:
        note = f"class_{TARGET_CLASS}" if TARGET_CLASS is not None else "all_classes"
        save_result(fid_score, kid_score, real_count, fake_count, OUTPUT_CSV, note)
        print("\n✅ 评估完成！")
        print(f"\n📊 最终结果:")
        print(f"   🎯 FID = {fid_score:.4f}  (越低越好，0为完美)")
        print(f"   🎯 KID = {kid_score:.4f}  (越低越好，对小样本更稳定)")
        print(f"   📈 样本数: 真实={real_count}, 生成={fake_count} (建议 >= 100)")
        print(f"   🏷️  对比模式: {'单类别（class ' + str(TARGET_CLASS) + '）' if TARGET_CLASS is not None else '全类别'}")
    else:
        print("\n❌ 评估失败，请检查数据目录与依赖")





