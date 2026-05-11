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
  python metrics/eval_fid.py --class 2 --patch  # Patch 级评估（按实例对齐）
  python metrics/eval_fid.py --class 2 --patch --expand 0.2  # Patch 级外扩比例

作者：你的论文项目
"""

import os
import csv
import sys
import shutil
import importlib
from datetime import datetime
from pathlib import Path
from PIL import Image

load_yolo_labels = importlib.import_module("utils.03_yolo_utils").load_yolo_labels


def count_images(directory):
    """统计目录下的图片数量"""
    count = 0
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in extensions:
                count += 1
    return count


def find_image_by_stem(image_dir, stem):
    extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in extensions:
        image_path = os.path.join(image_dir, f"{stem}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None


def expand_bbox(bbox, img_width, img_height, expand_ratio):
    x1, y1, x2, y2 = bbox
    dw = (x2 - x1) * expand_ratio
    dh = (y2 - y1) * expand_ratio
    x1 = max(0, x1 - dw)
    y1 = max(0, y1 - dh)
    x2 = min(img_width, x2 + dw)
    y2 = min(img_height, y2 + dh)
    return x1, y1, x2, y2


def ellipse_bounds_from_bbox(expanded_bbox):
    x1, y1, x2, y2 = expanded_bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = (x2 - x1)
    height = (y2 - y1) * 0.3
    return (
        cx - width / 2,
        cy - height / 2,
        cx + width / 2,
        cy + height / 2,
    )


def parse_generated_instance_id(stem):
    if "_gen_" not in stem:
        return None
    base, idx_str = stem.rsplit("_gen_", 1)
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    if base.lower().endswith(".jpg"):
        base = base[:-4]
    return base, idx


def build_real_instances(image_dir, label_dir, target_class):
    instances = {}
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(label_dir, label_file)
        stem = Path(label_file).stem
        image_path = find_image_by_stem(image_dir, stem)
        if image_path is None:
            print(f"  ⚠️  未找到真实图像: {stem}")
            continue
        with Image.open(image_path) as img:
            w, h = img.size
        labels = load_yolo_labels(label_path, w, h)
        for idx, item in enumerate(labels):
            if item["class"] != target_class:
                continue
            instance_id = f"{stem}__idx{idx}"
            instances[instance_id] = {
                "image_path": image_path,
                "bbox": item["bbox"],
                "source": stem,
            }
    return instances


def build_fake_instances(image_dir, label_dir, target_class):
    instances = {}
    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(label_dir, label_file)
        stem = Path(label_file).stem
        parsed = parse_generated_instance_id(stem)
        if parsed is None:
            continue
        base_stem, idx = parsed
        image_path = find_image_by_stem(image_dir, stem)
        if image_path is None:
            print(f"  ⚠️  未找到生成图像: {stem}")
            continue
        with Image.open(image_path) as img:
            w, h = img.size
        labels = load_yolo_labels(label_path, w, h)
        if idx < 0 or idx >= len(labels):
            print(f"  ⚠️  生成标签索引越界: {stem} idx={idx}")
            continue
        item = labels[idx]
        if item["class"] != target_class:
            print(f"  ⚠️  生成标签类别不匹配: {stem} idx={idx} class={item['class']}")
            continue
        instance_id = f"{base_stem}__idx{idx}"
        instances[instance_id] = {
            "image_path": image_path,
            "bbox": item["bbox"],
            "source": stem,
        }
    return instances


def save_patch(image_path, bbox, expand_ratio, output_path):
    with Image.open(image_path) as img:
        w, h = img.size
        x1, y1, x2, y2 = expand_bbox(bbox, w, h, expand_ratio)
        if x2 <= x1 or y2 <= y1:
            return False, None
        patch = img.crop((x1, y1, x2, y2)).convert("RGB")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        patch.save(output_path)
    return True, (x1, y1, x2, y2)


def build_patch_datasets(
    real_image_dir,
    real_label_dir,
    fake_image_dir,
    fake_label_dir,
    target_class,
    expand_ratio,
):
    temp_real_dir = "metrics/.temp_real_patches"
    temp_fake_dir = "metrics/.temp_fake_patches"
    for temp_dir in [temp_real_dir, temp_fake_dir]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

    real_instances = build_real_instances(real_image_dir, real_label_dir, target_class)
    fake_instances = build_fake_instances(fake_image_dir, fake_label_dir, target_class)

    real_ids = set(real_instances.keys())
    fake_ids = set(fake_instances.keys())
    matched_ids = sorted(real_ids & fake_ids)
    real_only = sorted(real_ids - fake_ids)
    fake_only = sorted(fake_ids - real_ids)

    if real_only or fake_only:
        print("\n⚠️  实例数量不一致：")
        print(f"  真实实例数: {len(real_ids)}")
        print(f"  生成实例数: {len(fake_ids)}")
        print(f"  可对齐实例数: {len(matched_ids)}")
        if real_only:
            print(f"  真实缺失匹配: {len(real_only)}")
        if fake_only:
            print(f"  生成缺失匹配: {len(fake_only)}")

    overflow_count = 0
    for instance_id in matched_ids:
        real_item = real_instances[instance_id]
        fake_item = fake_instances[instance_id]
        real_out = os.path.join(temp_real_dir, f"{instance_id}.jpg")
        fake_out = os.path.join(temp_fake_dir, f"{instance_id}.jpg")

        real_ok, real_expanded = save_patch(
            real_item["image_path"], real_item["bbox"], expand_ratio, real_out
        )
        fake_ok, _ = save_patch(
            fake_item["image_path"], fake_item["bbox"], expand_ratio, fake_out
        )
        if not real_ok or not fake_ok:
            continue

        ellipse_bounds = ellipse_bounds_from_bbox(real_expanded)
        x1, y1, x2, y2 = real_expanded
        ex1, ey1, ex2, ey2 = ellipse_bounds
        if ex1 < x1 or ey1 < y1 or ex2 > x2 or ey2 > y2:
            overflow_count += 1

    if overflow_count > 0:
        print(f"\n⚠️  椭圆掩码溢出 BBox: {overflow_count} 次")

    stats = {
        "real_total": len(real_ids),
        "fake_total": len(fake_ids),
        "matched": len(matched_ids),
        "real_only": len(real_only),
        "fake_only": len(fake_only),
        "overflow": overflow_count,
    }
    return temp_real_dir, temp_fake_dir, stats


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


def compute_metrics(
    real_dir,
    fake_dir,
    target_class=None,
    label_dir=None,
    patch_mode=False,
    fake_label_dir=None,
    expand_ratio=0.2,
):
    """计算 FID 与 KID"""
    print("\n" + "="*60)
    print("📊 FID / KID 评估")
    print("="*60)
    meta = {"mode": "full"}

    # 如果指定了目标类别，则筛选同类别的真实图
    if patch_mode:
        if target_class is None or label_dir is None or fake_label_dir is None:
            print("\n❌ Patch 模式需要 --class、真实标签目录与生成标签目录")
            return None, None, 0, 0, meta
        print(f"\n🔍 Patch 模式：按类别 {target_class} 构建实例对齐 patch")
        temp_real_dir, temp_fake_dir, stats = build_patch_datasets(
            real_dir,
            label_dir,
            fake_dir,
            fake_label_dir,
            target_class,
            expand_ratio,
        )
        real_dir = temp_real_dir
        fake_dir = temp_fake_dir
        meta.update(stats)
        meta["mode"] = "patch"
    elif target_class is not None and label_dir is not None:
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
        return None, None, real_count, fake_count, meta

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

        return fid_score, kid_score, real_count, fake_count, meta

    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        return None, None, real_count, fake_count, meta

    finally:
        # 清理临时目录
        if patch_mode:
            for temp_dir in ["metrics/.temp_real_patches", "metrics/.temp_fake_patches"]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        elif target_class is not None and label_dir is not None:
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
    FAKE_LABEL_DIR = "data/outputs/labels"
    OUTPUT_CSV = "metrics/eval_log.csv"
    TARGET_CLASS = None  # 默认对比全量，可通过 --class 参数指定
    PATCH_MODE = False
    EXPAND_RATIO = 0.2

    # 解析命令行参数
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv[1:]):
            if arg == "--class" and i + 1 < len(sys.argv) - 1:
                TARGET_CLASS = int(sys.argv[i + 2])
                print(f"ℹ️  已指定目标类别: {TARGET_CLASS}")
            if arg == "--patch":
                PATCH_MODE = True
            if arg == "--expand" and i + 1 < len(sys.argv) - 1:
                EXPAND_RATIO = float(sys.argv[i + 2])

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

    if PATCH_MODE and not os.path.isdir(FAKE_LABEL_DIR):
        print(f"❌ 错误：生成标签目录不存在: {FAKE_LABEL_DIR}")
        exit(1)

    # 计算指标
    fid_score, kid_score, real_count, fake_count, meta = compute_metrics(
        REAL_DIR, FAKE_DIR,
        target_class=TARGET_CLASS,
        label_dir=LABEL_DIR,
        patch_mode=PATCH_MODE,
        fake_label_dir=FAKE_LABEL_DIR,
        expand_ratio=EXPAND_RATIO,
    )

    # 保存结果
    if fid_score is not None:
        if PATCH_MODE:
            note = f"class_{TARGET_CLASS}_patch" if TARGET_CLASS is not None else "all_classes_patch"
        else:
            note = f"class_{TARGET_CLASS}" if TARGET_CLASS is not None else "all_classes"
        if meta.get("mode") == "patch" and (meta.get("real_only") or meta.get("fake_only")):
            note += f"_mismatch_r{meta.get('real_only')}_f{meta.get('fake_only')}"
        save_result(fid_score, kid_score, real_count, fake_count, OUTPUT_CSV, note)
        print("\n✅ 评估完成！")
        print(f"\n📊 最终结果:")
        print(f"   🎯 FID = {fid_score:.4f}  (越低越好，0为完美)")
        print(f"   🎯 KID = {kid_score:.4f}  (越低越好，对小样本更稳定)")
        print(f"   📈 样本数: 真实={real_count}, 生成={fake_count} (建议 >= 100)")
        mode_label = "Patch 级" if PATCH_MODE else ("单类别（class " + str(TARGET_CLASS) + "）" if TARGET_CLASS is not None else "全类别")
        print(f"   🏷️  对比模式: {mode_label}")
    else:
        print("\n❌ 评估失败，请检查数据目录与依赖")

