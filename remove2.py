"""
针对分辨率 500x352 图片的水印移除脚本
使用 template2/ 目录中的固定水印位置
"""

from PIL import Image
import cv2
import numpy as np
from pathlib import Path

# === 配置参数 ===
INPUT_FOLDER = "./input"
OUTPUT_FOLDER = "./output"
MODEL_NAME = "lama"
DEVICE = "cpu"  # Windows 使用 cpu，Mac M芯片使用 mps，有 Nvidia GPU 使用 cuda

TARGET_SIZE = (500, 352)  # 目标图片分辨率 (宽, 高)

POSITIONS_FILE = "./template2/watermark_positions.txt"

# 调试文件输出配置
SAVE_MASK = False
SAVE_PREVIEW = False


def init_model():
    print("正在加载AI模型...")
    from iopaint.model_manager import ModelManager
    model = ModelManager(name=MODEL_NAME, device=DEVICE)
    print("✅ 模型加载完成")
    return model


def cv2_imread_chinese(file_path):
    try:
        img_pil = Image.open(file_path).convert('RGB')
        img_array = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        print(f"❌ 读取图片失败: {file_path}, 错误: {e}")
        return None


def load_positions():
    """
    从 watermark_positions.txt 读取水印位置
    每行格式: x,y,w,h
    行1 对应 wm01.png，行2 对应 wm02.png
    """
    positions_file = Path(POSITIONS_FILE)
    if not positions_file.exists():
        print(f"❌ 位置文件不存在: {POSITIONS_FILE}")
        return []

    positions = []
    with open(positions_file, 'r') as f:
        for line in f:
            if line.strip():
                x, y, w, h = map(int, line.strip().split(','))
                positions.append((x, y, w, h))

    return positions


def build_mask(image_shape, positions):
    """
    根据水印位置列表创建 mask
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, ww, hh in positions:
        cv2.rectangle(mask, (x, y), (x + ww, y + hh), 255, -1)

    # 稍微扩大 mask 区域以提升修复效果
    kernel_dilate = np.ones((10, 10), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    return mask


def remove_watermark(model, image_path, output_path, positions):
    """
    对单张图片去除水印（使用 lama AI 模型）
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        mask = build_mask(img_array.shape, positions)

        if SAVE_MASK:
            mask_path = str(output_path).rsplit('.', 1)[0] + '_mask.png'
            Image.fromarray(mask, mode='L').save(mask_path)
            print(f"  💾 Mask 已保存: {mask_path}")

        if SAVE_PREVIEW:
            preview_path = str(output_path).rsplit('.', 1)[0] + '_preview.png'
            preview = img_array.copy()
            preview[mask > 0] = (preview[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
            Image.fromarray(preview).save(preview_path)
            print(f"  👁️  预览图已保存: {preview_path}")

        from iopaint.schema import InpaintRequest
        config = InpaintRequest(
            ldm_steps=20,
            hd_strategy='Original',
            enable_controlnet=False,
        )

        result = model(image=img_array, mask=mask, config=config)

        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb, mode='RGB')

        ext = str(output_path).lower()
        if ext.endswith('.jpg') or ext.endswith('.jpeg'):
            result_image.save(output_path, 'JPEG', quality=95, optimize=True)
        else:
            result_image.save(output_path, optimize=True)

        return True

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_process():
    """
    批量处理：扫描输入目录（含一层子文件夹），找出分辨率为 500x352 的图片并去水印
    """
    positions = load_positions()
    if not positions:
        print("❌ 未能加载水印位置，退出")
        return

    print(f"📍 已加载 {len(positions)} 个水印位置:")
    wm_names = ['wm01.png', 'wm02.png']
    for i, (x, y, w, h) in enumerate(positions):
        name = wm_names[i] if i < len(wm_names) else f'wm{i+1:02d}.png'
        print(f"   位置 {i+1} ({name}): x={x}, y={y}, w={w}, h={h}")

    model = init_model()

    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.png']

    # 收集候选文件（根目录 + 一层子文件夹）
    candidates = []
    for ext in image_extensions:
        candidates.extend(input_path.glob(ext))
        candidates.extend(input_path.glob(f'*/{ext}'))

    print(f"\n🔍 开始扫描，目标分辨率: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}\n")

    success_count = 0
    skipped_count = 0
    failed_count = 0
    matched_count = 0

    for img_file in sorted(set(candidates)):
        try:
            with Image.open(img_file) as probe:
                size = probe.size  # (宽, 高)
        except Exception as e:
            print(f"⚠️  无法读取图片尺寸: {img_file.name} — {e}")
            continue

        if size != TARGET_SIZE:
            skipped_count += 1
            continue

        matched_count += 1
        relative_path = img_file.relative_to(input_path)
        print(f"[{matched_count}] 处理: {relative_path}  ({size[0]}x{size[1]})")

        output_file = output_path / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if remove_watermark(model, img_file, output_file, positions):
            success_count += 1
            print(f"✅ 完成\n")
        else:
            failed_count += 1
            print(f"❌ 失败\n")

    print("=" * 50)
    print(f"🎉 处理完成!")
    print(f"📊 扫描总数: {matched_count + skipped_count} 张")
    print(f"🎯 匹配 500x352: {matched_count} 张")
    print(f"✅ 成功: {success_count} 张")
    print(f"❌ 失败: {failed_count} 张")
    print(f"⏭️  跳过 (分辨率不符): {skipped_count} 张")
    print(f"📁 输出目录: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    print("🚀 500x352 专用去水印工具启动\n")
    batch_process()
