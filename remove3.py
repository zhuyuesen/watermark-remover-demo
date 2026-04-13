"""
针对左下角双行水印的去除脚本
- 第一行：公司名，固定位置，从 template3/watermark_positions.txt 读取
- 第二行：摄影师名，相同 X 起点，宽度自动检测
"""

from PIL import Image
import cv2
import numpy as np
from pathlib import Path

# === 配置参数 ===
INPUT_FOLDER = "./input"
OUTPUT_FOLDER = "./output"
MODEL_NAME = "lama"
DEVICE = "mps"  # mac M芯片使用 mps，其他使用 cuda 或 cpu

POSITIONS_FILE = "./template3/watermark_positions.txt"

TARGET_WIDTH = 800  # 只处理宽度为 800px 的图片

# 第二行检测参数
ROW2_HEIGHT = 25        # 第二行水印高度（像素），与第一行相同
ROW2_MAX_WIDTH = 400    # 第二行最大扫描宽度
ROW2_BRIGHT_THRESHOLD = 180  # 白色文字亮度阈值

# mask 膨胀参数
DILATE_SIZE = 6

# 调试文件输出配置
SAVE_MASK = False
SAVE_PREVIEW = False


def init_model():
    print("正在加载AI模型...")
    from iopaint.model_manager import ModelManager
    model = ModelManager(name=MODEL_NAME, device=DEVICE)
    print("✅ 模型加载完成")
    return model


def load_row1_position():
    """
    从 watermark_positions.txt 读取第一行水印位置
    格式: x,y_from_bottom,w,h
    其中 y_from_bottom 是从图片底部到水印顶部的距离（像素）
    """
    positions_file = Path(POSITIONS_FILE)
    if not positions_file.exists():
        print(f"❌ 位置文件不存在: {POSITIONS_FILE}")
        return None

    with open(positions_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                x, y_from_bottom, w, h = map(int, line.split(','))
                return (x, y_from_bottom, w, h)

    print("❌ 位置文件中没有有效数据")
    return None


def detect_row2_width(img_array, x_start, y_start, height, max_width):
    """
    自动检测第二行水印的宽度。
    扫描指定区域中的白色像素，找到最右侧的文字边界。
    """
    img_h, img_w = img_array.shape[:2]

    # 防止越界
    y_end = min(y_start + height, img_h)
    x_end = min(x_start + max_width, img_w)

    if y_start >= img_h or x_start >= img_w:
        return 0

    strip = img_array[y_start:y_end, x_start:x_end]
    gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)

    # 检测白色/亮色文字
    _, bright_mask = cv2.threshold(gray, ROW2_BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 找到最右侧有文字像素的列
    col_sums = np.sum(bright_mask, axis=0)
    text_cols = np.where(col_sums > 0)[0]

    if len(text_cols) == 0:
        return 0

    # 返回宽度（加少量边距）
    rightmost = int(text_cols[-1])
    return rightmost + 5


def build_mask(image_shape, row1_pos, row2_pos):
    """根据两行水印位置创建 mask"""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1, w1, h1 = row1_pos
    cv2.rectangle(mask, (x1, y1), (x1 + w1, y1 + h1), 255, -1)

    if row2_pos is not None:
        x2, y2, w2, h2 = row2_pos
        if w2 > 0:
            cv2.rectangle(mask, (x2, y2), (x2 + w2, y2 + h2), 255, -1)

    kernel = np.ones((DILATE_SIZE, DILATE_SIZE), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def remove_watermark(model, image_path, output_path, row1_pos):
    """对单张图片去除双行水印"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)

        x1, y1_from_bottom, w1, h1 = row1_pos
        img_h = img_array.shape[0]

        # 从底部偏移计算实际 Y 坐标（从图片顶部）
        y1 = img_h - y1_from_bottom - h1

        # 第二行紧接第一行上方（水印在左下角，行1在行2下方？不对——行1是上方，行2在下方）
        # 行1是公司名（上），行2是摄影师名（下），两行都在左下角
        # 行2在行1下方，即更靠近图片底部
        y2 = y1 + h1
        row2_width = detect_row2_width(img_array, x1, y2, ROW2_HEIGHT, ROW2_MAX_WIDTH)

        if row2_width > 0:
            row2_pos = (x1, y2, row2_width, ROW2_HEIGHT)
            print(f"  📐 第二行水印宽度: {row2_width}px  位置: x={x1}, y={y2}")
        else:
            row2_pos = None
            print(f"  ⚠️  未检测到第二行水印")

        row1_actual = (x1, y1, w1, h1)
        mask = build_mask(img_array.shape, row1_actual, row2_pos)

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
    """批量处理：扫描输入目录（含一层子文件夹）"""
    row1_pos = load_row1_position()
    if row1_pos is None:
        print("❌ 未能加载第一行水印位置，退出")
        return

    x1, y1, w1, h1 = row1_pos
    print(f"📍 第一行水印位置: x={x1}, y={y1}, w={w1}, h={h1}")
    print(f"📍 第二行扫描起点: x={x1}, y={y1 + h1}，最大宽度: {ROW2_MAX_WIDTH}px\n")

    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    candidates = []
    for ext in image_extensions:
        candidates.extend(input_path.glob(f'**/{ext}'))

    candidates = sorted(set(candidates))
    print(f"🔍 共找到 {len(candidates)} 张图片，筛选宽度 = {TARGET_WIDTH}px\n")

    model = None
    success_count = 0
    failed_count = 0
    skipped_count = 0
    matched_count = 0

    for img_file in candidates:
        try:
            with Image.open(img_file) as probe:
                w, _ = probe.size
        except Exception as e:
            print(f"⚠️  无法读取: {img_file.name} — {e}")
            skipped_count += 1
            continue

        if w != TARGET_WIDTH:
            skipped_count += 1
            continue

        matched_count += 1

        if model is None:
            model = init_model()

        relative_path = img_file.relative_to(input_path)
        print(f"[#{matched_count}] 处理: {relative_path}")

        output_file = output_path / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if remove_watermark(model, img_file, output_file, row1_pos):
            success_count += 1
            print(f"✅ 完成\n")
        else:
            failed_count += 1
            print(f"❌ 失败\n")



    print("=" * 50)
    print(f"🎉 处理完成!")
    print(f"🎯 匹配: {matched_count} 张")
    print(f"✅ 成功: {success_count} 张")
    print(f"❌ 失败: {failed_count} 张")
    print(f"⏭️  跳过 (宽度不符): {skipped_count} 张")
    print(f"📁 输出目录: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    print("🚀 双行水印去除工具启动\n")
    batch_process()
