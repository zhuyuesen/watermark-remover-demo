"""
对比原图和处理后图片的色彩
"""

from PIL import Image
import numpy as np
from pathlib import Path

def compare_colors(original_path, processed_path):
    """对比两张图片的色彩"""

    # 读取图片
    original = Image.open(original_path).convert('RGB')
    processed = Image.open(processed_path).convert('RGB')

    # 转换为数组
    orig_array = np.array(original)
    proc_array = np.array(processed)

    # 采样几个点来对比（避开水印区域）
    # 取左上角、右上角、左下角的非水印区域
    sample_points = [
        (10, 10),     # 左上角
        (10, 600),    # 左下角
        (900, 10),    # 右上角
    ]

    print("🎨 色彩对比分析\n")
    print(f"原图: {original_path}")
    print(f"处理后: {processed_path}\n")

    print("采样点色彩对比（RGB 值）：")
    print("-" * 60)

    all_similar = True
    for i, (x, y) in enumerate(sample_points, 1):
        if y < orig_array.shape[0] and x < orig_array.shape[1]:
            orig_color = orig_array[y, x]
            proc_color = proc_array[y, x]

            # 计算色差
            diff = np.abs(orig_color.astype(int) - proc_color.astype(int))
            max_diff = np.max(diff)

            print(f"\n采样点 {i} ({x}, {y}):")
            print(f"  原图:     RGB{tuple(orig_color)}")
            print(f"  处理后:   RGB{tuple(proc_color)}")
            print(f"  色差:     {tuple(diff)} (最大差值: {max_diff})")

            if max_diff > 10:  # 如果色差超过 10，认为不相似
                print(f"  ⚠️  色彩差异较大！")
                all_similar = False
            else:
                print(f"  ✅ 色彩相似")

    print("\n" + "="*60)
    if all_similar:
        print("✅ 结论：色彩保持良好，无明显偏色")
    else:
        print("❌ 结论：存在色彩偏差，需要检查色彩空间转换")
    print("="*60)

if __name__ == "__main__":
    # 查找第一张处理后的图片
    input_dir = Path("./input")
    output_dir = Path("./output")

    # 获取第一张图片
    input_images = sorted(input_dir.glob("*.jpg"))
    if len(input_images) > 0:
        original = input_images[0]
        processed = output_dir / original.name

        if processed.exists():
            compare_colors(original, processed)
        else:
            print(f"❌ 未找到处理后的图片: {processed}")
    else:
        print("❌ 未找到输入图片")
