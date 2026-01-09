"""
测试 iopaint 的色彩处理
"""

from PIL import Image
import numpy as np
import cv2
from pathlib import Path

# 导入 iopaint
from iopaint.model_manager import ModelManager
from iopaint.schema import InpaintRequest

def test_color_processing():
    print("🧪 测试 iopaint 色彩处理\n")

    # 加载模型
    print("加载模型...")
    model = ModelManager(name="lama", device="cpu")

    # 读取一张测试图片
    test_image_path = "./input/215558000_1.jpg"
    img_pil = Image.open(test_image_path).convert('RGB')

    # 获取原始像素值
    img_array = np.array(img_pil)
    print(f"原始 PIL RGB 像素 [0,0]: {img_array[0, 0]}")

    # 创建一个全黑的 mask（不做任何修复）
    mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

    # 测试 1：直接传入 RGB
    print("\n测试 1：直接传入 RGB 数组")
    config = InpaintRequest(ldm_steps=1, hd_strategy='Original', enable_controlnet=False)
    result_rgb = model(image=img_array, mask=mask, config=config)
    print(f"  结果像素 [0,0]: {result_rgb[0, 0]}")
    print(f"  差异: {np.abs(img_array[0, 0].astype(int) - result_rgb[0, 0].astype(int))}")

    # 测试 2：传入 BGR
    print("\n测试 2：传入 BGR 数组")
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    print(f"  BGR 像素 [0,0]: {img_bgr[0, 0]}")
    result_bgr = model(image=img_bgr, mask=mask, config=config)
    print(f"  结果像素 [0,0]: {result_bgr[0, 0]}")

    # 转换回 RGB 查看
    result_bgr_to_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    print(f"  转回 RGB: {result_bgr_to_rgb[0, 0]}")
    print(f"  与原图差异: {np.abs(img_array[0, 0].astype(int) - result_bgr_to_rgb[0, 0].astype(int))}")

    # 保存测试结果
    print("\n保存测试结果...")
    Image.fromarray(result_rgb).save("test_rgb.jpg")
    Image.fromarray(result_bgr_to_rgb).save("test_bgr.jpg")

    print("\n✅ 测试完成！")
    print("请对比:")
    print("  - 原图: input/215558000_1.jpg")
    print("  - RGB 测试: test_rgb.jpg")
    print("  - BGR 测试: test_bgr.jpg")

if __name__ == "__main__":
    test_color_processing()
