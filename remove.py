"""
批量去除水印脚本
"""

from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import sys

# === 配置参数 ===
INPUT_FOLDER = "./input"
OUTPUT_FOLDER = "./output"
MODEL_NAME = "lama" # AI 模型名称
DEVICE = "mps" # mac M芯片使用 mps, 其他使用 cuda 或 cpu

# 调试文件输出配置
SAVE_MASK = False      # 是否保存 _mask.png 文件（水印检测区域）
SAVE_PREVIEW = False   # 是否保存 _preview.png 文件（红色标记预览图）

# === 函数定义 ===
def init_model():
  print("正在加载AI模型...")

  from iopaint.model_manager import ModelManager
  from iopaint.schema import HDStrategy

  model = ModelManager(
    name=MODEL_NAME,
    device=DEVICE
  )

  print("✅ 模型加载完成")
  return model

def detect_watermark_auto(image_path):
  """
  自动检测水印位置（改进版：检测亮色和暗色水印）
  """
  img = cv2_imread_chinese(str(image_path))
  if img is None:
    print(f"❌ 无法读取图片: {image_path}")
    return None

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 方法1：检测亮色水印（白色或浅色）
  _, mask_bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

  # 方法2：检测暗色水印（黑色或深色）
  _, mask_dark = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

  # 方法3：使用边缘检测找出文字或图案边缘
  edges = cv2.Canny(gray, 100, 200)

  # 合并三种检测结果
  mask = cv2.bitwise_or(mask_bright, mask_dark)
  mask = cv2.bitwise_or(mask, edges)

  # 形态学处理，去除噪点并填充空洞
  kernel = np.ones((5, 5), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

  return mask

def cv2_imread_chinese(file_path):
  """
  支持中文路径的 cv2.imread 替代方案
  """
  try:
    # 使用 numpy 和 PIL 读取，避免中文路径问题
    img_pil = Image.open(file_path).convert('RGB')
    # PIL 使用 RGB，OpenCV 使用 BGR，需要转换
    img_array = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return img_bgr
  except Exception as e:
    print(f"❌ 读取图片失败: {file_path}, 错误: {e}")
    return None

def detect_watermark_by_fixed_positions(image_path, template_path):
  """
  使用固定位置检测水印（适用于水印位置完全一致的情况）
  先用模板匹配找出所有水印位置，然后保存这些位置供后续使用
  """
  img = cv2_imread_chinese(str(image_path))
  if img is None:
    print(f"❌ 无法读取图片: {image_path}")
    return None

  template = cv2_imread_chinese(str(template_path))
  if template is None:
    print(f"❌ 模板文件不存在: {template_path}")
    return None

  # 检查是否已有保存的位置信息
  positions_file = Path("./template/watermark_positions.txt")

  if positions_file.exists():
    # 读取已保存的位置
    print(f"  📍 使用已保存的水印位置")
    positions = []
    with open(positions_file, 'r') as f:
      for line in f:
        if line.strip():
          x, y, w, h = map(int, line.strip().split(','))
          positions.append((x, y, w, h))

    # 创建 mask
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for x, y, w, h in positions:
      cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    print(f"  ✅ 应用 {len(positions)} 个固定水印位置")

    # 扩大 mask 区域
    kernel_dilate = np.ones((10, 10), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    return mask

  else:
    # 首次运行：使用模板匹配找出所有位置并保存
    print(f"  🔍 首次检测，正在查找所有水印位置...")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    h, w = template_gray.shape
    mask = np.zeros_like(img_gray)

    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.5
    locations = np.where(result >= threshold)

    candidates = []
    for pt in zip(*locations[::-1]):
      candidates.append((pt[0], pt[1], result[pt[1], pt[0]]))

    candidates.sort(key=lambda x: x[2], reverse=True)

    selected = []
    for x, y, _ in candidates:
      overlap = False
      for sx, sy in selected:
        if abs(x - sx) < w * 0.5 and abs(y - sy) < h * 0.5:
          overlap = True
          break
      if not overlap:
        selected.append((x, y))
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # 保存位置信息
    if len(selected) > 0:
      with open(positions_file, 'w') as f:
        for x, y in selected:
          f.write(f"{x},{y},{w},{h}\n")

      print(f"  💾 已保存 {len(selected)} 个水印位置到 {positions_file}")
      print(f"  📝 位置信息：")
      for i, (x, y) in enumerate(selected[:10]):
        print(f"     位置 {i+1}: ({x}, {y})")
    else:
      print(f"  ⚠️  未检测到水印，请检查模板图片")

    # 扩大 mask 区域
    kernel_dilate = np.ones((10, 10), np.uint8)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    return mask

def detect_watermark_by_template(image_path, template_path):
  """
  使用模板匹配检测水印位置（支持多个匹配结果）
  """
  # 读取彩色图片以获得更好的匹配效果
  img = cv2.imread(str(image_path))
  if img is None:
    print(f"❌ 无法读取图片: {image_path}")
    return None

  template = cv2.imread(str(template_path))

  if template is None:
    print(f"❌ 模板文件不存在: {template_path}")
    return None

  # 转换为灰度图进行匹配
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

  # 获取模板尺寸
  h, w = template_gray.shape

  # 创建空白掩码
  mask = np.zeros_like(img_gray)

  # 模板匹配
  result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

  # 降低阈值以检测更多水印
  threshold = 0.5
  locations = np.where(result >= threshold)

  # 使用非极大值抑制来避免重复检测
  # 先收集所有候选位置
  candidates = []
  for pt in zip(*locations[::-1]):
    candidates.append((pt[0], pt[1], result[pt[1], pt[0]]))

  # 按相似度排序
  candidates.sort(key=lambda x: x[2], reverse=True)

  # 非极大值抑制：去除重叠的检测框
  selected = []
  for x, y, _ in candidates:  # score 未使用，用 _ 代替
    overlap = False
    for sx, sy in selected:
      # 如果两个框中心距离小于模板尺寸的一半，认为是重复
      if abs(x - sx) < w * 0.5 and abs(y - sy) < h * 0.5:
        overlap = True
        break
    if not overlap:
      selected.append((x, y))
      cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

  match_count = len(selected)

  if match_count == 0:
    max_val = np.max(result)
    print(f"  ⚠️  未找到匹配的水印（最高相似度: {max_val:.2f}，阈值: {threshold}）")
    # 即使没找到，也尝试使用最高匹配度的位置
    _, _, _, max_loc = cv2.minMaxLoc(result)
    cv2.rectangle(mask, max_loc, (max_loc[0] + w, max_loc[1] + h), 255, -1)
    print(f"  🔍 使用最佳匹配位置: {max_loc}")
    match_count = 1
  else:
    print(f"  ✅ 检测到 {match_count} 个水印位置")
    for i, (x, y) in enumerate(selected[:5]):  # 只显示前5个
      print(f"     位置 {i+1}: ({x}, {y})")

  # 扩大 mask 区域，帮助 AI 更好地修复
  kernel_dilate = np.ones((10, 10), np.uint8)
  mask = cv2.dilate(mask, kernel_dilate, iterations=1)

  return mask

def remove_watermark(model, image_path, output_path, use_template=False):
  """
  去除单张图片的水印
  
  :param model: AI 模型实例
  :param image_path: 输入图片路径
  :param output_path: 输出图片路径
  :param use_template: 是否使用模板匹配
  """
  try:
    img = Image.open(image_path).convert('RGB')

    if use_template:
      template_path = "./template/watermark.png"
      # 使用固定位置检测（更快更准确）
      mask_araay = detect_watermark_by_fixed_positions(image_path, template_path)
    else:
      mask_araay = detect_watermark_auto(image_path)

    if mask_araay is None:
      return False

    # 调试信息
    print(f"  🔍 原始图片尺寸: {img.size}")
    print(f"  🔍 Mask 数组形状: {mask_araay.shape}")
    print(f"  🔍 Mask 数据类型: {mask_araay.dtype}")

    # 确保 mask 是正确的格式
    # OpenCV 返回的是灰度图 (H, W)，需要转换为 PIL 的 'L' 模式
    if len(mask_araay.shape) == 2:
      # 灰度图，直接使用 'L' 模式
      mask_image = Image.fromarray(mask_araay, mode='L')
    else:
      # 如果是彩色图，转换为灰度
      mask_image = Image.fromarray(mask_araay).convert('L')

    print(f"  🔍 转换后 Mask 尺寸: {mask_image.size}")

    # 关键修复：确保 mask 和图片尺寸完全一致
    # PIL 的 size 是 (宽, 高)，需要和图片完全匹配
    if mask_image.size != img.size:
      print(f"  ⚠️  调整 mask 尺寸: {mask_image.size} → {img.size}")
      mask_image = mask_image.resize(img.size, Image.LANCZOS)

    print(f"  ✅ 最终 Mask 尺寸: {mask_image.size}")

    # 保存 mask 用于调试
    if SAVE_MASK:
      mask_debug_path = str(output_path).replace('.jpg', '_mask.png').replace('.png', '_mask.png').replace('.jpeg', '_mask.png')
      mask_image.save(mask_debug_path)
      print(f"  💾 Mask 已保存到: {mask_debug_path}")

    # 创建可视化图片：将 mask 叠加到原图上（红色标记）
    if SAVE_PREVIEW:
      preview_path = str(output_path).replace('.jpg', '_preview.png').replace('.png', '_preview.png').replace('.jpeg', '_preview.png')
      preview_img = img.copy()
      preview_array = np.array(preview_img)
      mask_array_preview = np.array(mask_image)

      # 将 mask 区域标记为红色半透明
      preview_array[mask_array_preview > 0] = (
        preview_array[mask_array_preview > 0] * 0.5 +
        np.array([255, 0, 0]) * 0.5
      ).astype(np.uint8)

      Image.fromarray(preview_array).save(preview_path)
      print(f"  👁️  预览图已保存到: {preview_path}")

    # 关键修复：iopaint 模型需要 numpy 数组，不是 PIL Image！
    # 将 PIL Image 转换为 numpy 数组
    # iopaint 会自动处理 RGB，不需要手动转换
    img_array = np.array(img)  # PIL 默认是 RGB，保持原样
    mask_array = np.array(mask_image)

    print(f"  🔍 转换为 numpy - 图片: {img_array.shape}, Mask: {mask_array.shape}")

    from iopaint.schema import HDStrategy, InpaintRequest
    config = InpaintRequest(
      ldm_steps=20,  # 推理步数,越大效果越好但越慢
      hd_strategy='Original',
      enable_controlnet=False,
    )

    # 传入 RGB 数组 (iopaint 期望 RGB 输入)
    result = model(image=img_array, mask=mask_array, config=config)

    print(f"  🔍 结果数组类型: {result.dtype}, 形状: {result.shape}")

    # 确保是 uint8 类型
    if result.dtype != np.uint8:
      result = np.clip(result, 0, 255).astype(np.uint8)

    # ⚠️ 重要：iopaint 返回 BGR 格式，需要转换为 RGB
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(result_rgb, mode='RGB')

    # 优化保存参数，避免文件变大
    if str(output_path).lower().endswith('.jpg') or str(output_path).lower().endswith('.jpeg'):
      result_image.save(output_path, 'JPEG', quality=95, optimize=True)
    else:
      result_image.save(output_path, optimize=True)

    return True
  except Exception as e:
    print(f"❌ 处理失败: {e}")
    # 打印详细的错误堆栈信息
    import traceback
    traceback.print_exc()
    return False
  
def batch_process(use_template=False):
    """
    批量处理所有图片（支持一层子文件夹结构）

    类比 JS:
    async function batchProcess() {
        const files = fs.readdirSync('./input');
        for (const file of files) {
            await removeWatermark(file);
            console.log(`✅ ${file} 完成`);
        }
    }
    """
    # 1. 初始化模型
    model = init_model()

    # 2. 获取所有图片文件
    input_path = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)

    # 创建输出目录
    output_path.mkdir(exist_ok=True)

    # 边扫描边处理，显示实时进度
    print("🔍 开始扫描并处理图片文件...\n")

    success_count = 0
    processed_count = 0
    failed_count = 0

    # 定义图片扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png']

    # 处理根目录的图片
    for ext in image_extensions:
        for img_file in input_path.glob(ext):
            processed_count += 1
            relative_path = img_file.relative_to(input_path)
            print(f"[{processed_count}] 处理: {relative_path}")

            # 输出文件保持相同的文件夹结构
            output_file = output_path / relative_path

            # 创建子文件夹（如果需要）
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # 去水印
            if remove_watermark(model, img_file, output_file, use_template):
                success_count += 1
                print(f"✅ 完成\n")
            else:
                failed_count += 1
                print(f"❌ 失败\n")

    # 处理一层子文件夹中的图片
    for ext in image_extensions:
        for img_file in input_path.glob(f'*/{ext}'):
            processed_count += 1
            relative_path = img_file.relative_to(input_path)
            print(f"[{processed_count}] 处理: {relative_path}")

            # 输出文件保持相同的文件夹结构
            output_file = output_path / relative_path

            # 创建子文件夹（如果需要）
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # 去水印
            if remove_watermark(model, img_file, output_file, use_template):
                success_count += 1
                print(f"✅ 完成\n")
            else:
                failed_count += 1
                print(f"❌ 失败\n")

    # 4. 统计结果
    if processed_count == 0:
        print(f"❌ 在 {INPUT_FOLDER} 中没有找到图片")
        return

    print("=" * 50)
    print(f"🎉 处理完成!")
    print(f"📊 总计: {processed_count} 张")
    print(f"✅ 成功: {success_count} 张")
    print(f"❌ 失败: {failed_count} 张")
    print(f"📁 输出目录: {OUTPUT_FOLDER}")

# === 主程序入口 ===
if __name__ == "__main__":
    """
    类比 JS:
    if (require.main === module) {
        main();
    }
    """
    print("🚀 批量去水印工具启动\n")

    # 检查是否使用模板匹配
    use_template = False
    if len(sys.argv) > 1 and sys.argv[1] == "--template":
        use_template = True
        print("📋 使用固定位置检测模式\n")

    # 检查是否需要重新检测水印位置
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        print("🔄 重置水印位置信息\n")
        positions_file = Path("./template/watermark_positions.txt")
        if positions_file.exists():
            positions_file.unlink()
            print("✅ 已删除旧的位置信息，下次运行将重新检测\n")
        use_template = True

    # 执行批量处理
    batch_process(use_template)