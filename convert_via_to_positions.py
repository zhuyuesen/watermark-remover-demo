"""
将 VIA (VGG Image Annotator) 标注文件转换为水印位置文件
"""

import json
from pathlib import Path

def convert_via_to_positions(via_json_path, output_path):
    """
    读取 VIA JSON 文件，提取矩形标注，生成位置文件

    :param via_json_path: VIA JSON 文件路径
    :param output_path: 输出的位置文件路径
    """
    # 读取 VIA JSON 文件
    with open(via_json_path, 'r') as f:
        via_data = json.load(f)

    # 遍历所有图片的标注
    all_positions = []

    for image_key, image_data in via_data.items():
        filename = image_data.get('filename', 'unknown')
        regions = image_data.get('regions', [])

        print(f"📷 图片: {filename}")
        print(f"   找到 {len(regions)} 个标注区域")

        positions = []
        for region in regions:
            shape_attrs = region.get('shape_attributes', {})
            if shape_attrs.get('name') == 'rect':
                x = shape_attrs.get('x')
                y = shape_attrs.get('y')
                w = shape_attrs.get('width')
                h = shape_attrs.get('height')

                positions.append((x, y, w, h))
                print(f"   - 位置: x={x}, y={y}, w={w}, h={h}")

        all_positions.extend(positions)

    # 保存到位置文件
    with open(output_path, 'w') as f:
        for x, y, w, h in all_positions:
            f.write(f"{x},{y},{w},{h}\n")

    print(f"\n✅ 成功！已保存 {len(all_positions)} 个水印位置到: {output_path}")
    return len(all_positions)

if __name__ == "__main__":
    via_json = "./via_project_7Jan2026_10h26m_json.json"
    output = "./template/watermark_positions.txt"

    # 确保 template 目录存在
    Path("./template").mkdir(exist_ok=True)

    count = convert_via_to_positions(via_json, output)

    print("\n" + "="*50)
    print("🎉 转换完成！")
    print(f"📊 总共标注了 {count} 个水印位置")
    print(f"📁 位置文件: {output}")
    print("\n💡 下一步：运行以下命令去除水印")
    print("   python3 ./remove.py --template")
    print("="*50)
