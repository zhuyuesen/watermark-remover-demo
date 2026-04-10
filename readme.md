# 批量移除水印

前提：
多张图片的水印位置、角度一致:
1. 使用 https://www.robots.ox.ac.uk/~vgg/software/via/via.html 标注出水印位置，左上角 Annotation -> Export Annotations (as json) 保存水印位置文件到项目
2. 使用 convert_via_to_positions.py 把上述文件转为 template/watermark_positions.txt
3. 使用mac自带图片软件导出水印大概图片 到 template/watermark.png
4. 使用 remove.py 批量移除水印
5. 使用mac预览软件导出水印图片可能会失败，可使用 GIMP 软件操作。步骤:
    - 用模糊选择工具选中水印
    - 复制选区：Edit > Copy（Command+C）  
    - 粘贴为新图像：Edit > Paste as > New Image（粘贴为新图像）
    - 然后 File > Export As 导出保存为 PNG/JPG

## 环境配置

### 1. 安装 Python
确保已安装 Python 3.8 或更高版本：
```bash
python3 --version
```

### 2. 安装项目依赖
```bash
pip3 install -r requirements.txt
```

或使用虚拟环境（推荐）：
```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 设备配置
在 [remove.py](remove.py#L15) 中根据你的设备修改 `DEVICE` 参数：
- Mac M 芯片：`DEVICE = "mps"`
- NVIDIA GPU：`DEVICE = "cuda"`
- CPU：`DEVICE = "cpu"`

### 4. 目录配置
windows中可以配置如下输入输出目录  
1. 转义反斜杠
```py
INPUT_FOLDER = "D:\\imgs\\chuanxunwang"
```
2. 使用原始字符串
```py
INPUT_FOLDER = r"D:\imgs\chuanxunwang"
```
3. 使用正斜杠  
Python 和 pathlib 在 Windows 上也支持正斜杠:
```py
INPUT_FOLDER = "D:/imgs/chuanxunwang"
```

## 操作步骤
1. 把要移除水印的图片放入 [input](input/) 目录 或配置输入输出目录
2. 运行 `python3 ./remove.py --template`
3. 处理后的图片会保存在 [output](output/) 目录


## 移除多余的 _mask_mask.png 和 _preview_preview.png 文件
1. 预览模式（默认，不会删除文件）:  
`python cleanup_duplicates.py /path/to/directory`
2. 确认删除:  
`python cleanup_duplicates.py /path/to/directory --confirm`
3. 清理当前目录:  
`python cleanup_duplicates.py . --confirm`

`/path/to/directory` 为路径，如 `C:\watermark-remover\output`

claude命令: 移除水印2
remove.py 是现有已完成的水印移除脚本。
但是现在发现新的需求: 有一些分辨率为500*352的图片(类型A)的水印与其他图片是不一样的，导致使用remove.py移除水印失败
已知A类图片每张图片上只有2个水印，水印1对应wm01.png 水印2对应wm02.png
现在我在template2/watermark_positions.txt中记录了2个水印的位置，位置1对应wm01.png 位置2对应wm02.png
新建一个脚本，从配置参数的输入目录中查找分辨率为500*352的图片并根据位置和水印尝试移除水印并保存到输出目录

## 运行 remove2.py 步骤
要用 3.11 的 pip
1. py -3.11 -m pip install --upgrade iopaint diffusers transformers huggingface_hub

2. py -3.11 .\remove2.py


remove.py和remove2.py是现有2个版本的水印移除工具
现有新的水印移除需求：
现在有一种水印图片：左下角有2行水印：第一行是所有图片相同的一个公司名的文字，第二行是各自摄影师的名字。这种水印如何去除
已知2行水印的开始位置都相同、第二行文字不一样所以宽度也不一样
我已把第一行的水印和位置保存到template3

现在已知需要这么处理的图片的分辨率宽都是800，增加这个限制以加速处理图片