# 批量移除水印

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