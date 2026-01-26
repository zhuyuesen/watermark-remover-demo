#!/usr/bin/env python3
"""
清理多余的 _mask_mask.png 和 _preview_preview.png 文件
"""
import os
import sys
import argparse
from pathlib import Path


def cleanup_duplicate_files(directory, dry_run=False):
    """
    清理指定目录下的 _mask_mask.png 和 _preview_preview.png 文件

    Args:
        directory: 要清理的目录路径
        dry_run: 如果为True，只显示将要删除的文件，不实际删除
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"错误: 目录不存在 - {directory}")
        return

    if not directory.is_dir():
        print(f"错误: 不是一个目录 - {directory}")
        return

    # 要删除的文件模式
    patterns = ['*_mask_mask.png', '*_preview_preview.png']

    deleted_count = 0
    total_size = 0

    print(f"正在扫描目录: {directory}")
    print("-" * 60)

    for pattern in patterns:
        # 递归查找所有匹配的文件
        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                total_size += file_size

                if dry_run:
                    print(f"[预览] 将删除: {file_path} ({file_size / 1024:.2f} KB)")
                    deleted_count += 1
                else:
                    try:
                        file_path.unlink()
                        print(f"[已删除] {file_path} ({file_size / 1024:.2f} KB)")
                        deleted_count += 1
                    except Exception as e:
                        print(f"[错误] 无法删除 {file_path}: {e}")

    print("-" * 60)
    if dry_run:
        print(f"预览模式: 找到 {deleted_count} 个文件，总大小 {total_size / 1024:.2f} KB")
        print("使用 --confirm 参数确认删除")
    else:
        print(f"完成: 删除了 {deleted_count} 个文件，释放空间 {total_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(
        description='清理多余的 _mask_mask.png 和 _preview_preview.png 文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览将要删除的文件
  python cleanup_duplicates.py /path/to/directory

  # 确认删除
  python cleanup_duplicates.py /path/to/directory --confirm
        """
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='要清理的目录路径 (默认: 当前目录)'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='确认删除文件 (默认只预览)'
    )

    args = parser.parse_args()

    cleanup_duplicate_files(args.directory, dry_run=not args.confirm)


if __name__ == '__main__':
    main()
