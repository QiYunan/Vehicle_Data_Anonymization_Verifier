"""
统一检测启动器（GB/T 44464-2024）

弹出文件选择框，自由勾选**照片 + 视频（可混选、可多选）**，只处理选中的文件：
    视频 → 调脚本一抽帧（每 2 秒 1 帧）→ 帧图
    照片 → 直接采用
汇入同一个 run_N_时间戳 目录后 → 调脚本二做人脸/车牌检测，出 JSON + 可视化框图。

解决的痛点：以前的 bat 要么全量处理某文件夹所有照片、要么所有视频，无法挑选。
现在一个入口、混选、按需检测。

输出结构：
    test\detection_result\run_N_时间戳\
        images\photos\        选中的照片（原样复制）
        images\<视频名>\       该视频抽出的帧
        json\                 脚本二检测 JSON（镜像 images 结构）
        visualization\        画框图
"""

import os
import sys
import shutil
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FFMPEG = r"E:\FFmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
PYTHON = sys.executable
RESULT_BASE = r"E:\Vehicle_Data_Anonymization_Verifier\test\detection_result"
SAMPLE_DIR = r"E:\Vehicle_Data_Anonymization_Verifier\test\sample"

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VID_EXT = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")

_FILETYPES = [
    ("照片和视频", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mkv *.mov *.flv *.wmv"),
    ("照片", "*.jpg *.jpeg *.png *.bmp *.webp"),
    ("视频", "*.mp4 *.avi *.mkv *.mov *.flv *.wmv"),
    ("所有文件", "*.*"),
]


def _set_dpi_aware():
    """开启进程 DPI 感知，使原生文件框清晰不发糊（须在创建任何窗口前调用）。"""
    import ctypes
    for fn in (lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),   # 每显示器 DPI 感知
               lambda: ctypes.windll.user32.SetProcessDPIAware()):        # 退路：系统 DPI 感知
        try:
            fn()
            return
        except Exception:
            continue


def pick_files():
    """弹原生多选文件框，支持「多轮累加」跨文件夹选择（照片+视频混选）。
    Windows 文件框单次只能在一个文件夹内多选，故选完一批后可继续去别处再选。"""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    _set_dpi_aware()
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    initial = SAMPLE_DIR if os.path.isdir(SAMPLE_DIR) else None

    chosen = []
    while True:
        sel = filedialog.askopenfilenames(
            parent=root,
            title="勾选要检测的照片/视频（Ctrl/Shift 多选）— 可多轮跨文件夹添加",
            initialdir=initial,
            filetypes=_FILETYPES,
        )
        chosen.extend(sel)
        if not messagebox.askyesno(
            "继续添加？",
            f"已选 {len(set(chosen))} 个文件。\n是否再从其他文件夹继续添加？\n（选「否」即开始检测）",
            parent=root,
        ):
            break
    root.destroy()
    # 去重并保持先后顺序
    return list(dict.fromkeys(chosen))


def next_run_dir():
    """生成下一个 run_N_时间戳 目录路径。"""
    os.makedirs(RESULT_BASE, exist_ok=True)
    max_n = 0
    for name in os.listdir(RESULT_BASE):
        if name.startswith("run_") and os.path.isdir(os.path.join(RESULT_BASE, name)):
            parts = name.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                max_n = max(max_n, int(parts[1]))
    return os.path.join(RESULT_BASE, f"run_{max_n + 1}_{datetime.now().strftime('%Y%m%d_%H%M')}")


def _safe_dirname(name):
    """把文件名转成安全的子目录名。"""
    keep = "".join(c for c in name if c not in r'\/:*?"<>|').strip()
    return keep or "video"


def main():
    print("=" * 60)
    print(" GB/T 44464-2024 · 统一检测启动器")
    print("=" * 60)
    print("请在弹出的窗口中勾选要检测的照片/视频...")

    files = pick_files()
    if not files:
        print("未选择任何文件，已取消。")
        return

    imgs = [f for f in files if f.lower().endswith(IMG_EXT)]
    vids = [f for f in files if f.lower().endswith(VID_EXT)]
    other = [f for f in files if f not in imgs and f not in vids]
    if other:
        print(f"忽略 {len(other)} 个不支持的文件：" + ", ".join(os.path.basename(o) for o in other))
    if not imgs and not vids:
        print("选中的文件里没有可处理的照片或视频。")
        return
    print(f"选中：{len(imgs)} 张照片，{len(vids)} 个视频。")

    run_dir = next_run_dir()
    images_dir = os.path.join(run_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # 照片 → images\photos\（原样复制，防重名）
    if imgs:
        photos_dir = os.path.join(images_dir, "photos")
        os.makedirs(photos_dir, exist_ok=True)
        for src in imgs:
            base, ext = os.path.splitext(os.path.basename(src))
            dst = os.path.join(photos_dir, base + ext)
            k = 1
            while os.path.exists(dst):
                dst = os.path.join(photos_dir, f"{base}_{k}{ext}")
                k += 1
            shutil.copy2(src, dst)
        print(f"已收入 {len(imgs)} 张照片 → {photos_dir}")

    # 视频 → images\<视频名>\（逐个抽帧）
    if vids:
        from gbt_step1_splitter import process_single_video
        print("\n开始抽帧...")
        for i, v in enumerate(vids, 1):
            stem = _safe_dirname(os.path.splitext(os.path.basename(v))[0])
            sub = os.path.join(images_dir, stem)
            os.makedirs(sub, exist_ok=True)
            process_single_video(FFMPEG, v, sub, index=i)

    # 检测（复用脚本二，沿用现有命令行接口）
    json_dir = os.path.join(run_dir, "json")
    viz_dir = os.path.join(run_dir, "visualization")
    print("\n" + "=" * 60)
    print(" 开始检测（首次会加载模型，请耐心等待）...")
    print("=" * 60)
    ret = subprocess.run([
        PYTHON, os.path.join(HERE, "gbt_step2_detector.py"),
        "--input", images_dir, "--output-json", json_dir, "--output-viz", viz_dir,
    ])

    print("\n" + "=" * 60)
    if ret.returncode == 0:
        print(" 完成！结果目录：")
        print(f"   {run_dir}")
        print(f"   - 检测 JSON     : {json_dir}")
        print(f"   - 可视化框图    : {viz_dir}")
    else:
        print(f" 检测脚本返回非 0（{ret.returncode}），请看上方报错。")
    print("=" * 60)


if __name__ == "__main__":
    main()
