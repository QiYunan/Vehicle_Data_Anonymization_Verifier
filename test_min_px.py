"""
车牌识别最小像素阈值测试
测试方法：合成标准中国车牌图（蓝底白字），按真实长宽比(3.14:1)缩放到
不同最小边长(高度)，逐一跑 PaddleOCR，记录识别结果与置信度。
"""
import os, sys, re

# ── E 盘缓存重定向（与脚本二保持一致）──────────────────────────────────────
_CACHE_BASE = r"E:\Vehicle_Data_Anonymization_Verifier\model_cache"
os.environ.setdefault("HF_HOME",          os.path.join(_CACHE_BASE, "huggingface"))
os.environ.setdefault("TORCH_HOME",       os.path.join(_CACHE_BASE, "torch"))
os.environ.setdefault("PADDLE_HOME",      os.path.join(_CACHE_BASE, "paddle"))
os.environ.setdefault("PADDLEOCR_HOME",   os.path.join(_CACHE_BASE, "paddleocr"))
os.environ["XDG_CACHE_HOME"] =            os.path.join(_CACHE_BASE, "xdg")
os.environ["PADDLEX_HOME"] =              os.path.join(_CACHE_BASE, "paddlex")
# 跳过模型来源网络检查（模型已在本地缓存，无需联网）
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 车牌格式校验（与脚本二一致）──────────────────────────────────────────────
_PLATE_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
_PLATE_SUFFIX    = "港澳学警挂领使试"
_PLATE_RE = re.compile(
    rf"^[{_PLATE_PROVINCES}][A-Z][A-Z0-9]{{4,6}}[A-Z0-9{_PLATE_SUFFIX}]$"
)

def normalize_plate(text):
    return re.sub(r"[\s·•・.\-]", "", text or "").upper()

def is_valid_plate(text):
    return bool(_PLATE_RE.match(normalize_plate(text)))


# ── 合成车牌图（蓝底白字，标准比例） ─────────────────────────────────────────
def make_plate_image(plate_text: str, height_px: int) -> np.ndarray:
    """
    按中国标准蓝牌比例（440mm×140mm → 宽:高 ≈ 3.14:1）合成车牌图。
    返回 BGR ndarray，高度 = height_px。
    """
    # 先在高分辨率画，再缩到目标尺寸
    BASE_H = 200
    BASE_W = int(BASE_H * 3.14)
    img = Image.new("RGB", (BASE_W, BASE_H), color=(30, 90, 170))  # 蓝底

    draw = ImageDraw.Draw(img)
    font_size = int(BASE_H * 0.62)
    font = None
    for path in (r"C:\Windows\Fonts\msyh.ttc",
                 r"C:\Windows\Fonts\simhei.ttf",
                 r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), plate_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (BASE_W - tw) // 2
    y = (BASE_H - th) // 2
    draw.text((x, y), plate_text, font=font, fill=(255, 255, 255))

    # 缩到目标高度
    target_w = int(height_px * 3.14)
    img = img.resize((target_w, height_px), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── 初始化 PaddleOCR（只创建一次） ────────────────────────────────────────────
print("正在加载 PaddleOCR……")
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    lang="ch",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    enable_mkldnn=False,
)
print("PaddleOCR 加载完成。\n")


# ── 测试用车牌号 ───────────────────────────────────────────────────────────────
TEST_PLATES = ["京A12345", "沪B88888", "粤A·12345"]

# 测试的最小边长序列（px），从大到小
TEST_HEIGHTS = [128, 96, 64, 48, 40, 32, 28, 24, 20, 18, 16, 14, 12, 10, 8]

OCR_CONF_THRESHOLD = 0.80  # 与脚本二保持一致

# ── 保存测试图（可视化检查用） ─────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_min_px_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 跑测试 ─────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"{'高度(px)':>8}  {'宽度(px)':>8}  {'原始车牌':>12}  {'OCR结果':>15}  {'置信度':>6}  {'状态':>10}")
print("=" * 70)

results = []  # (height, plate_text, ocr_text, conf, success)

for plate_text in TEST_PLATES:
    clean_plate = normalize_plate(plate_text)  # 去掉·用于校验
    print(f"\n── 车牌: {plate_text} (标准化: {clean_plate}) ──")
    first_fail = None

    for h in TEST_HEIGHTS:
        img = make_plate_image(plate_text, h)
        w = img.shape[1]

        # 保存图片
        fname = f"plate_{clean_plate}_h{h:03d}.jpg"
        cv2.imwrite(os.path.join(OUT_DIR, fname), img)

        # OCR
        try:
            result = ocr.predict(img)
            texts  = result[0].get("rec_texts", []) if result else []
            scores = result[0].get("rec_scores", []) if result else []
            ocr_text = "".join(texts) if texts else ""
            conf     = round(float(min(scores)), 4) if scores else 0.0
        except Exception as e:
            ocr_text, conf = f"[ERR:{e}]", 0.0

        # 判定
        valid = bool(ocr_text) and conf >= OCR_CONF_THRESHOLD and is_valid_plate(ocr_text)
        status = "✅ 识别成功" if valid else "❌ 识别失败"
        if not valid and first_fail is None:
            first_fail = h

        print(f"  {h:>6}px  {w:>6}px  {plate_text:>12}  {ocr_text:>15}  {conf:>6.4f}  {status}")
        results.append((plate_text, h, w, ocr_text, conf, valid))

    if first_fail:
        print(f"  ★ 「{plate_text}」首次失败: 高度 {first_fail}px（宽约{int(first_fail*3.14)}px）")
    else:
        print(f"  ★ 「{plate_text}」在所有测试尺寸下均可识别")

# ── 汇总 ───────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("汇总：各车牌识别临界点")
print("=" * 70)
for plate_text in TEST_PLATES:
    plate_results = [(h, w, t, c, v) for p, h, w, t, c, v in results if p == plate_text]
    successes = [h for h, _, _, _, v in plate_results if v]
    failures  = [h for h, _, _, _, v in plate_results if not v]
    if successes and failures:
        min_ok  = min(successes)
        max_fail = max(failures)
        print(f"  {plate_text}: 最小可识别高度 = {min_ok}px（宽约{int(min_ok*3.14)}px）| "
              f"首次失败高度 = {max(failures)}px")
    elif not failures:
        print(f"  {plate_text}: 在所有测试尺寸（最小 {TEST_HEIGHTS[-1]}px）下均可识别")
    else:
        print(f"  {plate_text}: 所有尺寸均识别失败（最大 {TEST_HEIGHTS[0]}px 也失败）")

print(f"\n测试图片已保存至: {OUT_DIR}")
print("说明：高度 = 车牌框最小边长（min_side）；宽度 = 高度×3.14（标准蓝牌比例）。")
print("      OCR置信度阈值 = 0.80（与脚本二 ocr_conf_threshold 一致）。")
