"""
脚本二 · 识别检测（GB/T 44464-2024 汽车数据匿名化合规检测系统）

职责：对脚本一抽取的 1080P 图片集做车牌 / 人脸检测，
      输出带坐标、置信度、识别内容的标准化 JSON，并生成可视化图。

检测模型栈（方案A · 准确率优先，全离线）：
    人脸检测  ->  RetinaFace
    车牌定位  ->  YOLOv8 (license-plate)  [主检测器]
    车牌识别  ->  PaddleOCR (中文车牌)
    大角度车牌 -> we0091234 (YOLOv5+关键点+透视矫正+CRNN)  [副检测器，兜底]

OpenCV 仅用于图像 I/O、坐标绘制与结果可视化。
"""

import os
import sys

# 将 we0091234 的中文车牌识别仓库加入 Python 路径（副检测器依赖）
_CHINESE_PLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "chinese_plate_recognition")
if _CHINESE_PLATE_DIR not in sys.path:
    sys.path.insert(0, _CHINESE_PLATE_DIR)

# ======================================================================
# ⚠️ 强制缓存重定向（必须在导入任何 AI 框架之前执行）
# ----------------------------------------------------------------------
# 本机 C 盘已爆满：严禁任何模型权重 / 缓存写入 C 盘。下面把所有 AI 框架、
# 深度学习库的 HOME / CACHE 路径全部强行重定向到 E 盘项目目录。
# 用 environ[...] = 直接赋值（不是 setdefault），确保即使外部已设也被覆盖到 E 盘。
# ======================================================================
_MODEL_CACHE = r"E:\Vehicle_Data_Anonymization_Verifier\model_cache"


def _redirect_caches_to_e():
    sub = lambda *p: os.path.join(_MODEL_CACHE, *p)
    env = {
        # —— RetinaFace / DeepFace ——（权重 ~114MB）
        "DEEPFACE_HOME": sub("deepface"),
        # —— PaddleOCR / PaddleX ——（OCR 检测+识别+方向模型）
        "PADDLE_PDX_CACHE_HOME": sub("paddlex"),
        "PADDLE_PDX_MODEL_SOURCE": "BOS",          # 国内走百度 BOS 源
        # —— Ultralytics / YOLOv8 ——（settings.json + 下载的权重）
        "YOLO_CONFIG_DIR": sub("ultralytics"),
        # —— PyTorch ——（torch.hub / 预训练权重）
        "TORCH_HOME": sub("torch"),
        # —— TensorFlow / Keras ——
        "TF_USE_LEGACY_KERAS": "1",                # RetinaFace 需 Keras2 接口
        "KERAS_HOME": sub("keras"),
        "TFHUB_CACHE_DIR": sub("tfhub"),
        # —— HuggingFace ——（走镜像 + 缓存落 E 盘）
        "HF_HOME": sub("huggingface"),
        "HUGGINGFACE_HUB_CACHE": sub("huggingface", "hub"),
        "HF_ENDPOINT": "https://hf-mirror.com",
        # —— ModelScope ——（国内模型库）
        "MODELSCOPE_CACHE": sub("modelscope"),
        # —— matplotlib / 通用 XDG / 临时目录 ——
        "MPLCONFIGDIR": sub("matplotlib"),
        "XDG_CACHE_HOME": sub("xdg_cache"),
        "TMP": sub("tmp"),
        "TEMP": sub("tmp"),
    }
    for key, val in env.items():
        os.environ[key] = val          # 强制覆盖，不留余地
        if not val.startswith("http") and val not in ("BOS", "1"):
            os.makedirs(val, exist_ok=True)


_redirect_caches_to_e()

import re
import sys
import json
import traceback
from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np

# Windows 控制台默认 GBK，打印车牌中文/✅ 会报错；强制 stdout 用 UTF-8。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# Windows 下 cv2.imread/imwrite 无法处理含中文（非 ASCII）的路径，
# 而脚本一按视频名建子目录，常含中文。统一用以下两个 Unicode 安全函数。
def imread_unicode(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img_matrix):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img_matrix)
    if ok:
        buf.tofile(path)
    return ok


# cv2.putText 画不了中文（车牌号），用 PIL + Windows 中文字体渲染。
@lru_cache(maxsize=8)
def _load_cn_font(size):
    from PIL import ImageFont
    for path in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                 r"C:\Windows\Fonts\simsun.ttc"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ---- 中国车牌格式校验 ----
# 首字为大陆省份简称（不含台/港/澳——港澳车进大陆挂「粤Z+尾字港/澳」）。
# 末位允许特殊尾缀汉字：港澳(入境)、学(教练)、警(警车)、挂(挂车)、领/使(领馆)、试(试验)。
_PLATE_PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
_PLATE_SUFFIX = "港澳学警挂领使试"
# 省1 + 字母1 + 4~6位字母数字 + 末位(字母数字 或 尾缀汉字) → 普通7位/新能源8位/带尾缀
_PLATE_RE = re.compile(
    rf"^[{_PLATE_PROVINCES}][A-Z][A-Z0-9]{{4,6}}[A-Z0-9{_PLATE_SUFFIX}]$"
)


def normalize_plate(text):
    """去掉分隔符·•・.- 与空格，字母转大写。"""
    return re.sub(r"[\s·•・.\-]", "", text or "").upper()


def is_valid_plate(text):
    """是否符合中国车牌格式（用于剔除 GB/T 标准号等误检、残缺读数）。"""
    return bool(_PLATE_RE.match(normalize_plate(text)))


def _overlap_ratio(a, b):
    """交集面积 / 较小框面积（IoMin）。用于去重：一个框大部分落在另一个里即视为重复。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def draw_labels_cn(canvas_bgr, labels, font_size=22):
    """在 BGR 图上批量画中文标签。labels: [(text, x, y, (B,G,R)), ...]，返回新 BGR 图。"""
    if not labels:
        return canvas_bgr
    from PIL import Image, ImageDraw
    img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _load_cn_font(font_size)
    for text, x, y, (b, g, r) in labels:
        # 文字加深色描边底，避免在浅色车身上看不清
        draw.text((x, y), text, font=font, fill=(r, g, b),
                  stroke_width=2, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ======================================================================
# 一、人脸检测器（RetinaFace）
# ======================================================================
class FaceDetector:
    """基于 RetinaFace 的离线人脸检测。延迟加载，未安装时给出清晰提示。"""

    def __init__(self, conf_threshold=0.65):
        self.conf_threshold = conf_threshold
        self._engine = None

    def _ensure_engine(self):
        if self._engine is not None:
            return
        try:
            from retinaface import RetinaFace
        except ImportError as e:
            raise RuntimeError(
                "未安装 RetinaFace。请执行: pip install retina-face\n"
                "（需 Python<=3.12 与 tensorflow 支持）"
            ) from e
        self._engine = RetinaFace

    # 几何后处理过滤阈值（不改模型，只过滤形态异常的误检框）
    _MIN_FACE_PX      = 20      # 最小边长：<20px 必为噪点
    _MAX_AREA_RATIO   = 0.08    # 最大面积占比：>8% 画面视为异常（约407×407px@1080p）
    _HIGH_CONF_EXEMPT = 0.90    # 置信度≥0.90时豁免面积检查，保留极近距离真实人脸
    _ASPECT_MIN       = 0.4     # 宽/高下限：过窄的条状框不是人脸
    _ASPECT_MAX       = 1.5     # 宽/高上限：人脸宽不超过高的1.5倍
    _GROUND_Y_RATIO   = 0.90    # 框中心y > 90%画面高度 → 地面区域，排除

    def _is_plausible_face(self, x1, y1, x2, y2, conf, img_h, img_w):
        """几何合理性校验，排除停车场地面/纹理等环境误检。
        置信度≥0.90时豁免面积上限，保留极近距离的真实人脸。"""
        w, h = x2 - x1, y2 - y1
        if min(w, h) < self._MIN_FACE_PX:
            return False
        if (w * h) / (img_h * img_w) > self._MAX_AREA_RATIO and conf < self._HIGH_CONF_EXEMPT:
            return False
        aspect = w / h if h > 0 else 0
        if not (self._ASPECT_MIN <= aspect <= self._ASPECT_MAX):
            return False
        if (y1 + y2) / 2 > img_h * self._GROUND_Y_RATIO:
            return False
        return True

    def detect(self, img_matrix):
        """返回人脸列表: [{bbox:[x1,y1,x2,y2], confidence:float}, ...]"""
        self._ensure_engine()
        img_h, img_w = img_matrix.shape[:2]
        raw = self._engine.detect_faces(img_matrix, threshold=self.conf_threshold)
        faces = []
        if isinstance(raw, dict):
            for info in raw.values():
                x1, y1, x2, y2 = info["facial_area"]
                conf = round(float(info.get("score", 1.0)), 4)
                if not self._is_plausible_face(x1, y1, x2, y2, conf, img_h, img_w):
                    continue
                faces.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                })
        return faces


# ======================================================================
# 二、车牌定位 + 识别（YOLOv8 + PaddleOCR）
# ======================================================================
class PlateDetector:
    """YOLOv8 定位车牌区域，PaddleOCR 识别车牌文字。"""

    # 国标 §5.6.2.1：车牌边界框最小边长 ≥16px 才属匿名化对象。
    MIN_SIDE_PX = 16

    # 旋转暴力搜索角度（0° 已在正向 OCR 中尝试过，此处只补偿倾斜）
    _ROTATION_ANGLES = (-30, -20, -10, 10, 20, 30)

    def __init__(self, yolo_weights, conf_threshold=0.15, ocr_lang="ch",
                 imgsz=1920, ocr_conf_threshold=0.80, min_side_px=MIN_SIDE_PX,
                 enable_rotation_search=True):
        """
        conf_threshold     : YOLO 置信度门槛（调低以多召回远处小/斜车牌）
        imgsz              : YOLO 推理分辨率，默认 1920（YOLO 默认仅 640 会把远处小牌缩没）。
                             实测：太高(如3200)反而会漏掉近处「过大」的车牌，1920 兼顾近/远，
                             也正好匹配 1080p 抽帧工作流。远处小牌根治靠全分辨率源图。
        ocr_conf_threshold : OCR 文字置信度 ≥ 此值才算「读出有效号码」
        min_side_px        : 车牌最小边长阈值（国标 16px），用于第②层「是否符合国标」筛选
        enable_rotation_search : 对 unread 框启用旋转暴力搜索（-30°~+30°，10°步长）
        """
        self.yolo_weights = yolo_weights
        self.conf_threshold = conf_threshold
        self.ocr_lang = ocr_lang
        self.imgsz = imgsz
        self.ocr_conf_threshold = ocr_conf_threshold
        self.min_side_px = min_side_px
        self.enable_rotation_search = enable_rotation_search
        self._yolo = None
        self._ocr = None

    def _ensure_yolo(self):
        if self._yolo is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "未安装 ultralytics。请执行: pip install ultralytics"
            ) from e
        if not os.path.exists(self.yolo_weights):
            raise RuntimeError(f"未找到 YOLOv8 车牌权重文件: {self.yolo_weights}")
        self._yolo = YOLO(self.yolo_weights)

    def _ensure_ocr(self):
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise RuntimeError(
                "未安装 PaddleOCR。请执行: pip install paddlepaddle paddleocr\n"
                "（PaddlePaddle 当前不支持 Python 3.14，请使用 3.11/3.12 虚拟环境）"
            ) from e
        # PaddleOCR 3.x 新接口：车牌为单行紧凑文本，关闭文档方向/扭曲矫正，
        # 仅保留检测+识别，速度更快、误判更少。
        # enable_mkldnn=False：规避 PaddlePaddle 3.3.1 CPU oneDNN 在新 PIR 执行器下
        # 的 ConvertPirAttribute2RuntimeAttribute 崩溃。
        self._ocr = PaddleOCR(
            lang=self.ocr_lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )

    def _recognize_text(self, plate_crop):
        """对车牌裁切图做 OCR，返回 (文字, 文字置信度)。"""
        self._ensure_ocr()
        if plate_crop.size == 0:
            return "", 0.0
        # PaddleOCR 3.x：predict 返回 [OCRResult]，含 rec_texts / rec_scores 字段
        result = self._ocr.predict(plate_crop)
        if not result:
            return "", 0.0
        res = result[0]
        texts = res.get("rec_texts", []) or []
        scores = res.get("rec_scores", []) or []
        if not texts:
            return "", 0.0
        # 车牌通常单行，拼接所有识别片段，取最低置信度作为整体置信度
        full_text = "".join(texts)
        text_conf = round(float(min(scores)), 4) if scores else 0.0
        return full_text, text_conf

    @staticmethod
    def _rotate_crop(crop, angle):
        """将裁切图绕中心旋转 angle 度，画布自动扩展以避免截断。"""
        h, w = crop.shape[:2]
        cx, cy = w / 2, h / 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(crop, M, (new_w, new_h),
                              borderMode=cv2.BORDER_REPLICATE)

    def _try_rotated_ocr(self, crop):
        """对裁切图在 _ROTATION_ANGLES 各角度旋转后重试 OCR。
        返回 (text, text_conf, angle)；全部失败时返回 ("", 0.0, 0)。"""
        best_text, best_conf, best_angle = "", 0.0, 0
        for angle in self._ROTATION_ANGLES:
            rotated = self._rotate_crop(crop, angle)
            text, conf = self._recognize_text(rotated)
            if conf >= self.ocr_conf_threshold and is_valid_plate(text):
                if conf > best_conf:
                    best_text, best_conf, best_angle = text, conf, angle
                    if conf >= 0.90:
                        break
        return best_text, best_conf, best_angle

    def _classify(self, measured_min_side, text, text_conf):
        """两层漏斗判定（先读号码，再看尺寸）：
        第①层——能否读出【完整且符合中国车牌格式】的号码？读不出 → 不算车牌(unread)。
        第②层——在能读出的车牌里，最小边长是否 ≥16px(国标5.6.2.1)？
        - standard : 读出合法号码 且 ≥16px  → 可识别 且 符合国标
        - small    : 读出合法号码 但 <16px  → 算车牌，但尺寸不达国标
        - unread   : 未读出合法号码（遮挡/不清晰/非车牌误检如 GB/T）→ 不计为车牌
        说明：被遮挡导致号码不全 → 读不出 → 自动不算车牌（契合国标"遮挡不计"）。"""
        valid = bool(text) and text_conf >= self.ocr_conf_threshold and is_valid_plate(text)
        if not valid:
            return "unread"
        return "standard" if measured_min_side >= self.min_side_px else "small"

    def detect(self, img_matrix):
        """返回车牌列表: [{bbox, confidence, text, text_confidence, min_side, status}, ...]"""
        self._ensure_yolo()
        # imgsz 调高 + conf 调低，显著提升远处/小/斜车牌召回。
        results = self._yolo(img_matrix, conf=self.conf_threshold,
                             imgsz=self.imgsz, verbose=False)

        # 1) 先对每个框做 OCR，组装候选（含状态、面积、是否读出合法车牌）。
        h_img, w_img = img_matrix.shape[:2]
        candidates = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                det_conf = round(float(box.conf[0]), 4)
                measured_min_side = min(x2 - x1, y2 - y1)
                # 裁切时加 8% 边距，防止 YOLO bbox 偏小导致 OCR 读到残缺车牌
                pad_x = max(4, int((x2 - x1) * 0.08))
                pad_y = max(4, int((y2 - y1) * 0.08))
                crop = img_matrix[max(y1 - pad_y, 0):min(y2 + pad_y, h_img),
                                  max(x1 - pad_x, 0):min(x2 + pad_x, w_img)]
                text, text_conf = self._recognize_text(crop)
                status = self._classify(measured_min_side, text, text_conf)
                candidates.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": det_conf,
                    "text": text,
                    "text_confidence": text_conf,
                    "min_side": measured_min_side,
                    "status": status,
                    "_area": (x2 - x1) * (y2 - y1),
                    "_valid": status != "unread",   # 读出合法车牌号(standard/small)即为真牌
                })

        # 1.5) 对 unread 框启用旋转暴力搜索：在 -30°~+30° 逐角度旋转裁切图重跑 OCR，
        #       尝试挽救 YOLO 检测到形状但正向 OCR 因倾斜失败的车牌（如实测 006 帧）。
        if self.enable_rotation_search:
            for c in candidates:
                if c["status"] != "unread":
                    continue
                x1, y1, x2, y2 = c["bbox"]
                pad_x = max(4, int((x2 - x1) * 0.08))
                pad_y = max(4, int((y2 - y1) * 0.08))
                crop = img_matrix[max(y1 - pad_y, 0):min(y2 + pad_y, h_img),
                                  max(x1 - pad_x, 0):min(x2 + pad_x, w_img)]
                text, text_conf, angle = self._try_rotated_ocr(crop)
                if text:
                    c["text"] = text
                    c["text_confidence"] = text_conf
                    c["status"] = self._classify(c["min_side"], text, text_conf)
                    c["_valid"] = c["status"] != "unread"
                    c["rotation_angle"] = angle
                    print(f"     旋转矫正成功: {text} (角度{angle:+d}°, conf={text_conf:.2f})")

        # 2) 补一道去重（YOLO 自带 NMS 漏掉的嵌套重复框），保证「一块真牌 ↔ 一个框」，
        #    否则脚本五的目标计数(项5)和框配对(项1~3)都会出错。
        #    三级优先（前级相同才看后级）：① 读出合法车牌(_valid)
        #    ② 文字置信度更高 ③ 框面积更大。用 IoMin>0.75 判定是否同一块牌的重复。
        candidates.sort(key=lambda c: (c["_valid"], c["text_confidence"], c["_area"]),
                        reverse=True)
        plates = []
        for c in candidates:
            if all(_overlap_ratio(c["bbox"], k["bbox"]) <= 0.75 for k in plates):
                plates.append(c)
        for c in plates:
            c.pop("_area", None)
            c.pop("_valid", None)
        return plates


# ======================================================================
# 二-B、大角度车牌副检测器（we0091234: YOLOv5 关键点 + 透视矫正 + CRNN）
# ======================================================================
class AnglePlateDetector:
    """大角度车牌兜底检测器。
    使用 we0091234/Chinese_license_plate_detection_recognition 的完整流水线：
    YOLOv5（带四角关键点）→ four_point_transform 透视矫正 → CRNN 字符识别。
    专门处理主检测器（YOLO+PaddleOCR）漏检或读不出的极端倾角车牌。"""

    def __init__(self, detect_weights=None, rec_weights=None, img_size=640):
        base = os.path.join(_CHINESE_PLATE_DIR, "weights")
        self.detect_weights = detect_weights or os.path.join(base, "plate_detect.pt")
        self.rec_weights = rec_weights or os.path.join(base, "plate_rec_color.pth")
        self.img_size = img_size
        self._detect_model = None
        self._rec_model = None
        self._device = None
        self._infer_fn = None

    def _ensure_models(self):
        if self._detect_model is not None:
            return
        import torch
        from models.experimental import attempt_load
        from plate_recognition.plate_rec import init_model

        self._device = torch.device("cpu")
        self._detect_model = attempt_load(self.detect_weights,
                                          map_location=self._device)
        self._detect_model.eval()
        self._rec_model = init_model(self._device, self.rec_weights,
                                     is_color=True)
        from detect_plate import detect_Recognition_plate
        self._infer_fn = detect_Recognition_plate

    def detect(self, img_matrix):
        """返回与 PlateDetector.detect() 相同格式的列表，便于合并去重。"""
        self._ensure_models()
        raw_list = self._infer_fn(self._detect_model, img_matrix,
                                  self._device, self._rec_model,
                                  self.img_size, is_color=True)
        plates = []
        for r in raw_list:
            x1, y1, x2, y2 = r["rect"]
            text = r.get("plate_no", "")
            rec_scores = r.get("rec_conf", 0.0)
            try:
                text_conf = round(float(np.min(rec_scores)), 4)
            except (TypeError, ValueError):
                text_conf = 0.0
            det_conf = round(float(r.get("detect_conf", 0.0)), 4)
            measured_min_side = min(x2 - x1, y2 - y1)
            norm_text = normalize_plate(text)
            valid = bool(norm_text) and is_valid_plate(norm_text)
            if valid:
                status = "standard" if measured_min_side >= PlateDetector.MIN_SIDE_PX else "small"
            else:
                status = "unread"
            plates.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": det_conf,
                "text": norm_text if valid else text,
                "text_confidence": text_conf,
                "min_side": measured_min_side,
                "status": status,
                "source": "angle_plate",
            })
        return plates


# ======================================================================
# 三、检测调度器：遍历图片 -> 检测 -> JSON + 可视化
# ======================================================================
class GbtDetectionScheduler:
    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, image_folder, output_json_folder,
                 enable_face=True, enable_plate=True,
                 yolo_weights="weights/license_plate_yolov8.pt",
                 save_visualization=True,
                 enable_angle_plate=True,
                 viz_folder=None):
        self.image_folder = image_folder
        self.output_json_folder = output_json_folder
        self.save_visualization = save_visualization
        self.viz_folder = viz_folder or os.path.join(output_json_folder, "visualization")

        self.face_detector = FaceDetector() if enable_face else None
        self.plate_detector = PlateDetector(yolo_weights) if enable_plate else None
        self.angle_plate_detector = AnglePlateDetector() if enable_angle_plate else None

        os.makedirs(self.output_json_folder, exist_ok=True)
        if self.save_visualization:
            os.makedirs(self.viz_folder, exist_ok=True)

    def _list_images(self):
        """递归扫描（脚本一按视频名建子目录），返回相对 image_folder 的相对路径列表。"""
        if not os.path.exists(self.image_folder):
            print(f"未在硬盘中找到图片源文件夹: {self.image_folder}")
            return []
        rel_paths = []
        for root, _dirs, files in os.walk(self.image_folder):
            for f in files:
                if f.lower().endswith(self.IMAGE_EXTENSIONS):
                    full = os.path.join(root, f)
                    rel_paths.append(os.path.relpath(full, self.image_folder))
        return sorted(rel_paths)

    # 颜色 (B,G,R)
    _GREEN = (0, 200, 0)       # 符合国标车牌
    _ORANGE = (0, 165, 255)    # 可识别但 <16px
    _GREY = (150, 150, 150)    # 未读出(不算车牌)
    _RED = (0, 0, 255)         # 人脸框

    def _draw_and_save(self, img_matrix, img_name, faces, plates):
        canvas = img_matrix.copy()
        labels = []   # (text, x, y, (B,G,R))，统一用 PIL 渲染中文
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), self._RED, 2)
            labels.append((f"face {f['confidence']:.2f}", x1, max(y1 - 26, 2), self._RED))
        for p in plates:
            x1, y1, x2, y2 = p["bbox"]
            status = p.get("status")
            if status == "standard":          # 符合国标：绿框 + 号码
                color, text, thick = self._GREEN, f"{p['text']}({p['text_confidence']:.2f})", 2
            elif status == "small":           # 可识别但<16px：橙框 + 号码
                color, text, thick = self._ORANGE, f"{p['text']} <16px({p['min_side']}px)", 2
            else:                             # unread：灰细框，不算车牌（仅供核对漏读）
                color, text, thick = self._GREY, "未读出(非车牌)", 1
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
            labels.append((text, x1, max(y1 - 26, 2), color))
        canvas = draw_labels_cn(canvas, labels)
        viz_path = os.path.join(self.viz_folder, img_name)
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        imwrite_unicode(viz_path, canvas)

    @staticmethod
    def _merge_plates(primary, secondary):
        """合并主/副检测器结果，用 IoMin 去重：同一块牌只保留读出号码且置信度更高的。"""
        merged = list(primary)
        for s in secondary:
            is_dup = False
            for i, m in enumerate(merged):
                if _overlap_ratio(s["bbox"], m["bbox"]) > 0.75:
                    is_dup = True
                    s_valid = s.get("status") != "unread"
                    m_valid = m.get("status") != "unread"
                    if (s_valid, s.get("text_confidence", 0)) > (m_valid, m.get("text_confidence", 0)):
                        merged[i] = s
                    break
            if not is_dup:
                merged.append(s)
        return merged

    def _detect_one(self, img_name):
        full_path = os.path.join(self.image_folder, img_name)
        img_matrix = imread_unicode(full_path)
        if img_matrix is None:
            print(f"OpenCV 无法解码图片: {img_name}")
            return None

        height, width = img_matrix.shape[:2]
        faces = self.face_detector.detect(img_matrix) if self.face_detector else []
        plates = self.plate_detector.detect(img_matrix) if self.plate_detector else []

        # 副检测器（大角度透视矫正）：独立跑一遍，合并去重
        if self.angle_plate_detector:
            try:
                angle_plates = self.angle_plate_detector.detect(img_matrix)
                if angle_plates:
                    new_count = 0
                    before = len([p for p in plates if p.get("status") != "unread"])
                    plates = self._merge_plates(plates, angle_plates)
                    after = len([p for p in plates if p.get("status") != "unread"])
                    new_count = after - before
                    if new_count > 0:
                        print(f"     副检测器(透视矫正)新增 {new_count} 个车牌")
            except Exception as e:
                print(f"     副检测器异常(不影响主流程): {e}")

        if self.save_visualization:
            self._draw_and_save(img_matrix, img_name, faces, plates)

        # 两层漏斗：
        #  第①层 可识别车牌 = 读出合法号码（standard + small）
        #  第②层 符合国标车牌 = 可识别车牌里 ≥16px（standard）
        recognizable = [p for p in plates if p.get("status") in ("standard", "small")]
        standard = [p for p in plates if p.get("status") == "standard"]
        unread = [p for p in plates if p.get("status") == "unread"]

        recognizable_plates = [{"text": p["text"], "min_side": p["min_side"]} for p in recognizable]
        standard_plates = [{"text": p["text"], "min_side": p["min_side"]} for p in standard]

        return {
            "image_name": img_name,
            "image_path": full_path,
            "width": width,
            "height": height,
            "detect_time": datetime.now().isoformat(timespec="seconds"),
            "face_count": len(faces),
            # 第①层：可识别车牌（读出合法号码）
            "recognizable_plate_count": len(recognizable),
            "recognizable_plates": recognizable_plates,
            # 第②层：符合国标车牌（可识别 且 ≥16px）
            "standard_plate_count": len(standard),
            "standard_plates": standard_plates,
            # 未读出有效号码的框（不计为车牌，仅供核对算法漏读/误检）
            "unread_box_count": len(unread),
            "faces": faces,
            "plates": recognizable,   # 仅确认为车牌（读出号码）的，供脚本五比对
            "unread_boxes": unread,
        }

    @staticmethod
    def _fmt_plates(plate_list):
        """格式化车牌列表为 '苏U·XA512(45px) | 苏X·1234(12px,<16px不达国标)'。"""
        parts = []
        for p in plate_list:
            tag = f"{p['text']}({p['min_side']}px"
            tag += ",<16px不达国标)" if p["min_side"] < PlateDetector.MIN_SIDE_PX else ")"
            parts.append(tag)
        return " | ".join(parts)

    def run(self):
        image_files = self._list_images()
        total = len(image_files)
        print(f"在 {self.image_folder} 中扫描到 {total} 张待检图片。")
        if total == 0:
            return

        summary = []   # (img_name, recognizable_plates, standard_plates) 全局汇总
        for index, img_name in enumerate(image_files, 1):
            print(f"\n[检测进度 {index}/{total}] {img_name}")
            try:
                record = self._detect_one(img_name)
            except Exception as e:
                print(f"检测图片 {img_name} 时发生错误，已跳过。错误：{e}")
                traceback.print_exc()
                continue
            if record is None:
                continue

            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(self.output_json_folder, json_name)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as fp:
                json.dump(record, fp, ensure_ascii=False, indent=2)

            rec, std = record["recognizable_plates"], record["standard_plates"]
            print(f"  人脸 {record['face_count']} 个 | "
                  f"①可识别车牌 {record['recognizable_plate_count']} 个 | "
                  f"②符合国标(≥16px) {record['standard_plate_count']} 个 | "
                  f"未读出框 {record['unread_box_count']} 个 -> {json_name}")
            if rec:
                print("     ① 可识别车牌: " + self._fmt_plates(rec))
            if std:
                print("     ② 符合国标车牌: " + self._fmt_plates(std))
            summary.append((img_name, rec, std))

        print("\n" + "=" * 64 + "\n 识别检测任务已完成！汇总（每张图）：")
        print(" 说明：①可识别=读出合法车牌号；②符合国标=①里最小边长≥16px(国标§5.6.2.1)。未读出框不计为车牌。")
        print(" 号码后 (N px) = 该车牌边界框最小边长(宽、高中较小者)；标注 <16px 者不达国标尺寸。")
        for img_name, rec, std in summary:
            print(f"\n  {img_name}")
            print("    ① 可识别车牌: " + (self._fmt_plates(rec) if rec else "无"))
            print("    ② 符合国标车牌: " + (self._fmt_plates(std) if std else "无"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GBT 44464 脚本二 · 识别检测")
    parser.add_argument("--input", default=None, help="待检图片目录")
    parser.add_argument("--output-json", default=None, help="JSON 输出目录")
    parser.add_argument("--output-viz", default=None, help="可视化输出目录")
    args = parser.parse_args()

    print("=" * 60)
    print(" GBT 44464-2024 脚本二 · 识别检测程序启动")
    print("=" * 60)

    SRC_IMAGE_DIR = args.input or r"E:\Vehicle_Data_Anonymization_Verifier\self_check\unmasked\images"
    OUTPUT_JSON_DIR = args.output_json or r"E:\Vehicle_Data_Anonymization_Verifier\self_check\detection_json"
    OUTPUT_VIZ_DIR = args.output_viz
    YOLO_PLATE_WEIGHTS = r"weights\license_plate_yolov8.pt"

    scheduler = GbtDetectionScheduler(
        image_folder=SRC_IMAGE_DIR,
        output_json_folder=OUTPUT_JSON_DIR,
        enable_face=True,
        enable_plate=True,
        yolo_weights=YOLO_PLATE_WEIGHTS,
        save_visualization=True,
        viz_folder=OUTPUT_VIZ_DIR,
    )
    scheduler.run()
