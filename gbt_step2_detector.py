"""
脚本二 · 识别检测（GB/T 44464-2024 汽车数据匿名化合规检测系统）

职责：对脚本一抽取的 1080P 图片集做车牌 / 人脸检测，
      输出带坐标、置信度、识别内容的标准化 JSON，并生成可视化图。

检测模型栈（方案A · 准确率优先，全离线）：
    人脸检测  ->  RetinaFace
    车牌定位  ->  YOLOv8 (license-plate)
    车牌识别  ->  PaddleOCR (中文车牌)

OpenCV 仅用于图像 I/O、坐标绘制与结果可视化。
"""

import os

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

import json
import traceback
from datetime import datetime

import cv2
import numpy as np


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


# ======================================================================
# 一、人脸检测器（RetinaFace）
# ======================================================================
class FaceDetector:
    """基于 RetinaFace 的离线人脸检测。延迟加载，未安装时给出清晰提示。"""

    def __init__(self, conf_threshold=0.5):
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

    def detect(self, img_matrix):
        """返回人脸列表: [{bbox:[x1,y1,x2,y2], confidence:float}, ...]"""
        self._ensure_engine()
        # RetinaFace 接受 BGR ndarray，返回 {face_1: {facial_area, score, ...}}
        raw = self._engine.detect_faces(img_matrix, threshold=self.conf_threshold)
        faces = []
        if isinstance(raw, dict):
            for info in raw.values():
                x1, y1, x2, y2 = info["facial_area"]
                faces.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(float(info.get("score", 1.0)), 4),
                })
        return faces


# ======================================================================
# 二、车牌定位 + 识别（YOLOv8 + PaddleOCR）
# ======================================================================
class PlateDetector:
    """YOLOv8 定位车牌区域，PaddleOCR 识别车牌文字。"""

    def __init__(self, yolo_weights, conf_threshold=0.4, ocr_lang="ch"):
        self.yolo_weights = yolo_weights
        self.conf_threshold = conf_threshold
        self.ocr_lang = ocr_lang
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

    def detect(self, img_matrix):
        """返回车牌列表: [{bbox, confidence, text, text_confidence}, ...]"""
        self._ensure_yolo()
        results = self._yolo(img_matrix, conf=self.conf_threshold, verbose=False)
        plates = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                det_conf = round(float(box.conf[0]), 4)
                crop = img_matrix[max(y1, 0):y2, max(x1, 0):x2]
                text, text_conf = self._recognize_text(crop)
                plates.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": det_conf,
                    "text": text,
                    "text_confidence": text_conf,
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
                 save_visualization=True):
        self.image_folder = image_folder
        self.output_json_folder = output_json_folder
        self.save_visualization = save_visualization
        self.viz_folder = os.path.join(output_json_folder, "visualization")

        self.face_detector = FaceDetector() if enable_face else None
        self.plate_detector = PlateDetector(yolo_weights) if enable_plate else None

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

    def _draw_and_save(self, img_matrix, img_name, faces, plates):
        canvas = img_matrix.copy()
        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(canvas, f"face {f['confidence']:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        for p in plates:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{p['text'] or 'plate'} {p['confidence']:.2f}"
            cv2.putText(canvas, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        viz_path = os.path.join(self.viz_folder, img_name)
        os.makedirs(os.path.dirname(viz_path), exist_ok=True)
        imwrite_unicode(viz_path, canvas)

    def _detect_one(self, img_name):
        full_path = os.path.join(self.image_folder, img_name)
        img_matrix = imread_unicode(full_path)
        if img_matrix is None:
            print(f"OpenCV 无法解码图片: {img_name}")
            return None

        height, width = img_matrix.shape[:2]
        faces = self.face_detector.detect(img_matrix) if self.face_detector else []
        plates = self.plate_detector.detect(img_matrix) if self.plate_detector else []

        if self.save_visualization:
            self._draw_and_save(img_matrix, img_name, faces, plates)

        return {
            "image_name": img_name,
            "image_path": full_path,
            "width": width,
            "height": height,
            "detect_time": datetime.now().isoformat(timespec="seconds"),
            "face_count": len(faces),
            "plate_count": len(plates),
            "faces": faces,
            "plates": plates,
        }

    def run(self):
        image_files = self._list_images()
        total = len(image_files)
        print(f"在 {self.image_folder} 中扫描到 {total} 张待检图片。")
        if total == 0:
            return

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
            print(f"  人脸 {record['face_count']} 个 | 车牌 {record['plate_count']} 个"
                  f" -> {json_name}")

        print("\n" + "=" * 60 + "\n 识别检测任务已完成！")


if __name__ == "__main__":
    print("=" * 60)
    print(" GBT 44464-2024 脚本二 · 识别检测程序启动")
    print("=" * 60)

    # 自检·未打码图片（考核项1~5）。若要检测已打码图（项6~7），把 unmasked 改成 masked。
    SRC_IMAGE_DIR = r"E:\Vehicle_Data_Anonymization_Verifier\self_check\unmasked\images"
    OUTPUT_JSON_DIR = r"E:\Vehicle_Data_Anonymization_Verifier\self_check\detection_json"
    YOLO_PLATE_WEIGHTS = r"weights\license_plate_yolov8.pt"

    scheduler = GbtDetectionScheduler(
        image_folder=SRC_IMAGE_DIR,
        output_json_folder=OUTPUT_JSON_DIR,
        enable_face=True,
        enable_plate=True,
        yolo_weights=YOLO_PLATE_WEIGHTS,
        save_visualization=True,
    )
    scheduler.run()
