"""
SPIGA 98点(WFLW)人脸关键点 → 国标 §3.5 人脸边界框（CPU 封装）。

为什么要它：RetinaFace 只给 5 点(眼/鼻/嘴)，不含眉毛，无法定位国标要求的
「眉毛最上沿」。SPIGA 输出 WFLW 98 点，含完整眉毛、脸颊轮廓、下颌线，可逐项对应：
    顶边 = 眉毛点最小 y      （眉毛最上沿）
    底边 = 脸轮廓点最大 y    （颏底线）
    左右 = 脸轮廓(不含耳)横向极值（左右耳间不含耳）

WFLW 98 点索引（标准顺序）：
    0–32  脸轮廓(下颌/脸颊，不含耳)
    33–50 双眉(每侧9点，含上下沿)
    51–59 鼻；60–67 左眼；68–75 右眼；76–95 嘴；96/97 瞳孔中心

CPU 适配：SPIGA 官方 framework 把 .cuda() 写死，本机 torch 为 CPU 版。
本封装在「构造期」用一组可还原的运行时补丁把它 CPU 化（不改 site-packages）：
    ① _data2device 走目标 device   ② Module.cuda → .to(device)
    ③ torch.load 注入 map_location（权重为 GPU 序列化，CPU 直接 load 会报错）
"""

import torch
import numpy as np

# WFLW 98 点分区
_WFLW_CONTOUR = list(range(0, 33))    # 脸轮廓（不含耳）
_WFLW_BROWS = list(range(33, 51))     # 双眉


class SpigaLandmarker:
    """懒加载的 SPIGA 关键点器；gbt_face_box() 直接给国标人脸框。"""

    def __init__(self, dataset="wflw"):
        self._proc = None
        self._dataset = dataset
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure(self):
        if self._proc is not None:
            return
        from spiga.inference.config import ModelConfig
        import spiga.inference.framework as fw

        device = self._device

        # 补丁①：原版 _data2device 写死 data.cuda(...)，改为走目标 device
        def _data2device(self2, data):
            if isinstance(data, list):
                return [self2._data2device(v) for v in data]
            if isinstance(data, dict):
                return {k: self2._data2device(v) for k, v in data.items()}
            with torch.no_grad():
                return data.to(device)
        fw.SPIGAFramework._data2device = _data2device

        cfg = ModelConfig(self._dataset, load_model_url=False)  # 用本地权重，不走 Google Drive

        # 补丁②③：仅在构造期临时生效，用完还原
        orig_cuda = torch.nn.Module.cuda
        orig_load = torch.load

        def _cuda(module, *args, **kwargs):
            return module.to(device)

        def _load(f, *args, **kwargs):
            kwargs.setdefault("map_location", device)
            return orig_load(f, *args, **kwargs)

        torch.nn.Module.cuda = _cuda
        torch.load = _load
        try:
            self._proc = fw.SPIGAFramework(cfg, load3DM=True)
        finally:
            torch.nn.Module.cuda = orig_cuda
            torch.load = orig_load

    def gbt_face_box(self, image_bgr, facial_area):
        """输入整图(BGR) + RetinaFace facial_area [x1,y1,x2,y2]，
        返回国标 §3.5 人脸框 [x1,y1,x2,y2]；失败返回 None。"""
        try:
            self._ensure()
            x1, y1, x2, y2 = (float(v) for v in facial_area)
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            feats = self._proc.inference(image_bgr, [bbox_xywh])
            lm = np.asarray(feats["landmarks"][0], dtype=float)  # (98, 2)
            if lm.shape[0] < 51:
                return None
            brows = lm[_WFLW_BROWS]
            contour = lm[_WFLW_CONTOUR]
            top = brows[:, 1].min()
            bottom = contour[:, 1].max()
            left = min(contour[:, 0].min(), brows[:, 0].min())
            right = max(contour[:, 0].max(), brows[:, 0].max())
            return [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))]
        except Exception:
            return None
