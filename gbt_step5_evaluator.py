"""
脚本五 · 自检评测器（GB/T 44464-2024 汽车数据匿名化合规检测系统）

职责：在「阶段一 · 自检」中，把脚本二输出的检测 JSON 与人工标注真实值（labelme）
      逐图比对，按国标精度需求自动算出考核指标并判定是否通关。

本期实现考核项 1~5（未处理 500 张图，考核「裁判」几何精度）：
    项1  人脸边界框比值      —— ≥98% 落在 [0.9, 1.1]
    项2  小车牌边长平均绝对误差 —— ≤ 1 像素
    项3  大车牌边长平均比值    —— 落在 [0.9, 1.1]
    项4  人脸可见范围面积平均比值 —— 落在 [0.9, 1.1]（需额外标注「可见范围」，详见下方说明）
    项5  目标总数清点比例      —— 落在 [0.99, 1.01]
项6~7（已处理图的遮盖率/漏检比值）属合规判定范畴，留接口后置到脚本三/四。

国标锚点（GB/T 44464-2024 §5.6.2.1）：
    - 人脸边界框「最小边长」≥ 32 像素 才属匿名化对象
    - 汽车号牌边界框「最小边长」≥ 16 像素 才属匿名化对象
    «小/大车牌» 分界默认取 SMALL_PLATE_MAX_SIDE（可配置）。

标注约定（labelme，矩形 rectangle）：
    - 人脸：label = "face"
    - 车牌：label = "plate" 或 "plate:京A12345"（冒号后为车牌号真值，供 OCR 评测）
    - 坐标与脚本二一致：像素 [xmin, ymin, xmax, ymax]，左上角为原点
"""

import os
import sys
import json
import glob
from datetime import datetime

# Windows 控制台默认 GBK，无法打印 ✅/中文会报错；强制 stdout 用 UTF-8。
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ======================================================================
# 一、真实值 / 预测值 读取
# ======================================================================
def load_labelme_gt(json_path):
    """解析 labelme 单图 JSON，返回 {'faces': [...], 'plates': [...]}。"""
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    faces, plates = [], []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        pts = shape.get("points", [])
        if len(pts) != 2:
            continue
        (x1, y1), (x2, y2) = pts
        bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        label = str(shape.get("label", "")).strip()
        low = label.lower()
        if low.startswith("face"):
            faces.append({"bbox": bbox})
        elif low.startswith("plate"):
            text = label.split(":", 1)[1].strip() if ":" in label else ""
            plates.append({"bbox": bbox, "text": text})
    return {"faces": faces, "plates": plates}


def load_detection(json_path):
    """读取脚本二输出 JSON，返回 {'faces': [...], 'plates': [...]}（统一只取 bbox/text）。"""
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    faces = [{"bbox": f["bbox"]} for f in data.get("faces", [])]
    plates = [{"bbox": p["bbox"], "text": p.get("text", "")} for p in data.get("plates", [])]
    return {"faces": faces, "plates": plates}


# ======================================================================
# 二、几何工具：IoU、边长、贪心配对
# ======================================================================
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_sides(box):
    """返回 (宽, 高)。"""
    return abs(box[2] - box[0]), abs(box[3] - box[1])


def box_area(box):
    w, h = box_sides(box)
    return w * h


def greedy_match(preds, gts, iou_thr=0.5):
    """按 IoU 从高到低贪心配对，返回 (matched_pairs, unmatched_pred_idx, unmatched_gt_idx)。
    matched_pairs: [(pred_idx, gt_idx, iou), ...]"""
    candidates = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            v = iou(p["bbox"], g["bbox"])
            if v >= iou_thr:
                candidates.append((v, pi, gi))
    candidates.sort(reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for v, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi, v))
    unmatched_p = [i for i in range(len(preds)) if i not in used_p]
    unmatched_g = [i for i in range(len(gts)) if i not in used_g]
    return pairs, unmatched_p, unmatched_g


# ======================================================================
# 三、自检评测器
# ======================================================================
class GbtSelfCheckEvaluator:
    def __init__(self, gt_folder, detection_folder,
                 report_path="self_check_report.json",
                 iou_thr=0.5, small_plate_max_side=32.0):
        """
        gt_folder        : labelme 标注 JSON 根目录（可含子目录，结构与抽帧一致）
        detection_folder : 脚本二输出 opencv_output_json 根目录
        small_plate_max_side : «小车牌» 上限（最小边长 ≤ 此值算小牌）。
                               国标车牌阈值 16px、人脸 32px，默认 32px，可按需调整。
        """
        self.gt_folder = gt_folder
        self.detection_folder = detection_folder
        self.report_path = report_path
        self.iou_thr = iou_thr
        self.small_plate_max_side = small_plate_max_side

    # ---- 文件对齐：按相对路径 stem 匹配 GT 与 检测 ----
    def _pair_files(self):
        gt_files = glob.glob(os.path.join(self.gt_folder, "**", "*.json"), recursive=True)
        det_index = {}
        for d in glob.glob(os.path.join(self.detection_folder, "**", "*.json"), recursive=True):
            rel = os.path.relpath(d, self.detection_folder)
            det_index[os.path.splitext(rel)[0]] = d
        pairs, missing = [], []
        for g in gt_files:
            rel = os.path.relpath(g, self.gt_folder)
            stem = os.path.splitext(rel)[0]
            if stem in det_index:
                pairs.append((stem, g, det_index[stem]))
            else:
                missing.append(stem)
        return pairs, missing

    # ---- 主流程 ----
    def run(self):
        pairs, missing = self._pair_files()
        print(f"配对成功 {len(pairs)} 张；GT 有但检测结果缺失 {len(missing)} 张。")
        if not pairs:
            print("没有可比对的图片，请确认 labelme 标注目录与检测输出目录。")
            return None

        # 累积样本
        face_side_ratios = []          # 项1：人脸框每条边的 预测/真值 比
        face_visible_ratios = []       # 项4：人脸可见范围面积比（需额外标注，见下）
        small_plate_abs_err = []       # 项2：小车牌边长 |预测-真值|
        large_plate_side_ratios = []   # 项3：大车牌边长 预测/真值 比
        total_pred_targets = 0         # 项5
        total_gt_targets = 0
        item4_supported = False        # 是否具备可见范围标注+预测

        for stem, gt_path, det_path in pairs:
            gt = load_labelme_gt(gt_path)
            det = load_detection(det_path)

            # ---------- 人脸 ----------
            fpairs, _, _ = greedy_match(det["faces"], gt["faces"], self.iou_thr)
            for pi, gi, _ in fpairs:
                pw, ph = box_sides(det["faces"][pi]["bbox"])
                gw, gh = box_sides(gt["faces"][gi]["bbox"])
                if gw > 0:
                    face_side_ratios.append(pw / gw)
                if gh > 0:
                    face_side_ratios.append(ph / gh)

            # ---------- 车牌 ----------
            ppairs, _, _ = greedy_match(det["plates"], gt["plates"], self.iou_thr)
            for pi, gi, _ in ppairs:
                pw, ph = box_sides(det["plates"][pi]["bbox"])
                gw, gh = box_sides(gt["plates"][gi]["bbox"])
                gt_min_side = min(gw, gh)
                is_small = gt_min_side <= self.small_plate_max_side
                if is_small:
                    small_plate_abs_err.extend([abs(pw - gw), abs(ph - gh)])
                else:
                    if gw > 0:
                        large_plate_side_ratios.append(pw / gw)
                    if gh > 0:
                        large_plate_side_ratios.append(ph / gh)

            # ---------- 项5：目标计数 ----------
            total_pred_targets += len(det["faces"]) + len(det["plates"])
            total_gt_targets += len(gt["faces"]) + len(gt["plates"])

        results = self._summarize(
            face_side_ratios, small_plate_abs_err, large_plate_side_ratios,
            face_visible_ratios, total_pred_targets, total_gt_targets,
            item4_supported, len(pairs), len(missing),
        )
        self._print_and_save(results)
        return results

    # ---- 指标汇总 + 判定 ----
    @staticmethod
    def _ratio_in(vals, lo, hi):
        return sum(1 for v in vals if lo <= v <= hi) / len(vals) if vals else 0.0

    @staticmethod
    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    def _summarize(self, face_side_ratios, small_plate_abs_err, large_plate_side_ratios,
                   face_visible_ratios, total_pred, total_gt, item4_supported,
                   n_pairs, n_missing):
        items = []

        # 项1：≥98% 人脸框边比落在 [0.9,1.1]。无样本（无人脸或全未配上）记 N/A，
        # 漏检由项5 计数比兜底，避免「没数据」被误判为失败。
        p1 = self._ratio_in(face_side_ratios, 0.9, 1.1)
        items.append({
            "项": 1, "名称": "人脸边界框比值", "样本数": len(face_side_ratios),
            "实测": (f"{p1*100:.2f}% 落在[0.9,1.1]" if face_side_ratios else "N/A"),
            "标准": "≥98%",
            "通过": (p1 >= 0.98) if face_side_ratios else None,
        })

        # 项2：小车牌边长 MAE ≤ 1px
        mae = self._mean(small_plate_abs_err)
        items.append({
            "项": 2, "名称": "小车牌边长平均绝对误差", "样本数": len(small_plate_abs_err),
            "实测": (f"{mae:.4f} px" if small_plate_abs_err else "N/A"), "标准": "≤1px",
            "通过": (mae <= 1.0) if small_plate_abs_err else None,
        })

        # 项3：大车牌边长平均比值 落在 [0.9,1.1]
        m3 = self._mean(large_plate_side_ratios)
        items.append({
            "项": 3, "名称": "大车牌边长平均比值", "样本数": len(large_plate_side_ratios),
            "实测": (f"{m3:.4f}" if large_plate_side_ratios else "N/A"), "标准": "[0.9,1.1]",
            "通过": (0.9 <= m3 <= 1.1) if large_plate_side_ratios else None,
        })

        # 项4：人脸可见范围面积平均比值（需可见范围标注+预测，当前未支持）
        if item4_supported and face_visible_ratios:
            m4 = self._mean(face_visible_ratios)
            items.append({
                "项": 4, "名称": "人脸可见范围面积平均比值", "样本数": len(face_visible_ratios),
                "实测": f"{m4:.4f}", "标准": "[0.9,1.1]",
                "通过": 0.9 <= m4 <= 1.1,
            })
        else:
            items.append({
                "项": 4, "名称": "人脸可见范围面积平均比值", "样本数": 0,
                "实测": "N/A", "标准": "[0.9,1.1]", "通过": None,
                "备注": "需 labelme 额外标注「可见范围」多边形，且脚本二输出可见范围；当前检测仅有人脸框，暂不评测。",
            })

        # 项5：目标总数清点比例 落在 [0.99,1.01]
        r5 = (total_pred / total_gt) if total_gt > 0 else 0.0
        items.append({
            "项": 5, "名称": "目标总数清点比例", "样本数": total_gt,
            "实测": (f"{r5:.4f} (预测{total_pred}/真值{total_gt})" if total_gt > 0 else "N/A"),
            "标准": "[0.99,1.01]",
            "通过": (0.99 <= r5 <= 1.01) if total_gt > 0 else None,
        })

        gradable = [it for it in items if it["通过"] is not None]
        all_pass = bool(gradable) and all(it["通过"] for it in gradable)
        return {
            "评测时间": datetime.now().isoformat(timespec="seconds"),
            "配对图片数": n_pairs, "检测缺失数": n_missing,
            "IoU阈值": self.iou_thr, "小车牌最小边长上限px": self.small_plate_max_side,
            "考核项": items,
            "整体通关": all_pass,
        }

    def _print_and_save(self, results):
        print("\n" + "=" * 64)
        print(" GB/T 44464-2024 自检评测报告（项 1~5）")
        print("=" * 64)
        for it in results["考核项"]:
            mark = "—" if it["通过"] is None else ("✅" if it["通过"] else "❌")
            print(f" [项{it['项']}] {it['名称']}")
            print(f"     实测: {it['实测']} | 标准: {it['标准']} | {mark}  (样本 {it['样本数']})")
            if it.get("备注"):
                print(f"     备注: {it['备注']}")
        print("-" * 64)
        print(f" 整体通关: {'✅ 通过' if results['整体通关'] else '❌ 未通过'}")
        print("=" * 64)
        with open(self.report_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        print(f"报告已保存: {self.report_path}")


if __name__ == "__main__":
    print("=" * 64)
    print(" GBT 44464-2024 脚本五 · 自检评测器启动")
    print("=" * 64)

    # labelme 真值 .json 默认存图片旁，故 GT_DIR = 待检图片目录（与脚本二 SRC 同一个）。
    GT_DIR = r"E:\Vehicle_Data_Anonymization_Verifier\self_check\unmasked\images"
    DET_DIR = r"E:\Vehicle_Data_Anonymization_Verifier\self_check\detection_json"
    REPORT = r"E:\Vehicle_Data_Anonymization_Verifier\self_check\report\self_check_report.json"

    evaluator = GbtSelfCheckEvaluator(
        gt_folder=GT_DIR,
        detection_folder=DET_DIR,
        report_path=REPORT,
        iou_thr=0.5,
        small_plate_max_side=32.0,
    )
    evaluator.run()
