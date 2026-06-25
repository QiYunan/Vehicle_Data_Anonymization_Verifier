"""
脚本三 · 正式检测合规计算引擎（GB/T 44464-2024 汽车数据匿名化合规检测系统）

职责：在「阶段二 · 正式检测」中，对主机厂提交的**已匿名化数据**出合规结论。
      吃脚本二（gbt_step2_detector.py）在打码图上的检测 JSON，结合「真值/遮盖率记录」，
      计算国标 §5.6.2.2 / §5.6.2.3 的各项指标并判定是否合规。

本期实现四大指标（人脸、汽车号牌各算一套）：
    指标A  匿名化检出率   —— 正检数/应检数，≥90%        （§5.6.2.2.1，公式 B.1/B.2，强制）
    指标B  漏检率         —— 漏检数/应检数 = 1−检出率，≤10%（检出率补数，强制）
    指标C  匿名化误检率   —— 误检数/检出数，≤10%          （§5.6.2.2.2 / 附录C，本项目视作强制）
    指标D  匿名化效果     —— 已打码目标应无法被识别        （§5.6.2.3 / B.6.1 机器识别）

国标锚点：
    §5.6.2.1 匿名化对象（判「该不该打码」=应检对象）：
        - 人脸：边界框最小边长 ≥ 32px，且可见范围比值 > 50%（且五官可见，此条人工/标注给定）
        - 号牌：边界框最小边长 ≥ 16px，且无遮挡可识别全部字符（readable，标注给定）
    B.5.3 正检/漏检分界：应检目标中 遮盖率 ≥ 50% → 正检；< 50% → 漏检。
    附录C 误检：被标记打码、却与任一同类真实目标无交集的区域。

核心认知：在打码图上，脚本二**仍能检出/读出**的目标 = 仍可识别 = 漏检（也即匿名化效果不合格）；
          打好码的目标脚本二看不见，其「正检」身份由真值/遮盖率记录给出。

—— 数据来源（可切换）——
    1) 脚本二检测 JSON 目录（必需）：提供「仍可识别目标」，用于漏检交叉核对与匿名化效果。
    2) 真值/遮盖率记录目录（可选）：每图一份 JSON，提供所有真实目标的 bbox/可见范围/遮盖率，
       以及主机厂标记打码区域（算误检率）。其 schema 见 load_truth_record 文档。
       —— 若缺该记录 → 进入 detection_only 退化模式：只报「仍可识别目标数（漏检下限）」+ 告警，
          因应检数分母不全，无法给出真实检出率/误检率。
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
# 〇、国标阈值常量（§5.6.2.1 / §5.6.2.2 / B.5.3）
# ======================================================================
FACE_MIN_SIDE_PX = 32.0          # §5.6.2.1.1 人脸匿名化对象最小边长
PLATE_MIN_SIDE_PX = 16.0         # §5.6.2.1.2 号牌匿名化对象最小边长
FACE_VISIBLE_RATIO_MIN = 0.5     # §5.6.2.1.1 可见范围比值 > 50%
COVERAGE_PASS = 0.5              # B.5.3 遮盖率 ≥ 50% 计正检
DETECTION_RATE_PASS = 0.90       # §5.6.2.2.1 检出率 ≥ 90%
FALSE_RATE_PASS = 0.10           # §5.6.2.2.2 误检率 ≤ 10%（本项目视作强制）
IOU_MATCH = 0.5                  # 真实目标 ↔ 脚本二检出 的配对 IoU 阈值
OVERLAP_HIT = 0.5                # 匿名化效果核对：检出框与目标框 交/小面积 比 > 此值即视为命中


# ======================================================================
# 一、几何工具（与脚本五一致的约定：bbox = [xmin, ymin, xmax, ymax]）
# ======================================================================
def box_sides(box):
    """返回 (宽, 高)。"""
    return abs(box[2] - box[0]), abs(box[3] - box[1])


def box_min_side(box):
    w, h = box_sides(box)
    return min(w, h)


def box_area(box):
    w, h = box_sides(box)
    return w * h


def inter_area(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return iw * ih


def iou(a, b):
    inter = inter_area(a, b)
    if inter <= 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(a, b):
    """交集 / 较小框面积。用于「检出框是否落在目标框内」的命中判定。"""
    inter = inter_area(a, b)
    if inter <= 0:
        return 0.0
    smaller = min(box_area(a), box_area(b))
    return inter / smaller if smaller > 0 else 0.0


def boxes_intersect(a, b):
    """是否存在交集（面积 > 0）。用于误检判定（与任一目标无交集）。"""
    return inter_area(a, b) > 0


# ======================================================================
# 二、数据读取
# ======================================================================
def load_detection(json_path):
    """读取脚本二输出 JSON，返回 {'faces':[{bbox}], 'plates':[{bbox,text}]}。
    plates 仅含脚本二已读出号码的车牌（standard+small），即「仍可识别」的号牌。"""
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    faces = [{"bbox": f["bbox"]} for f in data.get("faces", [])]
    plates = [{"bbox": p["bbox"], "text": p.get("text", "")} for p in data.get("plates", [])]
    return {"faces": faces, "plates": plates}


def load_truth_record(json_path):
    """读取「真值/遮盖率记录」单图 JSON。

    期望 schema：
    {
      "image": "001/001_0001.jpg",                 // 可选，仅备注
      "targets": [                                  // 图中所有真实人脸/号牌目标（含已打好码的）
        { "type": "face", "bbox": [x1,y1,x2,y2],
          "visible_ratio": 0.85,                    // 人脸可见范围比值，判 §5.6.2.1（缺省视为 1.0）
          "coverage": 0.0 },                        // 遮盖率 0~1（≥0.5 正检 / <0.5 漏检）
        { "type": "plate", "bbox": [x1,y1,x2,y2],
          "readable": true,                         // 号牌无遮挡可识别全部字符，判 §5.6.2.1（缺省 true）
          "coverage": 0.95,
          "plate_text": "苏U..." }                  // 可选，供匿名化效果核对
      ],
      "marked_regions": [                           // 主机厂标记并打码的区域（算误检率，可选）
        { "type": "plate", "bbox": [x1,y1,x2,y2], "exclude": false }
        // exclude=true 表示属国标 C.1/C.4 例外（广告牌/倒影/动物脸/两轮车牌等），不计误检
      ]
    }
    返回标准化结构；缺字段给安全默认。
    """
    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    targets = []
    for t in data.get("targets", []):
        typ = str(t.get("type", "")).strip().lower()
        if typ not in ("face", "plate"):
            continue
        targets.append({
            "type": typ,
            "bbox": t["bbox"],
            "visible_ratio": float(t.get("visible_ratio", 1.0)),
            "readable": bool(t.get("readable", True)),
            "coverage": float(t.get("coverage", 0.0)),
            "plate_text": str(t.get("plate_text", "")),
        })

    marked = []
    for r in data.get("marked_regions", []):
        typ = str(r.get("type", "")).strip().lower()
        if typ not in ("face", "plate"):
            continue
        marked.append({
            "type": typ,
            "bbox": r["bbox"],
            "exclude": bool(r.get("exclude", False)),
        })
    return {"targets": targets, "marked_regions": marked}


# ======================================================================
# 三、§5.6.2.1 应检对象判定
# ======================================================================
def is_required_target(target):
    """判断真实目标是否为「应检（须匿名化）对象」。"""
    bbox = target["bbox"]
    if target["type"] == "face":
        return (box_min_side(bbox) >= FACE_MIN_SIDE_PX
                and target["visible_ratio"] > FACE_VISIBLE_RATIO_MIN)
    else:  # plate
        return (box_min_side(bbox) >= PLATE_MIN_SIDE_PX
                and target["readable"])


# ======================================================================
# 四、合规计算器
# ======================================================================
class ComplianceCalculator:
    def __init__(self, detection_folder, truth_folder=None,
                 report_path="compliance_report.json",
                 coverage_pass=COVERAGE_PASS, overlap_hit=OVERLAP_HIT):
        """
        detection_folder : 脚本二输出 JSON 根目录（必需）
        truth_folder     : 真值/遮盖率记录根目录（可选；None → detection_only 退化模式）
        coverage_pass    : 遮盖率正检阈值（默认 0.5）
        overlap_hit      : 匿名化效果命中判定的 交/小面积 阈值（默认 0.5）
        """
        self.detection_folder = detection_folder
        self.truth_folder = truth_folder
        self.report_path = report_path
        self.coverage_pass = coverage_pass
        self.overlap_hit = overlap_hit
        self.detection_only = truth_folder is None

    # ---- 文件对齐：按相对路径 stem 把 真值记录 与 检测 JSON 配对 ----
    def _index_detection(self):
        idx = {}
        for d in glob.glob(os.path.join(self.detection_folder, "**", "*.json"), recursive=True):
            rel = os.path.relpath(d, self.detection_folder)
            idx[os.path.splitext(rel)[0]] = d
        return idx

    def _pair_files(self):
        det_index = self._index_detection()
        if self.detection_only:
            # 无真值：直接遍历所有检测 JSON
            return [(stem, None, det_path) for stem, det_path in det_index.items()], []
        pairs, missing = [], []
        truth_files = glob.glob(os.path.join(self.truth_folder, "**", "*.json"), recursive=True)
        for g in truth_files:
            rel = os.path.relpath(g, self.truth_folder)
            stem = os.path.splitext(rel)[0]
            if stem in det_index:
                pairs.append((stem, g, det_index[stem]))
            else:
                missing.append(stem)
        return pairs, missing

    # ---- 匿名化效果（B.6.1 机器识别）：正检目标在脚本二输出里是否仍被识别 ----
    def _still_recognizable(self, target, det):
        """对一个已打码（正检）目标，查脚本二是否仍检出同类目标命中其 bbox。
        命中 → 仍可识别 → 匿名化效果不合格。返回 (是否命中, 命中说明)。"""
        dets = det["faces"] if target["type"] == "face" else det["plates"]
        for d in dets:
            if overlap_ratio(d["bbox"], target["bbox"]) > self.overlap_hit:
                if target["type"] == "plate":
                    return True, f"脚本二仍读出号牌「{d.get('text','')}」"
                return True, "脚本二仍检出人脸"
        return False, ""

    # ---- 主流程 ----
    def run(self):
        pairs, missing = self._pair_files()
        mode = "detection_only（无真值，退化模式）" if self.detection_only else "full（含真值/遮盖率）"
        print(f"运行模式：{mode}")
        print(f"配对成功 {len(pairs)} 张；真值有但检测结果缺失 {len(missing)} 张。")
        if not pairs:
            print("没有可处理的图片，请确认检测输出目录" + ("。" if self.detection_only else "与真值记录目录。"))
            return None

        if self.detection_only:
            return self._run_detection_only(pairs)
        return self._run_full(pairs, missing)

    # ---- 退化模式：只统计「仍可识别目标（漏检下限）」 ----
    def _run_detection_only(self, pairs):
        leak_face = leak_plate = 0
        details = []
        for stem, _gt, det_path in pairs:
            det = load_detection(det_path)
            # 仅统计满足 §5.6.2.1 最小边长的检出（仍可识别即漏检）
            f = sum(1 for d in det["faces"] if box_min_side(d["bbox"]) >= FACE_MIN_SIDE_PX)
            p = sum(1 for d in det["plates"] if box_min_side(d["bbox"]) >= PLATE_MIN_SIDE_PX)
            leak_face += f
            leak_plate += p
            if f or p:
                details.append({"图片": stem, "仍可识别人脸": f, "仍可识别号牌": p})

        results = {
            "评测时间": datetime.now().isoformat(timespec="seconds"),
            "运行模式": "detection_only",
            "配对图片数": len(pairs),
            "仍可识别人脸数（漏检下限）": leak_face,
            "仍可识别号牌数（漏检下限）": leak_plate,
            "明细": details,
            "告警": "缺真值/遮盖率记录，应检数分母不全，无法计算检出率/漏检率/误检率；"
                    "上述仅为脚本二在打码图上仍能识别的目标数，是漏检数的下限。",
            "整体合规": None,
        }
        self._print_and_save(results)
        return results

    # ---- 完整模式：检出率/漏检率/误检率/匿名化效果 ----
    def _run_full(self, pairs, missing):
        # 累加器：face / plate 各一套
        acc = {
            "face": {"应检": 0, "正检": 0, "漏检": 0, "误检": 0, "检出": 0},
            "plate": {"应检": 0, "正检": 0, "漏检": 0, "误检": 0, "检出": 0},
        }
        effect_failures = []   # 匿名化效果不合格清单
        per_image = []

        for stem, gt_path, det_path in pairs:
            truth = load_truth_record(gt_path)
            det = load_detection(det_path)
            img_face = {"正检": 0, "漏检": 0}
            img_plate = {"正检": 0, "漏检": 0}

            # ---------- 检出率/漏检率：遍历真实目标 ----------
            for t in truth["targets"]:
                if not is_required_target(t):
                    continue  # 非应检对象，不计入任何分母
                a = acc[t["type"]]
                a["应检"] += 1
                if t["coverage"] >= self.coverage_pass:
                    a["正检"] += 1
                    (img_face if t["type"] == "face" else img_plate)["正检"] += 1
                    # 匿名化效果（B.6.1）：已打码目标若仍被识别 → 不合格
                    hit, why = self._still_recognizable(t, det)
                    if hit:
                        effect_failures.append({
                            "图片": stem, "类型": t["type"], "bbox": t["bbox"],
                            "遮盖率": round(t["coverage"], 3), "说明": why,
                        })
                else:
                    a["漏检"] += 1
                    (img_face if t["type"] == "face" else img_plate)["漏检"] += 1

            # ---------- 误检率：遍历主机厂标记打码区域 ----------
            for r in truth["marked_regions"]:
                if r["exclude"]:
                    continue  # 国标 C.1/C.4 例外，不计入
                a = acc[r["type"]]
                a["检出"] += 1
                same_type_targets = [t for t in truth["targets"] if t["type"] == r["type"]]
                if not any(boxes_intersect(r["bbox"], t["bbox"]) for t in same_type_targets):
                    a["误检"] += 1

            per_image.append({"图片": stem, "人脸": img_face, "号牌": img_plate})

        results = self._summarize(acc, effect_failures, per_image, len(pairs), len(missing))
        self._print_and_save(results)
        return results

    # ---- 指标汇总 + 判定 ----
    def _summarize(self, acc, effect_failures, per_image, n_pairs, n_missing):
        items = {}
        for typ, name in (("face", "人脸"), ("plate", "号牌")):
            a = acc[typ]
            应检 = a["应检"]
            检出 = a["检出"]
            检出率 = (a["正检"] / 应检) if 应检 > 0 else None
            漏检率 = (a["漏检"] / 应检) if 应检 > 0 else None
            误检率 = (a["误检"] / 检出) if 检出 > 0 else None
            items[name] = {
                "应检数": 应检, "正检数": a["正检"], "漏检数": a["漏检"],
                "误检数": a["误检"], "检出数（标记打码总数）": 检出,
                "检出率": (round(检出率, 4) if 检出率 is not None else None),
                "漏检率": (round(漏检率, 4) if 漏检率 is not None else None),
                "误检率": (round(误检率, 4) if 误检率 is not None else None),
                "检出率达标(≥0.90)": (检出率 >= DETECTION_RATE_PASS) if 检出率 is not None else None,
                "漏检率达标(≤0.10)": (漏检率 <= (1 - DETECTION_RATE_PASS)) if 漏检率 is not None else None,
                "误检率达标(≤0.10)": (误检率 <= FALSE_RATE_PASS) if 误检率 is not None else None,
            }

        # 整体合规：所有可判定的达标项均通过 且 无匿名化效果不合格
        checks = []
        for v in items.values():
            for k in ("检出率达标(≥0.90)", "漏检率达标(≤0.10)", "误检率达标(≤0.10)"):
                if v[k] is not None:
                    checks.append(v[k])
        effect_ok = (len(effect_failures) == 0)
        overall = bool(checks) and all(checks) and effect_ok

        return {
            "评测时间": datetime.now().isoformat(timespec="seconds"),
            "运行模式": "full",
            "配对图片数": n_pairs, "真值缺检测数": n_missing,
            "阈值": {
                "人脸最小边长px": FACE_MIN_SIDE_PX, "号牌最小边长px": PLATE_MIN_SIDE_PX,
                "正检遮盖率": self.coverage_pass,
                "检出率达标线": DETECTION_RATE_PASS, "误检率达标线": FALSE_RATE_PASS,
            },
            "指标": items,
            "匿名化效果不合格数": len(effect_failures),
            "匿名化效果不合格清单": effect_failures,
            "整体合规": overall,
        }

    # ---- 打印 + 保存 ----
    @staticmethod
    def _mark(v):
        return "—" if v is None else ("✅" if v else "❌")

    @staticmethod
    def _pct(v):
        return "N/A" if v is None else f"{v*100:.2f}%"

    def _print_and_save(self, results):
        print("\n" + "=" * 64)
        print(" GB/T 44464-2024 正式检测合规报告（脚本三）")
        print("=" * 64)

        if results.get("运行模式") == "detection_only":
            print(f" 模式: detection_only（无真值，退化）  配对图片: {results['配对图片数']}")
            print(f" 仍可识别人脸数（漏检下限）: {results['仍可识别人脸数（漏检下限）']}")
            print(f" 仍可识别号牌数（漏检下限）: {results['仍可识别号牌数（漏检下限）']}")
            print(f" ⚠ {results['告警']}")
        else:
            for name, v in results["指标"].items():
                print(f"\n [{name}] 应检 {v['应检数']} | 正检 {v['正检数']} | 漏检 {v['漏检数']} | "
                      f"误检 {v['误检数']} | 检出(标记) {v['检出数（标记打码总数）']}")
                print(f"     检出率: {self._pct(v['检出率'])} (≥90%) {self._mark(v['检出率达标(≥0.90)'])}   "
                      f"漏检率: {self._pct(v['漏检率'])} (≤10%) {self._mark(v['漏检率达标(≤0.10)'])}   "
                      f"误检率: {self._pct(v['误检率'])} (≤10%) {self._mark(v['误检率达标(≤0.10)'])}")
            ef = results["匿名化效果不合格数"]
            print(f"\n [匿名化效果] 不合格目标: {ef} 个 {self._mark(ef == 0)}")
            for x in results["匿名化效果不合格清单"][:20]:
                print(f"     - {x['图片']} [{x['类型']}] 遮盖率{x['遮盖率']} → {x['说明']}")
            print("-" * 64)
            print(f" 整体合规: {'✅ 合规' if results['整体合规'] else '❌ 不合规'}")

        print("=" * 64)
        os.makedirs(os.path.dirname(self.report_path) or ".", exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as fp:
            json.dump(results, fp, ensure_ascii=False, indent=2)
        print(f"报告已保存: {self.report_path}")


if __name__ == "__main__":
    print("=" * 64)
    print(" GBT 44464-2024 脚本三 · 正式检测合规计算引擎启动")
    print("=" * 64)

    # ——— 正式检测目录（阶段二，按需调整）———
    # 脚本二在主机厂匿名化数据上的检测输出：
    DET_DIR = r"E:\Vehicle_Data_Anonymization_Verifier\detection\detection_json"
    # 真值/遮盖率记录目录（每图一份 JSON，schema 见 load_truth_record）。
    # 暂无主机厂数据 → 设为 None 走 detection_only 退化模式，仅报漏检下限。
    TRUTH_DIR = None
    REPORT = r"E:\Vehicle_Data_Anonymization_Verifier\detection\report\compliance_report.json"

    calculator = ComplianceCalculator(
        detection_folder=DET_DIR,
        truth_folder=TRUTH_DIR,
        report_path=REPORT,
    )
    calculator.run()
