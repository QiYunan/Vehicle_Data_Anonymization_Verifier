# 项目上下文交接文档

> 供切换 AI 模型 / 会话时快速恢复上下文。
> **最后更新**：2026-06-23

> ⚠️ **测试原则：一律用真实数据测试，绝不用人为编造的样本——编造数据测试没有意义。**

---

## 项目是什么

**汽车数据匿名化合规检测系统**，基于 **GB/T 44464-2024** 国标。

车企采集的行车视频/图片，依法须对车牌和人脸做脱敏处理（打码/模糊）。本系统的角色是**自动化裁判**——检查这些数据是否真的脱敏干净，若仍能识别出车牌或人脸，则判定不合规，最终输出合规报告。

---

## 系统规划：四脚本流水线

| 脚本 | 职责 |
|------|------|
| 脚本一 `gbt_step1_splitter.py` | 合规抽帧：视频 → 1080P 图片集（每 2 秒 1 帧） |
| 脚本二 `gbt_step2_detector.py` | 识别检测：对图片做车牌/人脸检测，输出带坐标的 JSON |
| 脚本三 `gbt_step3_compliance.py` | 正式检测合规计算：吃脚本二输出，算检出率/漏检率/误检率/匿名化效果，出合规判定 |
| 脚本四 *(待建)* | 审计报告：汇总结果，生成最终合规报告 |
| 脚本五 `gbt_step5_evaluator.py` | 自检评测：检测 JSON ↔ labelme 真值 比对，算国标考核项 1~5 并判通关 |

### 两阶段运行逻辑

**阶段一 · 自检**：先用 500 张车牌 + 500 张人脸图（人工标注真实值）测试系统自身识别准确率，达标后才进入阶段二。目的是先证明"裁判自己够格"。

**阶段二 · 正式检测**：用通过自检的系统，对主机厂提交的脱敏数据跑完四个脚本，出合规报告。

自检要求基于GBT+44464_2024标注精度需求跟踪.xlsx

---

## 当前进度

```
整体约 75%

脚本一 ████████████ ✅ 完成
脚本二 ████████████ ✅ 完成（三引擎集成 + JSON + 可视化，已端到端验证；2026-06-16 迭代优化中）
脚本三 ████████████ ✅ 完成（2026-06-23，正式检测合规计算引擎，合成样张验证）
脚本四 ░░░░░░░░░░░░ ❌ 未开始
脚本五 ████████████ ✅ 完成（自检评测器，项1~5，合成样张正/反例验证）
```

> 计算需求权威梳理见 `匿名化与检测要求.md`（按自检/正式检测两阶段拆条，附三个率分母对照）。

---

## 待做事项

### 脚本二（gbt_step2_detector.py）- ✅ 完成（2026-06-12），持续迭代中（2026-06-16）

- [x] 集成 RetinaFace（人脸检测）— zidane.jpg 实测 2 张人脸 conf 0.9996
- [x] 集成 YOLOv8（车牌定位）— 权重 `weights/license_plate_yolov8.pt`，类别 license_plate
- [x] 集成 PaddleOCR（车牌文字识别）— 合成中文车牌实测「京A12345」conf 1.0
- [x] 标准化输出格式：JSON（坐标 / 置信度 / 识别文字 / 文字置信度 / 计数 / 时间戳）
- [x] 输出写入 `opencv_output_json\`，并按视频子目录镜像结构
- [x] 图像可视化（OpenCV 绘制检测框，存 `opencv_output_json\visualization\`）
- [x] 递归扫描脚本一的「每视频一子目录」输出；中文路径用 imdecode/imencode 兼容
- [x] 已用脚本一真实抽帧端到端跑通

> 车牌识别（2026-06-12 第二/三轮，定稿为「两层漏斗」）：
> - **核心理念**：只有读出【完整且符合中国车牌格式】的号码才算车牌；读不出的(遮挡/糊/GB/T误检)一律不算车牌。
>   天然实现国标"遮挡致信息不全→不计"的要求。
> - **第①层 可识别车牌** = 读出合法车牌号(任意尺寸)；**第②层 符合国标车牌** = ①里最小边长≥16px(§5.6.2.1)。
>   两层各自计数 + 逐一列号码(带 minside)，写入 JSON + 控制台汇总。
> - status: standard(符合国标) / small(可识别但<16px) / unread(未读出，不算车牌，单列 unread_boxes 供核对漏读)。
>   JSON 的 `plates` 字段只含已确认车牌(standard+small)，脚本五比对不再被误检污染。
> - **召回**：YOLO `imgsz=1920`(默认640会把远处小牌缩没；实测3200反而漏掉近处过大车牌) + `conf=0.25`；远处小牌根治靠全分辨率源图(1080p抽帧)。
> - **可视化**：PIL 中文——绿框=符合国标+号码；橙框=可识别但<16px；灰细框="未读出(非车牌)"。
> - **车牌格式正则**：省份汉字(不含台港澳首字)+字母+5~6位，含港澳/学/警等尾缀；剔除 GB/T、残缺、misread(如5H61107)。
> - **嵌套框去重**：补 YOLO NMS 漏掉的重复框(IoMin>0.6)，优先级 读出合法车牌>文字置信>面积，保证一牌一框。
> - 已知遗留：远处过小/过糊车牌仍会漏检或读不出(物理极限，用全分辨率原图缓解)；个别省份汉字误读需车牌专用OCR根治。

#### 2026-06-16 迭代优化（已改代码，待提交）

- [x] **`_classify()` 两层漏斗方法**：独立抽离判定逻辑（standard / small / unread），代码更清晰
- [x] **`is_valid_plate()` + `normalize_plate()`**：车牌格式校验拆成独立函数，正则常量化（`_PLATE_RE`），支持去掉分隔符·、空格再校验
- [x] **`_overlap_ratio()`**：独立去重函数（IoMin），供 `detect()` 的嵌套框去重复用
- [x] **`draw_labels_cn()`**：PIL 中文标签渲染独立成全局函数（含描边），`lru_cache` 缓存字体加载
- [x] **JSON 输出字段拆分**：新增 `recognizable_plate_count` / `recognizable_plates` / `standard_plate_count` / `standard_plates` / `unread_box_count`；`plates` 字段语义更精确（只含读出号码的真牌）
- [x] **控制台汇总**：每图输出两层计数 + 全局任务完成后打印汇总表（所有图的可识别/符合国标车牌列表）
- [x] **`_fmt_plates()` 静态方法**：格式化车牌为「号码(Npx) | ...」，标注 <16px 不达国标
- [x] **PlateDetector 构造参数**：新增 `imgsz`、`ocr_conf_threshold`、`min_side_px` 显式参数（原硬编码）
- [x] **Windows 控制台 UTF-8**：顶部加 `sys.stdout.reconfigure(encoding="utf-8")`，彻底解决中文车牌打印乱码

> 关键坑（已解决，备查）：
> - **Python 3.14 装不了 Paddle**：必须用 Python 3.12。已在 E 盘建 venv（见下）。
> - **PaddleOCR 3.x 改 API**：`use_angle_cls/show_log` 废弃，改 `use_textline_orientation`，`.ocr()`→`.predict()`（取 `rec_texts/rec_scores`）。
> - **Paddle 3.3.1 CPU oneDNN 崩溃**：构造 PaddleOCR 时须 `enable_mkldnn=False`。
> - **RetinaFace 需 tf-keras**：脚本顶部已置 `TF_USE_LEGACY_KERAS=1`。
> - **C 盘已满**：脚本顶部 `_redirect_caches_to_e()` 把所有框架缓存强制重定向到 E 盘 `model_cache\`。

---

#### <span style="color:red">⚠️ 2026-06-16 实测遗留问题与待解方案（下轮优先处理）</span>

**实测背景**：10 段 1080p 停车场视频抽帧后跑脚本二，同一辆车（苏U·Y2192）在不同角度的检测结果：

| 帧 | 结果 | 说明 |
|----|------|------|
| 001~003, 005, 008~010 | ✅ 正常识别 | 7/10 帧正常 |
| <span style="color:red">**004**</span> | <span style="color:red">❌ 0 个框，完全漏检</span> | <span style="color:red">倾角最大，YOLO 在该位置激活值接近 0</span> |
| <span style="color:red">**006**</span> | <span style="color:red">⚠️ 灰框"未读出(非车牌)"</span> | <span style="color:red">YOLO 检测到形状，OCR 无法读取斜视文字</span> |
| <span style="color:red">**007**</span> | <span style="color:red">❌ 0 个框，完全漏检</span> | <span style="color:red">倾角次大，同 004</span> |

**同次运行已修复的问题**：
- ✅ RetinaFace 误检停车场地面为人脸（`face 0.56`）→ 新增几何过滤器 `_is_plausible_face()` 后消除
  - 过滤规则：最小边长<20px / 面积占比>8%且置信度<0.90 / 宽高比<0.4或>1.5 / 框中心在画面底部90%以下
  - 置信度门槛从 0.5 调整为 0.65
- ✅ YOLO conf 从 0.25 降至 0.15（扩大召回）
- ✅ OCR 裁切加 8% 边距（防止 bbox 偏小导致字符被截）

---

<span style="color:red">**未解决问题根因分析**</span>

- <span style="color:red">**004/007 完全漏检**：YOLO 输出轴对齐矩形框，车牌极端倾角时透视变形为平行四边形，YOLO 激活值本身接近 0，不是"低于阈值"而是"根本无输出"。降低 conf 无法解决（已验证 conf=0.15 仍漏检），继续降至 0.01~0.05 每帧 OCR 调用增至 20~100 次（速度下降 3~7 倍）且仍对激活为零的帧无效。根治必须换模型。</span>

- <span style="color:red">**006 OCR 失败**：YOLO 检测到形状但 PaddleOCR 对严重倾斜文字识别率低。YOLO 给的是轴对齐矩形框，送入 OCR 的裁切图是斜视图。</span>

---

<span style="color:red">**待实施解决方案（按优先级）**</span>

| 优先级 | 问题 | 方案 | 难度 |
|--------|------|------|------|
| 🔴 高 | 004/007 完全漏检 | **换 YOLOv8-OBB**（Oriented Bounding Box）：输出带旋转角的定向矩形框，专为极端倾角目标设计，需重新下载 OBB 车牌权重 | 中 |
| 🟡 中 | 006 OCR 失败 | **旋转暴力搜索**：对 YOLO 裁切图在 -30°~+30° 内以 10° 步长逐角度旋转，每角度跑一次 OCR，取通过 `_PLATE_RE` 正则的结果。简单，`cv2.rotate` 几行代码，但每个 unread 框多跑 6~12 次 OCR（约增加 2~5 秒/帧） | 低 |
| 🟢 低 | 006 OCR 失败（根治） | **透视矫正**：在裁切图内用边缘检测找车牌真实四角点，`cv2.getPerspectiveTransform + warpPerspective` 拉平后送 OCR，效果最好但需精确角点检测，实现复杂；对 004/007（无裁切图）无效 | 高 |

> **关于 conf 降至 0 的讨论结论**：OCR 正则 `_PLATE_RE` 是强过滤器（非车牌区域极难通过省份+字母+数字格式验证），理论上可以靠它兜底。但 conf 降至 0.01 以下每帧 OCR 调用量爆增，且对激活值本身为零的极端倾角帧（004/007）完全无效。**结论：conf 维持 0.15，根治靠换 YOLOv8-OBB**。

### 脚本三（gbt_step3_compliance.py）- ✅ 完成（2026-06-23）

正式检测合规计算引擎（阶段二）。吃脚本二在主机厂匿名化数据上的检测 JSON，结合「真值/遮盖率记录」算四大指标并判合规。需求依据见 `匿名化与检测要求.md`。

- [x] §5.6.2.1 应检对象判定（人脸 ≥32px 且可见范围>50%；号牌 ≥16px 且可识别）
- [x] 检出率/漏检率：应检目标按遮盖率≥50% 分正检/漏检；检出率=正检/应检，≥90% 达标
- [x] 误检率（视作强制）：标记打码区与任一同类真实目标无交集→误检；误检/检出数，≤10% 达标；支持 exclude 标记 C.1/C.4 例外
- [x] 匿名化效果（B.6.1 机器识别）：对正检目标查脚本二是否仍检出/读出→不合格清单（第二套模型留接口）
- [x] 真值/遮盖率记录 schema（targets: type/bbox/visible_ratio/coverage/readable/plate_text + marked_regions）
- [x] **detection_only 退化模式**：无真值时只报「仍可识别目标数（漏检下限）」+ 告警（当前无主机厂数据可用此模式跑通）
- [x] 报告 compliance_report.json + 控制台（✅/❌，UTF-8）；合成正/反例自测通过

> 关键设计：打码图上脚本二「仍能检出/读出 = 仍可识别 = 漏检/匿名化效果不合格」；打好码目标脚本二看不见，其正检身份由真值记录给出。遮盖率/匿名化区域来源做成可切换接口（主机厂数据到位前未定）。

### 脚本四（待建）
- [ ] 汇总检测结果
- [ ] 生成最终合规报告

### 脚本五（gbt_step5_evaluator.py）- ✅ 完成（2026-06-12，本期项 1~5）

- [x] 解析 labelme 矩形标注真值（label：`face` / `plate:车牌号`）
- [x] 检测 JSON ↔ 真值 按相对路径对齐，按 IoU(默认0.5) 贪心配对
- [x] 项1 人脸框比值（≥98%∈[0.9,1.1]）/ 项2 小车牌边长MAE(≤1px) / 项3 大车牌边长比值([0.9,1.1]) / 项5 计数比([0.99,1.01])
- [x] 项4 可见范围面积比值：留接口（需 labelme 加标「可见范围」多边形 + 脚本二输出可见范围，暂 N/A）
- [x] 合成样张正/反例验证：正例全 ✅、反例计数越界判 ❌
- [x] 强制 UTF-8 stdout（Win 控制台打印 ✅/中文不崩）

> 国标锚点（§5.6.2.1）：人脸框最小边长 ≥32px、车牌框最小边长 ≥16px 才属匿名化对象；
> 「小/大车牌」分界用参数 `small_plate_max_side`，默认 32px（国标未直接定义，可调）。
> 项6~7（已处理图遮盖率/漏检比值，§B.5）属脚本三/四范畴，已留接口后置。

**自检操作流程**（拿到 1000 张数据后）：
1. 用 labelme 在 500 未处理图上画框：人脸 `face`、车牌 `plate:车牌号`（每图导出同名 JSON）
2. 跑脚本二出检测 JSON（→ `opencv_output_json\`）
3. 跑脚本五：配置 `GT_DIR`(labelme 目录) 与 `DET_DIR`，得 `self_check_report.json` + 终端通关判定

---

## 技术方案（已确定）

### 检测模型栈（方案A — 准确率优先）

| 模块 | 技术方案 | 职责 |
|------|--------|------|
| **人脸检测** | RetinaFace | 离线开源，准确率业界最高，处理多尺度人脸 |
| **车牌定位** | YOLOv8 (license-plate) | 通用目标检测，轻量化 |
| **车牌识别** | PaddleOCR (中文车牌模块) | 离线百度开源，中国车牌准确率 >98% |

**特性**：
- ✅ 全离线部署，隐私数据100%本地处理
- ✅ 准确率优先（无时间限制）
- ✅ OpenCV 保留用于图像 I/O、坐标绘制、结果可视化

**自检时间表**：
- 当前阶段：完成脚本一～四的完整框架实现
- 项目审批后获得 1000 张自检数据 → 执行阶段一自检
- 自检通过 → 执行阶段二正式检测

---

## 运行环境（2026-06-12 搭建）

- **解释器**：Python 3.12.8，装在 `E:\Python312`（系统默认的 3.14/3.13 装不了 Paddle）
- **虚拟环境**：`E:\Vehicle_Data_Anonymization_Verifier\venv312`
  运行脚本二：`E:\Vehicle_Data_Anonymization_Verifier\venv312\Scripts\python.exe gbt_step2_detector.py`
- **已装库**：ultralytics 8.4 / paddlepaddle 3.3.1 / paddleocr 3.7 / retina-face 0.0.18 / tf-keras / opencv / **torch 2.12.0+cu126 / torchvision 0.27.0+cu126**（GPU 版）/ tensorflow 2.21 / **spiga 0.0.6**
- **GPU（2026-06-24 启用）**：RTX 4060 Laptop 8GB，驱动 555.97/CUDA12.5；torch 用 cu126（次版本兼容 12.5 驱动，无需升级）。`torch.cuda.is_available()=True`。
  - **上 GPU**：YOLO 车牌、SPIGA 人脸关键点、CRNN（均 torch）。**仍 CPU**：RetinaFace（TF，Win 无原生 GPU）、PaddleOCR（Paddle CPU 版）。
  - torch CUDA 版是超集，CPU/GPU 都能跑；换回 CPU 版：`pip install torch==2.12.0 torchvision==0.27.0`（默认 PyPI 即 CPU）。
  - 重装 GPU 版：`pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.12.0 torchvision==0.27.0`（官方源慢/断流时，用 curl 断点续传下 wheel 再本地装）。
- **人脸边界框矫正（2026-06-24）**：RetinaFace 只给 5 点不含眉毛，不符国标 §3.5。改用 **SPIGA(WFLW 98点)** 量出「眉毛最上沿→颏底线、耳间不含耳」三边。
  - 封装 `spiga_face_landmarks.py`（含 CPU/GPU 自适应 + 对 SPIGA 写死 cuda 的运行时补丁）；FaceDetector 优先用 SPIGA，失败退回 5 点比例法。
  - SPIGA 权重 255MB 在 `model_cache\deepface\..`? 实际在 `venv312\Lib\site-packages\spiga\models\weights\spiga_wflw.pt`（gdown 从 Google Drive 下，避开 C 盘）。GPU 热推理 ~0.14s/脸。
- **模型缓存**：框架权重强制落 `E:\Vehicle_Data_Anonymization_Verifier\model_cache\`（原 C 盘满；现 C 已腾出 500G，但缓存仍留 E）
  - venv 的 pip 缓存也已固定到 E（`venv312\pip.ini`）
- **车牌权重**：`weights\license_plate_yolov8.pt`（来源 HF `Koushim/yolov8-license-plate-detection`，经 hf-mirror.com 下载）

### 怎么运行（重要）
- **不能用 VSCode 默认"运行"按钮**（会用系统 Python 3.14，缺库必报错）。
- 双击 `.bat` 启动器（在仓库目录，自动用 E 盘 venv）：
  - `运行_检测_选择文件.bat` ← **主入口**（2026-06-25）：弹文件框**勾选照片/视频(可混选、多轮跨文件夹累加)** → 视频抽帧+照片直采 → 脚本二检测。调 `gbt_run_detection.py`，输出到 `detection_result\run_N_时间戳\`。
  - `运行_脚本5_自检评测.bat`、`打开_labelme标注.bat`
  - 已删旧 bat：`运行_完整流水线_抽帧+检测.bat`/`运行_脚本2_检测.bat`/`运行_脚本2_照片检测.bat`（被统一入口取代；连带 `_find_latest_run.py`/`_make_run_dir.py` 已无用）。
- 或终端用全路径：`E:\...\venv312\Scripts\python.exe 脚本.py`
- **数据规格**：所有样本统一使用 **1080p 视频**抽帧（之前用的 1707px 图片不再使用）
- 标注工具 **labelme 6.3.1** 已装进 venv，真值 `.json` 默认存图片旁。

### 测试目录结构（数据均在仓库外，不入库）— 2026-06-23 改名：self_check→test、unmasked→sample
```
E:\Vehicle_Data_Anonymization_Verifier\
├─ test\                       测试总目录
│  ├─ sample\                  未打码样本
│  │  ├─ video\                视频源 → 脚本一抽帧
│  │  └─ image\                ★直接放照片处（脚本二照片直检读这里）+ labelme真值.json存图片旁
│  ├─ masked\                  已打码（项6~7，结构 video\ + images\）
│  ├─ detection_result\        ★脚本二输出：run_N_时间戳\{images,json,visualization}
│  ├─ detection_json\          脚本二默认输出（无参时）/ 脚本五读取
│  └─ report\                  脚本五自检报告
├─ model_cache\  venv312\  Vehicle_Data_Anonymization_Verifier\(代码仓库)
```
用法（2026-06-25 起统一入口）：
- **检测（照片/视频混选）**：照片放 `test\sample\image`、视频放 `test\sample\video`（也可放任意处）→ 双击 `运行_检测_选择文件.bat`
  → 弹框勾选要检测的文件（可多轮跨文件夹累加）→ 视频抽帧+照片直采 → 脚本二检测
  → 输出 `detection_result\run_N_时间戳\{images\photos, images\<视频名>, json, visualization}`。
- 自检评测：在 `image` 上用 labelme 标注 → 脚本五比对（GT_DIR/DET_DIR 见脚本五 `__main__`）。
- 阶段二（正式检测、按客户分目录）：待客户数据到位后再设计，当前不建。

---

## 关键参考

- **国标**：GB/T 44464-2024
`"E:\Vehicle_Data_Anonymization_Verifier\GBT+44464-2024.pdf"`
- **架构笔记**：`E:\Vehicle_Data_Anonymization_Verifier\gemini-code-...txt`
- **代码仓库**：`E:\Vehicle_Data_Anonymization_Verifier\Vehicle_Data_Anonymization_Verifier\`
- **标注精度要求**:
`E:\Vehicle_Data_Anonymization_Verifier\GBT+44464_2024标注精度需求跟踪.xlsx"`
