# Butterfly Detector · 蝴蝶检测器

[English](README.md) · [中文](README.zh-CN.md)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Model: YOLO26s](https://img.shields.io/badge/Model-YOLO26s-46c08d.svg)](#)

一个**单类目标检测模型**，用于在图像中**框出每一只蝴蝶的位置**（不做物种分类）。
基于 [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26s，在约 6300 张人工标注的真实蝴蝶照片上训练而成，对收翅、伪装、多目标场景有较好鲁棒性。

> 本仓库**只包含检测器**及其推理源码；物种分类（识别"是什么蝴蝶"）不在本仓库范围内。

## ✨ 特性

- 🦋 **单类检测**：只判定"是不是蝴蝶、在哪"，泛化到未见过的物种
- 🪶 **轻量**：YOLO26s，9.9M 参数，CPU 可实时推理
- 🎯 **召回优先**：在真实多蝶/伪装场景下保持高召回
- 🔌 **开箱即用**：CLI、Python 库、HTTP 服务三种用法

## 📊 性能

在 235 张人工标注的真实蝴蝶照片留出集上（单类，IoU=0.6）：

| 指标 | 数值 |
|---|---|
| Precision | 0.942 |
| **Recall** | **0.924** |
| mAP@50 | 0.953 |
| mAP@50-95 | 0.813 |

模型规格：YOLO26s · 9,948,638 参数 · 单类 `butterfly` · 训练分辨率 640。

## 🚀 快速开始

```bash
pip install -r requirements.txt
```

**命令行：**
```bash
python detect.py photo.jpg                 # 打印检出的归一化框
python detect.py photo.jpg --save out.jpg  # 同时保存带框预览
python detect.py photo.jpg --device cuda   # GPU；mps 为 Apple 芯片
```

**作为 Python 库：**
```python
from detect import detect
boxes = detect("photo.jpg", conf=0.25)
# -> [{"bbox": [x1, y1, x2, y2], "conf": 0.93}, ...]   bbox 为归一化 0~1 坐标
```

**作为 HTTP 服务：**
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
# POST /detect  (multipart: image)  ->  {"count": N, "detections": [{bbox, conf}]}
curl -F "image=@photo.jpg" http://127.0.0.1:8000/detect
```

## 🧠 训练说明

- **数据**：约 6300 张真实蝴蝶照片，人工标注 YOLO 单类边界框（butterfly）。
- **基座**：YOLO26s 预训练权重微调。
- **要点**：训练分辨率 640；混合多目标样本，强化"一图多蝶"与收翅/伪装等硬样本的召回。
- **经验**：通用背景（如 COCO 随机图）当负样本会压低召回且与真实误检源不匹配；本模型以保召回为优先。

`data.yaml` 给出了单类数据集的标准配置，可据此用 `ultralytics` 复现训练。

## 📦 仓库结构

```
butterfly-detector/
├── weights/butterfly_detector.pt   # 训练好的检测权重（20MB）
├── detect.py                       # 推理：CLI + Python 函数
├── serve.py                        # 推理：FastAPI HTTP 服务
├── data.yaml                       # 单类数据集配置
├── requirements.txt
├── LICENSE                         # AGPL-3.0
└── README.md
```

## 📄 许可证 License

本项目以 **GNU AGPL-3.0** 开源。

本模型基于 Ultralytics YOLO（AGPL-3.0）训练，依据其许可，本衍生作品同样采用 **AGPL-3.0**。这意味着：

- 你可以自由使用、修改、再分发本项目；
- 若你**修改本项目并作为网络服务对外提供**，须向使用者提供**完整对应源码**（AGPL §13）；
- 衍生作品须同样以 AGPL-3.0 开源。

> 如需在闭源/商业场景中使用 Ultralytics YOLO 而不受 AGPL 约束，可向 Ultralytics 购买[企业授权](https://www.ultralytics.com/license)。

⚠️ 请将本仓库的 `LICENSE` 文件替换为 AGPL-3.0 官方完整文本（GitHub 创建文件时选择 "GNU AGPLv3" 模板即可自动填入）。

## 🙏 致谢 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- 标注数据来自真实蝴蝶观测照片（人工逐框标注）

## 📌 引用 Citation

```bibtex
@software{butterfly_detector,
  title  = {Butterfly Detector: a single-class YOLO detector for butterflies},
  year   = {2026},
  note   = {Trained with Ultralytics YOLO26s, released under AGPL-3.0}
}
```
