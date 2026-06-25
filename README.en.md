# Butterfly Detector

[English](README.en.md) · [中文](README.md)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Model: YOLO26s](https://img.shields.io/badge/Model-YOLO26s-46c08d.svg)](#)

A **single-class object detector** that **locates every butterfly in an image** (it does *not* classify species).
Built on [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26s and trained on ~6,300 hand-annotated real-world butterfly photos, with good robustness to closed wings, camouflage, and multi-target scenes.

> This repository contains **only the detector** and its inference code. Species classification ("which butterfly is this") is out of scope.

## ✨ Features

- 🦋 **Single-class detection** — decides only "is this a butterfly, and where", generalizing to unseen species
- 🪶 **Lightweight** — YOLO26s, 9.9M parameters, real-time on CPU
- 🎯 **Recall-first** — keeps high recall on real multi-butterfly / camouflaged scenes
- 🔌 **Ready to use** — CLI, Python library, and HTTP service

## 📊 Performance

On a held-out set of 235 hand-annotated real butterfly photos (single class, IoU=0.6):

| Metric | Value |
|---|---|
| Precision | 0.942 |
| **Recall** | **0.924** |
| mAP@50 | 0.953 |
| mAP@50-95 | 0.813 |

Model: YOLO26s · 9,948,638 parameters · single class `butterfly` · training resolution 640.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
```

**Command line:**
```bash
python detect.py photo.jpg                 # print detected normalized boxes
python detect.py photo.jpg --save out.jpg  # also save an annotated preview
python detect.py photo.jpg --device cuda   # GPU; use mps for Apple Silicon
```

**As a Python library:**
```python
from detect import detect
boxes = detect("photo.jpg", conf=0.25)
# -> [{"bbox": [x1, y1, x2, y2], "conf": 0.93}, ...]   bbox is normalized 0~1
```

**As an HTTP service:**
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
# POST /detect  (multipart: image)  ->  {"count": N, "detections": [{bbox, conf}]}
curl -F "image=@photo.jpg" http://127.0.0.1:8000/detect
```

## 🧠 Training Notes

- **Data**: ~6,300 real butterfly photos, manually annotated with single-class (butterfly) YOLO bounding boxes.
- **Base**: fine-tuned from YOLO26s pretrained weights.
- **Highlights**: training resolution 640; mixed multi-target samples to strengthen recall on "many butterflies per image" and hard cases (closed wings, camouflage).
- **Lesson learned**: using generic backgrounds (e.g. random COCO images) as negatives lowers recall and does not match the real false-positive sources; this model prioritizes recall.

`data.yaml` provides the standard single-class dataset config, so training can be reproduced with `ultralytics`.

## 📦 Repository Layout

```
butterfly-detector/
├── weights/butterfly_detector.pt   # trained detection weights (20MB)
├── detect.py                       # inference: CLI + Python function
├── serve.py                        # inference: FastAPI HTTP service
├── data.yaml                       # single-class dataset config
├── requirements.txt
├── LICENSE                         # AGPL-3.0
└── README.md
```

## 📄 License

This project is released under the **GNU AGPL-3.0**.

The model is trained with Ultralytics YOLO (AGPL-3.0); per its license, this derivative work is also licensed under **AGPL-3.0**. This means:

- You are free to use, modify, and redistribute this project;
- If you **modify it and offer it as a network service**, you must provide the **complete corresponding source code** to its users (AGPL §13);
- Derivative works must also be released under AGPL-3.0.

> To use Ultralytics YOLO in a closed-source / commercial setting without AGPL obligations, you may purchase an [Enterprise License](https://www.ultralytics.com/license) from Ultralytics.

⚠️ Replace this repository's `LICENSE` file with the full official AGPL-3.0 text (GitHub's "GNU AGPLv3" license template inserts it automatically).

## 🙏 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Annotations from real butterfly observation photos (manually boxed)

## 📌 Citation

```bibtex
@software{butterfly_detector,
  title  = {Butterfly Detector: a single-class YOLO detector for butterflies},
  year   = {2026},
  note   = {Trained with Ultralytics YOLO26s, released under AGPL-3.0}
}
```
