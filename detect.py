#!/usr/bin/env python3
# Butterfly Detector — single-class YOLO detector for locating butterflies.
# Copyright (C) 2026  dx76my-ctrl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
蝴蝶检测器（单类目标检测）：在一张图里框出每一只蝴蝶，返回归一化坐标框。
仅做检测、不做物种分类。

用法：
  python detect.py path/to/image.jpg
  python detect.py path/to/image.jpg --conf 0.25 --save out.jpg
作为库：
  from detect import detect
  boxes = detect("image.jpg")            # -> [{"bbox":[x1,y1,x2,y2], "conf":..}]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = Path(__file__).resolve().parent / "weights" / "butterfly_detector.pt"
_model: YOLO | None = None


def _load() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(WEIGHTS))
    return _model


def detect(image, conf: float = 0.25, iou: float = 0.55, imgsz: int = 640,
           device: str = "cpu") -> list[dict]:
    """检测蝴蝶。返回归一化框列表 [{"bbox":[x1,y1,x2,y2] in 0~1, "conf": float}]。"""
    r = _load().predict(source=image, conf=conf, iou=iou, imgsz=imgsz,
                        device=device, verbose=False)[0]
    h, w = r.orig_shape
    out = []
    if r.boxes is not None:
        for b, c in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist()):
            out.append({"bbox": [round(b[0] / w, 4), round(b[1] / h, 4),
                                 round(b[2] / w, 4), round(b[3] / h, 4)],
                        "conf": round(float(c), 4)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="蝴蝶检测器（单类）")
    ap.add_argument("image", help="输入图片路径")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    ap.add_argument("--save", default="", help="可选：保存带框预览图到此路径")
    args = ap.parse_args()

    boxes = detect(args.image, conf=args.conf, iou=args.iou,
                   imgsz=args.imgsz, device=args.device)
    print(f"检出 {len(boxes)} 只蝴蝶：")
    for i, b in enumerate(boxes, 1):
        print(f"  {i}. bbox(归一化)={b['bbox']}  conf={b['conf']}")

    if args.save:
        from PIL import Image, ImageDraw
        im = Image.open(args.image).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(70, 192, 141), width=3)
        im.save(args.save)
        print(f"预览已保存 -> {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
