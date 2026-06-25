#!/usr/bin/env python3
# Butterfly Detector inference service.
# Copyright (C) 2026  dx76my-ctrl
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.  See <https://www.gnu.org/licenses/>.
"""
蝴蝶检测推理服务（HTTP）。仅做检测、不含物种分类。

启动：  uvicorn serve:app --host 0.0.0.0 --port 8000
接口：
  GET  /health           健康检查
  POST /detect           multipart image → {count, detections:[{bbox,conf}]}
"""
from __future__ import annotations

import threading
from io import BytesIO

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from detect import detect

app = FastAPI(title="Butterfly Detector", version="1.0.0")
_lock = threading.Lock()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/detect")
async def detect_api(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.55),
    imgsz: int = Form(640),
):
    try:
        img = Image.open(BytesIO(await image.read())).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": f"无法读取图片: {exc}"}, status_code=400)
    with _lock:  # CPU 串行
        boxes = detect(img, conf=conf, iou=iou, imgsz=imgsz)
    return {"ok": True, "count": len(boxes), "detections": boxes}
