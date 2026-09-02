"""FastAPI line endpoint. No page OCR route."""

from __future__ import annotations

import io
import os
import time
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import torch

from pipeline.training.infer import recognize_line
from pipeline.training.model_contract import (
    apply_trocr_generation_config,
    load_htr_model,
    load_htr_processor,
    resolve_htr_device,
)
from pipeline.training.ship_gate import PROVEN_SHIP_MODEL, assert_shippable_checkpoint

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MIN_DIM = 16


def create_app(
    model: Any = None,
    processor: Any = None,
    model_id: str = PROVEN_SHIP_MODEL,
    preload: bool = True,
    num_beams: int = 4,
) -> FastAPI:
    assert_shippable_checkpoint(model_id)
    app = FastAPI(title="Offline line HTR", version="1.0.0")
    state: dict[str, Any] = {
        "model": model,
        "processor": processor,
        "model_id": model_id,
        "num_beams": num_beams,
    }

    @app.on_event("startup")
    def _startup() -> None:
        if not preload:
            return
        if state["model"] is None or state["processor"] is None:
            state["processor"] = load_htr_processor(model_id)
            state["model"] = load_htr_model(model_id)
            apply_trocr_generation_config(state["model"], state["processor"], num_beams=state["num_beams"])
        device = resolve_htr_device()
        state["model"].to(device)
        dummy = Image.new("RGB", (384, 384), color="white")
        try:
            recognize_line(
                state["model"],
                state["processor"],
                dummy,
                device=device,
                num_beams=state["num_beams"],
            )
        except Exception:
            pass

    @app.post("/v1/recognize-line")
    async def recognize_line_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="file larger than 10 MB")
        try:
            image = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"not an image: {exc}") from exc
        if min(image.size) < MIN_DIM:
            raise HTTPException(status_code=400, detail="crop smaller than 16 px")
        if state["model"] is None or state["processor"] is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        t0 = time.perf_counter()
        text = recognize_line(
            state["model"],
            state["processor"],
            image,
            num_beams=state["num_beams"],
        )
        ms = (time.perf_counter() - t0) * 1000.0
        return {"text": text, "ms": round(ms, 2), "model_id": state["model_id"]}

    return app


def main() -> None:
    import uvicorn

    model_id = assert_shippable_checkpoint(os.environ.get("HTR_MODEL_ID", PROVEN_SHIP_MODEL))
    num_beams = int(os.environ.get("HTR_NUM_BEAMS", "4"))
    app = create_app(model_id=model_id, preload=True, num_beams=num_beams)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8088")), workers=1)


if __name__ == "__main__":
    main()
