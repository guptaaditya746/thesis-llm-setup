"""GLiNER (a small named-entity model) as an HTTP service on loopback, next to a model on one GPU.

Optional: only a profile with ``gliner.enabled: true`` starts it (supervisor.py), and it needs the
``gliner`` extra (``uv sync --extra serve --extra gliner``). The thesis evidence engine calls it:

    GET  /health   -> {"status": "ok"}                       (the supervisor's and doctor's check)
    GET  /info     -> {"model": <id>, "revision": <commit or null>, "device": <device>, "max_chars": <int>}
    POST /predict  {"text": str, "labels": [str, ...], "threshold": float in [0, 1], default 0.5}
                   -> {"entities": [{"start", "end", "text", "label", "score"}, ...]}
                   -> HTTP 413 when text is longer than max_chars (default 1500), which /info reports:
                      GLiNER would cut a long text off silently, so the caller must send smaller chunks

``start``/``end`` are character offsets into ``text``; entities are sorted by (start, end, label).
The model loads before the server listens, so /health answers only once it is on the GPU. gliner and
torch are imported inside ``load_model`` only, so this module (and its tests) run without them.

    python -m llm_setup.gliner_server --model urchade/gliner_medium-v2.1 --revision <sha> --port 8090
"""

from __future__ import annotations

import argparse
import os
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str
    labels: list[str] = Field(min_length=1)
    threshold: float = Field(0.5, ge=0.0, le=1.0)


MAX_CHARS = 1500     # the evidence engine's chunk size; about 250 words, inside GLiNER medium's 384


def create_app(model: Any, model_name: str, revision: str | None = None, device: str = "cuda",
               max_chars: int = MAX_CHARS) -> FastAPI:
    """The service around a loaded model: anything with GLiNER's ``predict_entities(text, labels,
    threshold=...)`` returning dicts with start, end, text, label and score."""
    app = FastAPI(title="gliner")
    lock = threading.Lock()          # one prediction at a time on the GPU

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {"model": model_name, "revision": revision, "device": device, "max_chars": max_chars}

    @app.post("/predict")
    def predict(request: PredictRequest) -> dict[str, list[dict[str, Any]]]:
        if len(request.text) > max_chars:
            raise HTTPException(413, f"text has {len(request.text)} characters; at most {max_chars} "
                                     "(send smaller chunks: GLiNER would cut it off)")
        if not request.text.strip():
            return {"entities": []}
        with lock:
            found = model.predict_entities(request.text, request.labels, threshold=request.threshold)
        entities = [{"start": int(e["start"]), "end": int(e["end"]), "text": str(e["text"]),
                     "label": str(e["label"]), "score": float(e["score"])} for e in found]
        return {"entities": sorted(entities, key=lambda e: (e["start"], e["end"], e["label"]))}

    return app


def load_model(model_name: str, revision: str | None, device: str) -> Any:
    """Load the pinned GLiNER weights onto ``device``; offline when HF_HUB_OFFLINE=1 (as the supervisor
    sets it), so the weights and the backbone's tokenizer must already be in the Hugging Face cache."""
    from gliner import GLiNER  # imports torch: only here

    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    model = GLiNER.from_pretrained(model_name, revision=revision, local_files_only=offline,
                                   map_location=device)
    model = model.to(device)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llm_setup.gliner_server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default="cuda", help="CUDA_VISIBLE_DEVICES picks the GPU; default cuda")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS, help="longest /predict text (HTTP 413 above)")
    args = parser.parse_args(argv)
    print(f"loading GLiNER {args.model} ({args.revision or 'no pinned revision'}) on {args.device}", flush=True)
    model = load_model(args.model, args.revision, args.device)
    print("GLiNER loaded; serving", flush=True)
    import uvicorn

    uvicorn.run(create_app(model, args.model, args.revision, args.device, args.max_chars), host="127.0.0.1",
                port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
