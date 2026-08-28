"""
main.py — FastAPI application.

Endpoints:
    GET  /                     → the sculpture (index.html)
    GET  /api/manifest         → latest walk manifest (frames metadata)
    POST /api/generate         → launch a latent-walk render (async job)
    GET  /api/jobs/{job_id}    → poll generation progress
    GET  /frames/...           → the pre-rendered frame sequences (static)

Run:
    uvicorn backend.main:app --reload
"""
from __future__ import annotations

import json
import threading
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .latent_walk import EASINGS, LatentWalker, WalkConfig

settings = get_settings()
settings.ensure_dirs()   # StaticFiles requires dirs to exist at import time


# --------------------------------------------------------------------------- #
# Minimal in-memory job registry                                               #
# (single-process by design — swap for Celery/RQ + Redis for multi-worker)     #
# --------------------------------------------------------------------------- #
class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, params: dict) -> str:
        jid = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[jid] = {
                "status": "queued", "progress": 0.0, "message": "queued",
                "params": params, "result": None, "error": None,
            }
        return jid

    def update(self, jid: str, **kw: Any) -> None:
        with self._lock:
            self._jobs[jid].update(kw)

    def get(self, jid: str) -> Dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(jid)
            return dict(job) if job else None

    def any_running(self) -> bool:
        with self._lock:
            return any(j["status"] in ("queued", "running") for j in self._jobs.values())


JOBS = JobRegistry()


# --------------------------------------------------------------------------- #
# Schemas                                                                       #
# --------------------------------------------------------------------------- #
class GenerateRequest(BaseModel):
    steps_per_transition: int = Field(48, ge=8, le=240)
    fps: int = Field(30, ge=12, le=60)
    dwell: float = Field(0.18, ge=0.0, le=0.6)
    easing: str = "smootherstep"
    noise_level: float = Field(0.035, ge=0.0, le=0.3)
    seed: int = 42
    max_keyframes: int = Field(24, ge=0, le=512)


# --------------------------------------------------------------------------- #
# Background generation worker                                                  #
# --------------------------------------------------------------------------- #
def _run_generation(job_id: str, req: GenerateRequest) -> None:
    """Runs in a daemon thread: encode → walk → decode → manifest."""
    try:
        paths = sorted(p for p in settings.processed_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if len(paths) < 2:
            raise RuntimeError("Need ≥ 2 processed images — run `python -m backend.ingest` first.")

        walker = LatentWalker(settings)
        cfg = WalkConfig(
            steps_per_transition=req.steps_per_transition, fps=req.fps,
            dwell=req.dwell, easing=req.easing, noise_level=req.noise_level,
            seed=req.seed, max_keyframes=req.max_keyframes,
        )
        # The walker reports (done, total, msg) → expose it as job progress.
        JOBS.update(job_id, status="running", message="starting")
        manifest = walker.render_walk(
            paths, cfg,
            progress=lambda d, t, m: JOBS.update(
                job_id, status="running", progress=round(d / max(t, 1), 4), message=m),
        )
        JOBS.update(job_id, status="done", progress=1.0,
                    message="done", result={"walk_id": manifest["id"],
                                            "frame_count": manifest["frame_count"]})
    except Exception as exc:
        traceback.print_exc()
        JOBS.update(job_id, status="failed", error=str(exc), message=f"failed: {exc}")


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    yield


app = FastAPI(title="Machine Hallucinations", version="1.0.0", lifespan=lifespan)
app.add_middleware(  # permissive for local dev; tighten for production
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "device_hint": "cuda" if _cuda() else "cpu"}


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


@app.post("/api/generate", status_code=202)
def generate(req: GenerateRequest) -> dict:
    if req.easing not in EASINGS:
        raise HTTPException(422, f"easing must be one of {list(EASINGS)}")
    if JOBS.any_running():
        raise HTTPException(409, "A generation job is already running.")
    jid = JOBS.create(req.model_dump())
    threading.Thread(target=_run_generation, args=(jid, req), daemon=True).start()
    return {"job_id": jid, "status_url": f"/api/jobs/{jid}"}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j


@app.get("/api/manifest")
def manifest() -> dict:
    """Resolve the 'latest' pointer → the active walk's manifest.json."""
    pointer = settings.frames_dir / "latest.json"
    if not pointer.exists():
        raise HTTPException(404, "No latent walk generated yet.")
    walk_id = json.loads(pointer.read_text())["id"]
    mfile = settings.frames_dir / walk_id / "manifest.json"
    if not mfile.exists():
        raise HTTPException(404, f"walk '{walk_id}' not found on disk")
    return json.loads(mfile.read_text())


# --- static assets ------------------------------------------------------------
app.mount("/frames", StaticFiles(directory=settings.frames_dir), name="frames")
app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")
