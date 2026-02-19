from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import uuid
from pathlib import Path

import cv2
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

app = FastAPI(title="VideoRLM Web")

UPLOAD_DIR = Path("/tmp/rlm_web_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "web_static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


def _seconds_to_label(seconds: float) -> str:
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    frac = seconds - int(seconds)
    frac_str = f".{int(frac * 10)}" if frac >= 0.05 else ""
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}{frac_str}"
    return f"{m}:{s:02d}{frac_str}"


def _parse_timestamps(text: str) -> list[dict]:
    found: list[float] = []

    for m in re.finditer(r"\[TS:\s*(\d+(?:\.\d+)?)\s*(?:s)?\]", text, re.IGNORECASE):
        found.append(float(m.group(1)))

    for m in re.finditer(r"\[TS:\s*(\d+):(\d{2})(?::(\d{2}))?\]", text, re.IGNORECASE):
        g = m.groups()
        if g[2] is not None:
            t = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2])
        else:
            t = int(g[0]) * 60 + int(g[1])
        found.append(float(t))

    seen: set[float] = set()
    unique: list[dict] = []
    for t in found:
        if t not in seen:
            seen.add(t)
            unique.append({"seconds": t, "label": _seconds_to_label(t)})

    return sorted(unique, key=lambda x: x["seconds"])


def _render_answer_html(text: str) -> str:
    def replacer(m: re.Match) -> str:
        raw = m.group(0)
        try:
            if ":" in m.group(1):
                parts = m.group(1).split(":")
                if len(parts) == 3:
                    t = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                else:
                    t = int(parts[0]) * 60 + int(parts[1])
            else:
                t = float(m.group(1))
        except ValueError:
            return raw
        label = _seconds_to_label(float(t))
        return (
            f'<button class="ts-inline" data-t="{t}" '
            f'onclick="seekTo({t})">'
            f'⏱ {label}</button>'
        )

    pattern = r"\[TS:\s*([\d:.]+)\s*(?:s)?\]"
    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)


SCENE_MODEL = "facebook/vjepa2-vitl-fpc64-256"
VISUAL_EMBED_MODEL = "google/siglip2-base-patch16-256"
TEXT_EMBED_MODEL = "google/embedding-gemma-300m"


@app.get("/api/arch")
async def arch_info():
    return JSONResponse({
        "scene_model": SCENE_MODEL,
        "visual_embed_model": VISUAL_EMBED_MODEL,
        "text_embed_model": TEXT_EMBED_MODEL,
    })


# Lazily loaded V-JEPA indexer — persists across requests so the model loads only once.
_vjepa_indexer = None
_log = logging.getLogger(__name__)


def _get_vjepa_indexer():
    global _vjepa_indexer
    if _vjepa_indexer is None:
        from rlm.video.video_indexer import VideoIndexer
        _vjepa_indexer = VideoIndexer(
            embedding_model=VISUAL_EMBED_MODEL,
            scene_model=SCENE_MODEL,
        )
    return _vjepa_indexer


def _vjepa_keyframes(
    video_path: str,
    max_frames: int = 12,
    resize: tuple[int, int] = (640, 480),
    quality: int = 75,
) -> list[tuple[float, bytes]]:
    """V-JEPA 2 scene detection → representative keyframes.

    Extracts frames at ~1 fps, groups them into 16-frame clips, encodes
    with V-JEPA 2 (Ward-linkage clustering), and returns one keyframe per
    detected scene.  Falls back to uniform sampling on any error.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    duration = total / native_fps if total > 0 else 0.0
    step = max(1, int(native_fps))  # ~1 fps pass for scene detection

    frames_1fps: list = []
    ts_1fps: list[float] = []
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames_1fps.append(f)
            ts_1fps.append(idx / native_fps)
        idx += step
    cap.release()

    scene_ts: list[float] = []
    try:
        from rlm.video.scene_detection import detect_scenes

        indexer = _get_vjepa_indexer()
        indexer._ensure_scene_model()
        clips, clip_ts = indexer._group_frames_into_clips(frames_1fps, ts_1fps, 16)
        clip_embs = indexer._encode_clips_vjepa(clips)
        clip_reps = [c[len(c) // 2] for c in clips]

        def _embed_fn(_frames):
            return clip_embs  # pre-computed — ignore passed frame list

        scenes = detect_scenes(clip_reps, clip_ts, embed_fn=_embed_fn, threshold=0.3, min_duration=2.0)
        scene_ts = [(s + e) / 2 for s, e in scenes]
        _log.info("V-JEPA 2: %d scenes detected in %s", len(scenes), video_path)
    except Exception as exc:
        _log.warning("V-JEPA scene detection failed (%s) — uniform fallback", exc)

    if not scene_ts:
        n = min(max_frames, 8)
        scene_ts = [(i + 0.5) * duration / n for i in range(n)]

    if len(scene_ts) > max_frames:
        step_f = len(scene_ts) / max_frames
        scene_ts = [scene_ts[int(i * step_f)] for i in range(max_frames)]

    result: list[tuple[float, bytes]] = []
    cap = cv2.VideoCapture(video_path)
    for t in scene_ts:
        fidx = int(t * native_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        if resize:
            frame = cv2.resize(frame, resize)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        result.append((t, buf.tobytes()))
    cap.release()
    return result


def _direct_analyze(video_path: str, question: str, model: str, api_key: str) -> str:
    """Single VLM call using V-JEPA 2 scene-aware keyframe selection."""
    keyframes = _vjepa_keyframes(video_path)

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"{question}\n\n"
                "Below are keyframes selected by V-JEPA 2 scene detection — "
                "each represents a distinct scene in the video. "
                "Examine every frame carefully before answering. "
                "Cite specific moments as [TS: X.X] (seconds) "
                "placed right after the relevant statement."
            ),
        }
    ]
    for t, jpeg_bytes in keyframes:
        content.append({"type": "text", "text": f"[Frame at {t:.1f}s]"})
        b64 = base64.b64encode(jpeg_bytes).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"},
        })

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": content}]},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    question: str = Form(...),
    backend: str = Form(default="openrouter"),
    model: str = Form(default="openai/gpt-4o"),
    mode: str = Form(default="direct"),
):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    video_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{video_id}{suffix}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        api_key = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
        }.get(backend)

        if mode == "direct":
            response_text = _direct_analyze(
                str(video_path), question, model,
                api_key or os.getenv("OPENROUTER_API_KEY", ""),
            )
        else:
            from rlm.video.video_rlm import VideoRLM

            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            caption_fn = None
            if openrouter_key:
                from rlm.clients.openai import OpenAIClient

                _caption_lm = OpenAIClient(
                    model_name="openai/gpt-4o-mini",
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                )

                def caption_fn(frames, context=""):
                    parts = ["Describe this video segment in 1-2 sentences. Focus on: what is happening, who/what is present, and any key actions. Be concise."]
                    parts.extend(frames[:3])
                    return _caption_lm.completion(parts)

            video_rlm = VideoRLM(
                backend=backend,
                backend_kwargs={"model_name": model, "api_key": api_key},
                num_segments=4,
                max_frames_per_segment=3,
                resize=(480, 360),
                image_quality=65,
                auto_fps=True,
                max_iterations=15,
                token_budget=90_000,
                enable_search=False,
                caption_fn=caption_fn,
            )

            augmented = (
                f"{question}\n\n"
                "When answering, cite the exact moments in the video using this format: "
                "[TS: X.X] where X.X is the time in seconds (e.g. [TS: 12.5] or [TS: 90.0]). "
                "Place each timestamp right after the statement it supports. "
                "Include at least one timestamp per key claim."
            )
            result = video_rlm.completion(str(video_path), prompt=augmented)
            response_text = result.response

        timestamps = _parse_timestamps(response_text)
        answer_html = _render_answer_html(response_text)

        return JSONResponse({
            "answer": response_text,
            "answer_html": answer_html,
            "timestamps": timestamps,
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=True)
