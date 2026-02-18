from __future__ import annotations

import base64
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


def _direct_analyze(video_path: str, question: str, model: str, api_key: str, preset: dict) -> str:
    n_segs = preset["num_segments"]
    n_per_seg = preset["max_frames_per_segment"]
    resize = preset["resize"]
    quality = preset["quality"]

    cap = cv2.VideoCapture(video_path)
    total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    duration = total_frames_cv / native_fps if total_frames_cv > 0 else 0.0

    # Sample evenly: divide into n_segs segments, take n_per_seg evenly-spaced
    # frames inside each segment (centered in each sub-interval so we never
    # sample only the very first frame of a segment).
    timestamps: list[float] = []
    for seg_i in range(n_segs):
        seg_start = seg_i * duration / n_segs
        seg_end = (seg_i + 1) * duration / n_segs
        sub_dur = (seg_end - seg_start) / n_per_seg
        for j in range(n_per_seg):
            t = seg_start + (j + 0.5) * sub_dur   # centre of sub-interval
            timestamps.append(min(t, duration - 0.05))

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"{question}\n\n"
                "Below are frames sampled uniformly across the entire video. "
                "Examine every frame carefully before answering. "
                "When referencing a specific moment, cite it as [TS: X.X] "
                "(seconds) placed right after the relevant statement."
            ),
        }
    ]

    for t in timestamps:
        frame_idx = int(t * native_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if resize:
            frame = cv2.resize(frame, resize)
        content.append({"type": "text", "text": f"[Frame at {t:.1f}s]"})
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64 = base64.b64encode(buf.tobytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"},
        })

    cap.release()

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
    full_index: bool = Form(default=False),
    detail: str = Form(default="low"),
    mode: str = Form(default="direct"),
):
    DETAIL_PRESETS = {
        "low":    dict(num_segments=3, max_frames_per_segment=2, resize=(320, 240), quality=55,  max_iterations=10, token_budget=60_000),
        "medium": dict(num_segments=4, max_frames_per_segment=3, resize=(480, 360), quality=65,  max_iterations=15, token_budget=90_000),
        "high":   dict(num_segments=5, max_frames_per_segment=4, resize=(640, 480), quality=75,  max_iterations=20, token_budget=None),
    }
    preset = DETAIL_PRESETS.get(detail, DETAIL_PRESETS["low"])
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    video_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{video_id}{suffix}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        backend_map = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "openrouter": os.getenv("OPENROUTER_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
        }
        api_key = backend_map.get(backend)

        if mode == "direct":
            response_text = _direct_analyze(
                str(video_path), question, model,
                api_key or os.getenv("OPENROUTER_API_KEY", ""),
                preset,
            )
        else:
            from rlm.video.video_rlm import VideoRLM

            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            caption_fn = None
            if full_index and openrouter_key:
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
                num_segments=preset["num_segments"],
                max_frames_per_segment=preset["max_frames_per_segment"],
                resize=preset["resize"],
                image_quality=preset["quality"],
                auto_fps=True,
                max_iterations=preset["max_iterations"],
                token_budget=preset["token_budget"],
                enable_search=full_index,
                embedding_model=VISUAL_EMBED_MODEL if full_index else None,
                scene_model=SCENE_MODEL if full_index else None,
                text_embedding_model=TEXT_EMBED_MODEL if full_index else None,
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
