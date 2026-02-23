"""Gemini-based captioning factories for KUAVi video indexing.

Provides caption_fn, frame_caption_fn, and refine_fn factories that can be
used with VideoIndexer.index_video() from both the MCP server and experiment runner.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _frame_to_image_dict(frame: np.ndarray) -> dict[str, str]:
    """Convert a BGR numpy frame to a base64 image dict."""
    import cv2

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {
        "data": base64.b64encode(buf.tobytes()).decode(),
        "mime_type": "image/jpeg",
    }


def _build_gemini_parts(frames: list[Any]) -> list:
    """Convert a mixed list of strings and numpy frames to Gemini Part objects."""
    from google.genai import types

    parts: list = []
    for item in frames:
        if isinstance(item, str):
            parts.append(types.Part(text=item))
        elif isinstance(item, np.ndarray):
            img = _frame_to_image_dict(item)
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type=img["mime_type"],
                        data=base64.b64decode(img["data"]),
                    )
                )
            )
        elif isinstance(item, dict) and "data" in item:
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type=item.get("mime_type", "image/jpeg"),
                        data=base64.b64decode(item["data"]),
                    )
                )
            )
    return parts


def make_gemini_caption_fn(
    model: str = "gemini-2.5-flash",
    max_frames: int = 6,
    api_key: str | None = None,
) -> callable:
    """Create a segment-level caption function using Gemini.

    Returns a callable with signature ``(frames: list[np.ndarray | str]) -> dict``
    producing structured annotations with summary/action/actor fields.
    """
    from google import genai

    prompt_text = (
        "Analyze this video segment and produce a structured JSON annotation.\n\n"
        "Return ONLY valid JSON with this exact structure:\n"
        "{\n"
        '  "summary": {\n'
        '    "brief": "<single sentence, ~20 words, describing what happens>",\n'
        '    "detailed": "<comprehensive description, ~95 words>"\n'
        "  },\n"
        '  "action": {\n'
        '    "brief": "<imperative verb phrase, 2-5 words, e.g. stir sauce>",\n'
        '    "detailed": "<imperative sentence with details>",\n'
        '    "actor": "<noun phrase describing who performs the action>"\n'
        "  }\n"
        "}\n\n"
        "Rules:\n"
        "- summary.brief: One sentence, ~20 words. Focus on visual content.\n"
        "- summary.detailed: ~95 words. Describe events chronologically.\n"
        "- action.brief: Imperative verb phrase (2-5 words). Use 'N/A' if no action.\n"
        "- Do NOT hallucinate. Only describe what is visually observable.\n"
        "- Do NOT infer speech content, names, or internal states.\n"
        "- Only describe what is consistently visible across multiple frames.\n"
        "- Do not add claims not supported by at least 2 frame observations.\n"
        "- Verify action.brief is truly an imperative verb phrase.\n"
        "- If frames conflict, describe only what is consistently observed.\n"
    )

    def caption_fn(frames: list) -> dict:
        # Separate string context tokens from real frames
        real_frames = [f for f in frames if not isinstance(f, str)]
        str_tokens = [f for f in frames if isinstance(f, str)]

        # Limit frames sent to the model
        if len(real_frames) > max_frames:
            step = len(real_frames) / max_frames
            real_frames = [real_frames[int(i * step)] for i in range(max_frames)]

        # Build parts: text prompt + context tokens + frame images
        parts = [prompt_text]
        for tok in str_tokens:
            parts.append(tok)

        gemini_parts = _build_gemini_parts(parts + real_frames)

        try:
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
            response = client.models.generate_content(model=model, contents=gemini_parts)
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Gemini caption_fn failed: %s", e)
            return {
                "summary": {"brief": "", "detailed": ""},
                "action": {"brief": "N/A", "detailed": "", "actor": None},
            }

    return caption_fn


def make_gemini_frame_caption_fn(
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
) -> callable:
    """Create a single-keyframe caption function using Gemini.

    Returns a callable with signature ``(frames: list[np.ndarray]) -> str``.
    """
    from google import genai

    prompt_text = (
        "Describe this single video frame in 1-2 sentences. "
        "Focus on what is shown visually: objects, people, text, actions, "
        "scene setting. Be specific and concise. Do not hallucinate."
    )

    def frame_caption_fn(frames: list) -> str:
        if not frames:
            return ""
        frame = frames[0]
        gemini_parts = _build_gemini_parts([prompt_text, frame])
        try:
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
            response = client.models.generate_content(model=model, contents=gemini_parts)
            return response.text.strip()
        except Exception as e:
            logger.warning("Gemini frame_caption_fn failed: %s", e)
            return ""

    return frame_caption_fn


def make_gemini_refine_fn(
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
) -> callable:
    """Create a self-refine function using Gemini.

    Returns a callable with signature ``(draft: str, context: str, effort: str) -> str``
    that returns a JSON string with the refined annotation.
    """
    from google import genai

    def refine_fn(draft: str, context: str, effort: str = "high") -> str:
        prompt = (
            "You are refining a video segment annotation. "
            "Return ONLY valid JSON with the same structure as the input.\n\n"
            f"## Context\n{context}\n\n"
            f"## Task\n{draft}\n\n"
            "Return ONLY the refined JSON annotation. No markdown fences, no explanation."
        )
        try:
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
            response = client.models.generate_content(model=model, contents=prompt)
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            # Validate it's valid JSON before returning
            json.loads(text)
            return text
        except Exception as e:
            logger.warning("Gemini refine_fn failed: %s", e)
            return draft  # Return original draft on failure

    return refine_fn
