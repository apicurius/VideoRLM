"""ASR transcript extraction: Qwen3-ASR, faster-whisper, and audio utilities."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Batch size for ASR inference.  On CPU/MPS the autoregressive decoder is
# memory-bound; small batches avoid excessive padding.  On CUDA, larger
# batches saturate the GPU.
ASR_BATCH_CPU = 4
ASR_BATCH_CUDA = 16
# Duration (seconds) of each audio chunk we feed to qwen_asr.  Shorter
# chunks decode faster (less autoregressive steps) and produce less padding
# waste when batched.  30s is a sweet spot — short enough to keep decoding
# fast, long enough to avoid excessive chunk overhead.
# IMPORTANT: Must stay < 180 to avoid double-offset issues with
# qwen_asr's internal MAX_FORCE_ALIGN_INPUT_SECONDS limit.
ASR_CHUNK_SEC = 30
ASR_OVERLAP_SEC = 1  # overlap between chunks to avoid boundary word loss


def is_faster_whisper_model(model_name: str) -> bool:
    """Check if model_name refers to a faster-whisper model."""
    if model_name.startswith("faster-whisper/"):
        return True
    # Also accept bare whisper size names
    return model_name in (
        "tiny",
        "base",
        "small",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
        "turbo",
        "large",
    )


def load_transcript_file(path: str) -> list[dict]:
    """Load a transcript from a JSON file.

    Validates that each entry has the required keys (start_time, end_time,
    text).  Malformed entries are skipped with a warning.
    """
    try:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, list):
            logger.warning("Transcript file %s is not a JSON list; ignoring.", path)
            return []
        required = ("start_time", "end_time", "text")
        valid = [e for e in data if isinstance(e, dict) and all(k in e for k in required)]
        if len(valid) < len(data):
            logger.warning(
                "Skipped %d invalid transcript entries in %s",
                len(data) - len(valid),
                path,
            )
        return valid
    except Exception:
        logger.warning("Failed to load transcript from %s", path, exc_info=True)
    return []


def extract_audio(video_path: str, out_wav: str) -> bool:
    """Extract audio track to a WAV file using ffmpeg."""
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                out_wav,
            ],
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("Audio extraction failed for %s (ffmpeg may be missing).", video_path)
        return False


def split_audio_chunks(
    wav_path: str,
    chunk_sec: int,
    tmp_dir: str,
    overlap_sec: float = 0.0,
) -> list[tuple[str, float]]:
    """Split a WAV file into fixed-duration chunks using ffmpeg.

    Returns list of (chunk_path, offset_seconds) tuples.  When
    *overlap_sec* > 0 each chunk (except the first) starts that many
    seconds before the nominal boundary so words at the cut point are
    captured by both the previous and the current chunk.
    """
    assert chunk_sec < 180, (
        f"chunk_sec={chunk_sec} must be < 180 (qwen_asr MAX_FORCE_ALIGN_INPUT_SECONDS limit)"
    )
    import wave

    with wave.open(wav_path, "rb") as wf:
        duration = wf.getnframes() / wf.getframerate()

    if duration <= chunk_sec:
        return [(wav_path, 0.0)]

    stride = chunk_sec - overlap_sec
    chunks: list[tuple[str, float]] = []
    offset = 0.0
    idx = 0
    while offset < duration:
        chunk_dur = chunk_sec if idx == 0 else chunk_sec + overlap_sec
        chunk_path = str(Path(tmp_dir) / f"chunk_{idx:04d}.wav")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    str(offset),
                    "-t",
                    str(chunk_dur),
                    "-i",
                    wav_path,
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    chunk_path,
                ],
                check=True,
                capture_output=True,
            )
            chunks.append((chunk_path, offset))
        except subprocess.CalledProcessError:
            logger.warning("Failed to split audio chunk at offset %.1f", offset)
            break
        offset += stride if idx == 0 else stride
        idx += 1

    return chunks if chunks else [(wav_path, 0.0)]


def collect_transcript_segments(
    asr_result,
    offset: float,
    transcript: list[dict],
    *,
    skip_before: float = 0.0,
) -> None:
    """Extract sentence-level segments from one ASR chunk result.

    Appends to *transcript* in-place, adding *offset* to all timestamps.
    When *skip_before* > 0, words whose offset-corrected start_time falls
    before that threshold are dropped — they belong to the overlap region
    and are better covered by the previous chunk.
    """
    if asr_result.time_stamps is not None and asr_result.time_stamps.items:
        items = asr_result.time_stamps.items
        seg_words: list[dict] = []
        seg_start: float | None = None
        seg_end: float = 0.0

        def _flush_segment() -> None:
            nonlocal seg_words, seg_start, seg_end
            if not seg_words:
                return
            text = " ".join(w["text"] for w in seg_words).strip()
            if text:
                transcript.append(
                    {
                        "start_time": round(seg_start + offset, 3),
                        "end_time": round(seg_end + offset, 3),
                        "text": text,
                        "words": seg_words,
                    }
                )
            seg_words = []
            seg_start = None

        for idx, item in enumerate(items):
            # Skip words in the overlap region covered by previous chunk
            if skip_before > 0 and (item.start_time + offset) < skip_before:
                continue

            if seg_start is not None and (item.start_time - seg_end) > 1.0:
                _flush_segment()

            if seg_start is None:
                seg_start = item.start_time
            seg_words.append(
                {
                    "text": item.text,
                    "start_time": round(item.start_time + offset, 3),
                    "end_time": round(item.end_time + offset, 3),
                }
            )
            seg_end = item.end_time

            is_sentence_end = item.text.rstrip().endswith((".", "!", "?"))
            at_end = idx == len(items) - 1
            if is_sentence_end or at_end:
                _flush_segment()

    elif asr_result.text.strip():
        transcript.append(
            {
                "start_time": round(offset, 3),
                "end_time": round(offset, 3),
                "text": asr_result.text.strip(),
                "words": [],
            }
        )


def ensure_asr_model(
    current_model,
    current_model_name: str | None,
    model_name: str,
    device_pref: str = "auto",
) -> tuple[Any, str, int]:
    """Lazily load and cache the Qwen3-ASR model.

    Returns (model, model_name, batch_size) tuple.
    If model is already loaded for the same name, returns it as-is.
    """
    if current_model is not None and current_model_name == model_name:
        batch_size = (
            ASR_BATCH_CUDA if "cuda" in str(getattr(current_model, "device", "")) else ASR_BATCH_CPU
        )
        return current_model, model_name, batch_size

    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError:
        logger.info("qwen_asr not installed; skipping ASR.")
        return None, None, ASR_BATCH_CPU

    import torch

    device = device_pref
    if device == "auto":
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    batch_size = ASR_BATCH_CUDA if device == "cuda" else ASR_BATCH_CPU
    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    logger.info("[pipeline] Qwen3-ASR: loading model on %s (batch=%d)", device, batch_size)

    try:
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=batch_size,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=dtype,
                device_map=device,
            ),
        )
    except (OSError, RuntimeError, ImportError, ValueError):
        logger.info(
            "Qwen3-ForcedAligner-0.6B unavailable; proceeding without word-level alignment."
        )
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=batch_size,
        )
    logger.info("[pipeline] Qwen3-ASR: model loaded")
    return model, model_name, batch_size


def run_faster_whisper(
    video_path: str,
    model_name: str,
    *,
    _cached_model=None,
    _cached_model_size: str | None = None,
) -> tuple[list[dict], Any, str | None]:
    """Run faster-whisper ASR on a video file.

    Returns (transcript, model_instance, model_size) so caller can cache the model.
    """
    model_size = model_name.removeprefix("faster-whisper/")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        logger.info("faster-whisper not installed — skipping ASR")
        return [], None, None

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        if not extract_audio(video_path, wav_path):
            return [], _cached_model, _cached_model_size

        # Device selection
        import torch

        if torch.cuda.is_available():
            device, compute_type = "cuda", "float16"
        else:
            device, compute_type = "cpu", "int8"

        # Cache model instance to avoid reloading on repeated calls
        if _cached_model is not None and _cached_model_size == model_size:
            model = _cached_model
            logger.info("[pipeline] faster-whisper: reusing cached %s model", model_size)
        else:
            logger.info("[pipeline] faster-whisper: loading model %s", model_size)
            model = WhisperModel(model_size, device=device, compute_type=compute_type)

        logger.info("[pipeline] faster-whisper: transcribing audio")
        segments_gen, _info = model.transcribe(
            wav_path,
            word_timestamps=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Convert to KUAVi transcript format
        transcript: list[dict] = []
        for segment in segments_gen:
            entry: dict[str, Any] = {
                "start_time": round(segment.start, 3),
                "end_time": round(segment.end, 3),
                "text": segment.text.strip(),
            }
            if segment.words:
                entry["words"] = [
                    {
                        "text": w.word.strip(),
                        "start_time": round(w.start, 3),
                        "end_time": round(w.end, 3),
                    }
                    for w in segment.words
                ]
            transcript.append(entry)

        logger.info("[pipeline] faster-whisper: %d transcript segments", len(transcript))
        return transcript, model, model_size


def run_asr(
    video_path: str,
    model_name: str,
    *,
    asr_model=None,
    asr_batch_size: int | None = None,
    _faster_whisper_model=None,
    _faster_whisper_model_size: str | None = None,
) -> tuple[list[dict], Any, str | None]:
    """Transcribe audio using Qwen3-ASR with word-level timestamps.

    Splits audio into short chunks (default 30s) before transcription.

    Returns (transcript, faster_whisper_model, faster_whisper_model_size).
    The last two are only set when using faster-whisper, for caller caching.
    """
    # Route to faster-whisper if appropriate
    if is_faster_whisper_model(model_name):
        return run_faster_whisper(
            video_path,
            model_name,
            _cached_model=_faster_whisper_model,
            _cached_model_size=_faster_whisper_model_size,
        )

    if asr_model is None:
        return [], None, None

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = str(Path(tmp) / "audio.wav")
        if not extract_audio(video_path, wav_path):
            return [], None, None

        # Split into short chunks for faster decoding (with overlap for dedup)
        overlap = ASR_OVERLAP_SEC
        chunks = split_audio_chunks(
            wav_path,
            ASR_CHUNK_SEC,
            tmp,
            overlap_sec=overlap,
        )
        chunk_paths = [c[0] for c in chunks]
        chunk_offsets = [c[1] for c in chunks]
        logger.info(
            "[pipeline] Qwen3-ASR: transcribing %d chunk(s) of %ds (overlap=%ds)",
            len(chunks),
            ASR_CHUNK_SEC,
            overlap,
        )

        try:
            # Transcribe in batches so we can log progress on long videos.
            bs = asr_batch_size or len(chunk_paths)
            all_results: list = []
            for i in range(0, len(chunk_paths), bs):
                batch = chunk_paths[i : i + bs]
                r = asr_model.transcribe(
                    audio=batch,
                    return_time_stamps=True,
                )
                all_results.extend(r)
                logger.info(
                    "[pipeline] Qwen3-ASR: %d/%d chunks done",
                    min(i + bs, len(chunk_paths)),
                    len(chunk_paths),
                )

            if not all_results:
                return [], None, None

            # Merge results from all chunks, offsetting timestamps.
            # For chunks after the first, skip words in the overlap
            # region that are better covered by the previous chunk.
            transcript: list[dict] = []
            for chunk_idx, (asr_result, offset) in enumerate(
                zip(all_results, chunk_offsets, strict=False)
            ):
                skip_before = offset + overlap / 2 if chunk_idx > 0 and overlap > 0 else 0.0
                collect_transcript_segments(
                    asr_result,
                    offset,
                    transcript,
                    skip_before=skip_before,
                )

            logger.info("[pipeline] Qwen3-ASR: %d segments transcribed", len(transcript))
            return transcript, None, None
        except Exception:
            logger.warning("Qwen3-ASR transcription failed.", exc_info=True)
            return [], None, None


def get_transcript(
    video_path: str,
    *,
    asr_model_name: str = "Qwen/Qwen3-ASR-0.6B",
    transcript_path: str | None = None,
    asr_model=None,
    asr_batch_size: int | None = None,
    _faster_whisper_model=None,
    _faster_whisper_model_size: str | None = None,
) -> tuple[list[dict], Any, str | None]:
    """Return ASR transcript as a list of ``{start_time, end_time, text}`` dicts.

    Returns (transcript, faster_whisper_model, faster_whisper_model_size).
    """
    if transcript_path is not None:
        return load_transcript_file(transcript_path), None, None

    return run_asr(
        video_path,
        asr_model_name,
        asr_model=asr_model,
        asr_batch_size=asr_batch_size,
        _faster_whisper_model=_faster_whisper_model,
        _faster_whisper_model_size=_faster_whisper_model_size,
    )
