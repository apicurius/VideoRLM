import torch
import numpy as np
from pathlib import Path
from .base_embedder import BaseEmbedder
import torch.nn as nn
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings

# Monkeypatch CLIPVisionEmbeddings that assumes square inputs (strict check in transformers 4.40+)
def patched_clip_vision_embeddings_forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
    batch_size, _, height, width = pixel_values.shape
    
    # LanguageBind audio uses non-square [112, 1036], patch to support list/tuple image_size
    if isinstance(self.image_size, (list, tuple)):
        if not interpolate_pos_encoding and (height != self.image_size[0] or width != self.image_size[1]):
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
    elif not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
        raise ValueError(
            f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
        )
        
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    if interpolate_pos_encoding:
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
    else:
        embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings

# Global monkeypatch
CLIPVisionEmbeddings.forward = patched_clip_vision_embeddings_forward


class LanguageBindEmbedder(BaseEmbedder):
    """
    Local multimodal embedder using LanguageBind.
    Embeds video, audio, and text in one shared space.
    Runs entirely on local GPU — zero API cost.

    Models used:
      Video: LanguageBind/LanguageBind_Video_FT
      Audio: LanguageBind/LanguageBind_Audio_FT
      Text:  shared text encoder across all modalities
    """

    VIDEO_MODEL = "LanguageBind/LanguageBind_Video_FT"
    AUDIO_MODEL = "LanguageBind/LanguageBind_Audio_FT"
    TEXT_MAX_LEN = 64

    def __init__(self, device: str | None = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._video_model = None
        self._video_model_cpu = None
        self._audio_model = None
        self._tokenizer = None

    @staticmethod
    def _is_cuda_device_assert(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "cuda error" in message
            or "device-side assert" in message
            or "torch.acceleratorerror" in message
        )

    def _load_video_for_device(self, target_device: str):
        from languagebind import LanguageBind

        if target_device == "cpu":
            if self._video_model_cpu is None:
                self._video_model_cpu = LanguageBind(
                    clip_type={"video": self.VIDEO_MODEL.split('/')[-1]},
                    cache_dir=".languagebind_cache",
                )
                self._video_model_cpu = self._video_model_cpu.to("cpu")
                self._video_model_cpu.eval()
            return self._video_model_cpu

        if self._video_model is None:
            # The vendored LanguageBind expects a dict for clip_type where key=modality, value is the model name suffix.
            self._video_model = LanguageBind(
                clip_type={"video": self.VIDEO_MODEL.split('/')[-1]},
                cache_dir=".languagebind_cache",
            )
            self._video_model = self._video_model.to(target_device)
            self._video_model.eval()
        return self._video_model

    def _load_video(self):
        return self._load_video_for_device(self.device)

    def _load_audio(self):
        if self._audio_model is None:
            from languagebind import LanguageBind
            self._audio_model = LanguageBind(
                clip_type={"audio": self.AUDIO_MODEL.split('/')[-1]},
                cache_dir=".languagebind_cache",
            )
            self._audio_model = self._audio_model.to(self.device)
            self._audio_model.eval()
        return self._audio_model

    def embed_text(self, text: str) -> list[float]:
        from languagebind import LanguageBindProcessor
        # LanguageBind text encoder is unstable on some CUDA setups (device-side
        # assert in CLIP embedding lookup). Run text/query on CPU for stability.
        run_device = "cpu"
        model = self._load_video_for_device(run_device)
        try:
            processor = LanguageBindProcessor.from_pretrained(
                self.VIDEO_MODEL,
                cache_dir=".languagebind_cache",
                local_files_only=True,
            )
        except Exception:
            processor = LanguageBindProcessor.from_pretrained(
                self.VIDEO_MODEL,
                cache_dir=".languagebind_cache",
            )
        raw_inputs = processor(
            text=[text],
            return_tensors="pt",
        )
        inputs = {k: v.to(run_device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}

        language_encoder = model.modality_encoder["language"]
        max_positions = getattr(getattr(language_encoder, "config", None), "max_position_embeddings", None)
        if max_positions is None and hasattr(language_encoder, "embeddings"):
            position_embedding = getattr(language_encoder.embeddings, "position_embedding", None)
            if position_embedding is not None:
                max_positions = position_embedding.num_embeddings
        if "input_ids" in inputs:
            if max_positions is None:
                max_positions = self.TEXT_MAX_LEN
            else:
                max_positions = min(int(max_positions), self.TEXT_MAX_LEN)
            seq_len = inputs["input_ids"].shape[-1]
            if seq_len > max_positions:
                inputs["input_ids"] = inputs["input_ids"][..., :max_positions]
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = inputs["attention_mask"][..., :max_positions]
                if "position_ids" in inputs:
                    inputs["position_ids"] = inputs["position_ids"][..., :max_positions]

            batch_size, seq_len = inputs["input_ids"].shape
            inputs["position_ids"] = (
                torch.arange(seq_len, device=run_device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch_size, seq_len)
            )
        try:
            with torch.no_grad():
                # In the vendored LanguageBind, we use the forward method which handles the encoders
                # For query (text), we need to pass a dict {'language': inputs}
                out = model({"language": inputs})
                emb = out["language"]
        except Exception as error:
            if run_device != "cuda" or not self._is_cuda_device_assert(error):
                raise

            # CUDA context is poisoned after device-side assert. Never move the
            # existing CUDA model; reinitialize a clean CPU model and retry.
            self.device = "cpu"
            model = self._load_video_for_device("cpu")
            inputs = {k: v for k, v in raw_inputs.items()}
            with torch.no_grad():
                out = model({"language": inputs})
                emb = out["language"]
        return emb[0].cpu().numpy().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_video_segment(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> list[float]:
        import tempfile, subprocess
        from languagebind import LanguageBindProcessor
        model = self._load_video()
        try:
            processor = LanguageBindProcessor.from_pretrained(
                self.VIDEO_MODEL,
                cache_dir=".languagebind_cache",
                local_files_only=True,
            )
        except Exception:
            processor = LanguageBindProcessor.from_pretrained(
                self.VIDEO_MODEL,
                cache_dir=".languagebind_cache",
            )
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as tmp:
            clip_path = tmp.name
        try:
            duration = min(end_sec - start_sec, 60.0)
            # Move -ss after -i for better compatibility with some files, 
            # and remove -c copy to allow re-encoding if needed at segment boundaries.
            subprocess.run([
                "ffmpeg", "-y",
                "-i", video_path,
                "-ss", str(start_sec),
                "-t", str(duration),
                "-f", "mp4",
                "-an", # No audio for video model
                clip_path,
            ], check=True, capture_output=True)
            inputs = processor(
                images=[clip_path],
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            with torch.no_grad():
                # Pass a dict with the modality key
                out = model({'video': inputs})
                emb = out['video']
            return emb[0].cpu().numpy().tolist()
        finally:
            Path(clip_path).unlink(missing_ok=True)

    def embed_audio_segment(
        self,
        audio_path: str,
    ) -> list[float]:
        from languagebind import LanguageBindProcessor
        model = self._load_audio()
        try:
            processor = LanguageBindProcessor.from_pretrained(
                self.AUDIO_MODEL,
                cache_dir=".languagebind_cache",
                local_files_only=True,
            )
        except Exception:
            processor = LanguageBindProcessor.from_pretrained(
                self.AUDIO_MODEL,
                cache_dir=".languagebind_cache",
            )
        inputs = processor(
            images=[audio_path],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            out = model({'audio': inputs})
            emb = out['audio']
        return emb[0].cpu().numpy().tolist()
