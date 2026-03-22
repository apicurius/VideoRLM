import os
from pathlib import Path
from .base_embedder import BaseEmbedder

class GeminiEmbedder(BaseEmbedder):
    """
    Optional upgrade embedder using Gemini Embedding 2.
    Enable with: EMBEDDING_BACKEND=gemini in .env
    Requires: GEMINI_API_KEY from https://aistudio.google.com

    Higher quality than LanguageBind but requires API calls
    during indexing. Query-time search is still $0 (local vectors).
    """

    MODEL = "gemini-embedding-exp-03-07"

    def __init__(self):
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. "
                "Get one at https://aistudio.google.com "
                "or use EMBEDDING_BACKEND=languagebind (default)."
            )
        genai.configure(api_key=api_key)
        self._genai = genai

    def embed_text(self, text: str) -> list[float]:
        result = self._genai.embed_content(
            model=self.MODEL,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    def embed_query(self, text: str) -> list[float]:
        result = self._genai.embed_content(
            model=self.MODEL,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def embed_video_segment(
        self,
        video_path: str,
        start_sec: float,
        end_sec: float,
    ) -> list[float]:
        import tempfile, subprocess
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as tmp:
            clip_path = tmp.name
        try:
            duration = min(end_sec - start_sec, 120.0)
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy", clip_path,
            ], check=True, capture_output=True)
            uploaded = self._genai.upload_file(clip_path)
            result = self._genai.embed_content(
                model=self.MODEL,
                content=uploaded,
                task_type="retrieval_document",
            )
            self._genai.delete_file(uploaded.name)
            return result["embedding"]
        finally:
            Path(clip_path).unlink(missing_ok=True)

    def embed_audio_segment(
        self, audio_path: str,
    ) -> list[float]:
        uploaded = self._genai.upload_file(audio_path)
        result = self._genai.embed_content(
            model=self.MODEL,
            content=uploaded,
            task_type="retrieval_document",
        )
        self._genai.delete_file(uploaded.name)
        return result["embedding"]
