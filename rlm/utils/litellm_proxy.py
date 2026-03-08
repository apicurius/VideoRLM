"""LiteLLM proxy launcher and URL resolution.

When ``LITELLM_PROXY_URL`` is set, *all* LLM backends are routed through a single
LiteLLM proxy instance using the :class:`~rlm.clients.litellm.LiteLLMClient`.

Use :func:`get_litellm_proxy_url` to check if the env var is set and
:func:`start_litellm_proxy` to launch a local proxy process.

Example:
    export LITELLM_PROXY_URL=http://localhost:4000
    # Now all get_client() calls return a LiteLLMClient pointed at that URL.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)


def get_litellm_proxy_url() -> str | None:
    """Return the ``LITELLM_PROXY_URL`` env var or ``None`` if unset."""
    return os.environ.get("LITELLM_PROXY_URL") or None


def start_litellm_proxy(
    config_path: str | None = None,
    host: str = "0.0.0.0",
    port: int = 4000,
    wait: bool = True,
    timeout: float = 30.0,
) -> subprocess.Popen:
    """Start a local LiteLLM proxy server as a background process.

    Args:
        config_path: Optional path to a ``litellm_config.yaml``.
        host: Bind address (default ``0.0.0.0``).
        port: Listen port (default ``4000``).
        wait: If ``True``, block until the ``/health`` endpoint responds.
        timeout: Max seconds to wait for health (default 30).

    Returns:
        The :class:`subprocess.Popen` handle of the proxy process.

    Raises:
        FileNotFoundError: If ``litellm`` CLI is not on PATH.
        TimeoutError: If proxy does not become healthy within *timeout*.
    """
    if shutil.which("litellm") is None:
        raise FileNotFoundError(
            "litellm CLI not found. Install with: pip install 'litellm[proxy]'"
        )

    cmd = ["litellm", "--host", host, "--port", str(port)]
    if config_path:
        cmd.extend(["--config", config_path])

    logger.info("Starting LiteLLM proxy: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    url = f"http://{host}:{port}"
    os.environ["LITELLM_PROXY_URL"] = url

    if wait:
        import requests

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{url}/health", timeout=2)
                if r.status_code == 200:
                    logger.info("LiteLLM proxy healthy at %s", url)
                    return proc
            except Exception:
                pass
            time.sleep(0.5)
        proc.terminate()
        raise TimeoutError(f"LiteLLM proxy did not become healthy within {timeout}s")

    return proc
