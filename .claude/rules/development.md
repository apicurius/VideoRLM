# Development

## Setup

```bash
# Install with all optional deps
uv sync
uv pip install torch transformers sentence-transformers pillow torchvision
```

## Commands

```bash
# Run tests
uv run python -m pytest tests/

# Start MCP server
uv run python -m kuavi.mcp_server

# CLI usage
uv run python -m kuavi.cli index <video>
uv run python -m kuavi.cli search <query> --index-dir <path>
uv run python -m kuavi.cli analyze <video> -q "question"
```

## Dependencies

- **Core**: opencv-python, numpy, scikit-learn, mcp
- **Embeddings** (optional): torch, transformers, sentence-transformers
- **ASR** (optional): qwen-asr
