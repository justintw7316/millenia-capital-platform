from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


@dataclass
class AppConfig:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:14b")
    vlm_model: str = os.getenv("VLM_MODEL", "llama3.2-vision:latest")
    runs_dir: Path = Path(os.getenv("RUNS_DIR", "runs"))


def get_config() -> AppConfig:
    return AppConfig()
