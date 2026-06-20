from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

import requests

from .utils import encode_file_b64, extract_json_from_text


class OllamaError(RuntimeError):
    pass


def chat_text(
    prompt: str,
    model: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    timeout: int = 180,
) -> str:
    url = ollama_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        raise OllamaError(f"Ollama text request failed: {exc}") from exc
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def chat_json(
    prompt: str,
    model: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    timeout: int = 180,
) -> Any:
    text = chat_text(prompt, model=model, ollama_url=ollama_url, temperature=temperature, timeout=timeout)
    return extract_json_from_text(text)


def chat_vision(
    prompt: str,
    image_path: Path,
    model: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    timeout: int = 240,
) -> str:
    url = ollama_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encode_file_b64(image_path)],
            }
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        raise OllamaError(f"Ollama vision request failed: {exc}") from exc
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def chat_text_stream(
    prompt: str,
    model: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    timeout: int = 180,
) -> Generator[str, None, None]:
    url = ollama_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout, stream=True)
        response.raise_for_status()
    except Exception as exc:
        raise OllamaError(f"Ollama streaming text request failed: {exc}") from exc
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("message", {}).get("content", "")
        if token:
            yield token
        if data.get("done"):
            break


def chat_vision_stream(
    prompt: str,
    image_path: Path,
    model: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    timeout: int = 240,
) -> Generator[str, None, None]:
    url = ollama_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encode_file_b64(image_path)],
            }
        ],
        "stream": True,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout, stream=True)
        response.raise_for_status()
    except Exception as exc:
        raise OllamaError(f"Ollama streaming vision request failed: {exc}") from exc
    for line in response.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("message", {}).get("content", "")
        if token:
            yield token
        if data.get("done"):
            break


def chat_vision_json(
    prompt: str,
    image_path: Path,
    model: str,
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    timeout: int = 240,
) -> Any:
    text = chat_vision(prompt, image_path=image_path, model=model, ollama_url=ollama_url, temperature=temperature, timeout=timeout)
    return extract_json_from_text(text)
