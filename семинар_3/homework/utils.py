"""Небольшие общие функции для домашнего пайплайна."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

from pydantic import TypeAdapter


STARTER_CLIENT = Path(__file__).parents[1] / "starter" / "llm_client.py"
spec = importlib.util.spec_from_file_location("starter_llm_client", STARTER_CLIENT)
if spec is None or spec.loader is None:
    raise ImportError(f"Не найден клиент: {STARTER_CLIENT}")

llm_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_client)

client = llm_client.make_client()
model = llm_client.get_model()
usage_log: list[dict] = []


def ask(response_model, system_prompt: str, user_text: str, label: str):
    """Один structured-output запрос с обязательными ретраями."""
    started = time.time()
    result, completion = client.chat.completions.create(
        model=model,
        response_model=response_model,
        max_retries=3,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        with_completion=True,
    )

    usage = completion.usage
    usage_log.append(
        {
            "label": label,
            "seconds": round(time.time() - started, 2),
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "cache_hit_tokens": int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0),
            "cache_miss_tokens": int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0),
        }
    )
    return result


def save_json(path: Path, data) -> None:
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_json(path: Path, model_type=None):
    data = json.loads(path.read_text(encoding="utf-8"))
    if model_type is None:
        return data
    return TypeAdapter(model_type).validate_python(data)
