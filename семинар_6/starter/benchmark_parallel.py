"""
Замер ускорения параллельного исполнения уровней PWC.

Запуск:
    python benchmark_parallel.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator import run_pwc

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass


QUESTIONS = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
    },
    {
        "id": "parallel_bonus",
        "query": (
            "Сравни одновременно курс USD, EUR и CNY на 1 января 2022 "
            "и на 1 апреля 2026, затем скажи, какая валюта выросла сильнее."
        ),
    },
]


def _run(query: str, *, parallel: bool) -> dict:
    start = time.perf_counter()
    try:
        result = run_pwc(
            query,
            max_iter=3,
            verbose=False,
            use_validator=True,
            parallel=parallel,
        )
        status = "ok" if result.get("answer") else "error"
    except Exception as e:
        result = {"answer": None, "error": f"{type(e).__name__}: {e}"}
        status = "exception"
    return {
        "status": status,
        "elapsed_sec": round(time.perf_counter() - start, 3),
        "answer_preview": (result.get("answer") or result.get("error") or "")[:220],
    }


def run_benchmark(*, repeat: int = 1) -> list[dict]:
    rows = []
    for q in QUESTIONS:
        for i in range(repeat):
            seq = _run(q["query"], parallel=False)
            par = _run(q["query"], parallel=True)
            speedup = None
            if par["elapsed_sec"] > 0:
                speedup = round(seq["elapsed_sec"] / par["elapsed_sec"], 3)
            row = {
                "id": q["id"],
                "query": q["query"],
                "run": i + 1,
                "sequential": seq,
                "parallel": par,
                "speedup": speedup,
            }
            rows.append(row)
            print(
                f"{q['id']} run {i + 1}: seq={seq['elapsed_sec']}s "
                f"par={par['elapsed_sec']}s speedup={speedup}"
            )

    out = Path(__file__).parent / "parallel_benchmark.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")
    return rows


def main():
    run_benchmark(repeat=1)


if __name__ == "__main__":
    main()
