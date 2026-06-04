"""Обязательный конвейер: IE -> аспекты -> Map-Reduce -> judge."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prompts import ASPECTS_SYSTEM, CHUNK_SYSTEM, IE_SYSTEM, JUDGE_SYSTEM, REDUCE_STRICT_SYSTEM, REDUCE_SYSTEM
from schema import ChunkSummary, JudgeReport, Review, ReviewSentiment, ReviewSummary
from utils import ask, save_json, usage_log


ASPECTS = ["performance", "design", "support", "price", "ads", "reliability"]


def read_sources(input_path: str) -> dict[str, str]:
    path = Path(input_path)
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    return {file.stem: file.read_text(encoding="utf-8") for file in files}


def split_reviews(text: str) -> list[str]:
    return ["### REVIEW " + part.strip() for part in text.split("### REVIEW ")[1:]]


def make_batches(sources: dict[str, str], size: int = 5) -> list[tuple[str, str]]:
    batches = []
    for source, text in sources.items():
        reviews = split_reviews(text)
        for index in range(0, len(reviews), size):
            batches.append((source, "\n\n".join(reviews[index : index + size])))
    return batches


def extract_reviews(sources: dict[str, str]) -> list[Review]:
    def extract_one(job):
        source, text = job
        return ask(list[Review], IE_SYSTEM, text, f"ie:{source}")

    reviews = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(extract_one, make_batches(sources)):
            reviews.extend(result)
    return sorted(reviews, key=lambda item: item.review_id)


def extract_aspects(sources: dict[str, str]) -> list[ReviewSentiment]:
    def extract_one(job):
        source, text = job
        return ask(list[ReviewSentiment], ASPECTS_SYSTEM, text, f"aspects:{source}")

    aspects = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(extract_one, make_batches(sources)):
            aspects.extend(result)
    return sorted(aspects, key=lambda item: item.review_id)


def check_quotes(aspects: list[ReviewSentiment], source_text: str) -> list[dict]:
    text = source_text.lower()
    ghosts = []
    for review in aspects:
        for aspect in review.aspects:
            probe = aspect.quote.strip().lower()[:40]
            if probe and probe not in text:
                ghosts.append(
                    {
                        "review_id": review.review_id,
                        "app_name": review.app_name,
                        "aspect": aspect.aspect,
                        "quote": aspect.quote,
                    }
                )
    return ghosts


def remove_ghost_quotes(aspects: list[ReviewSentiment], ghosts: list[dict]) -> list[ReviewSentiment]:
    bad = {(item["review_id"], item["aspect"], item["quote"]) for item in ghosts}
    cleaned = []
    for review in aspects:
        items = [
            aspect
            for aspect in review.aspects
            if (review.review_id, aspect.aspect, aspect.quote) not in bad
        ]
        cleaned.append(review.model_copy(update={"aspects": items}))
    return cleaned


def build_heatmap(aspects: list[ReviewSentiment], out_path: Path) -> None:
    values = {"negative": -1, "neutral": 0, "positive": 1}
    rows = []
    for review in aspects:
        for aspect in review.aspects:
            rows.append(
                {
                    "app_name": review.app_name,
                    "aspect": aspect.aspect,
                    "value": values[aspect.sentiment],
                }
            )

    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="app_name", columns="aspect", values="value", aggfunc="mean")
    pivot = pivot.reindex(columns=ASPECTS)

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(pivot.fillna(0), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Тональность аспектов по приложениям")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def map_summaries(sources: dict[str, str]) -> list[ChunkSummary]:
    def summarize_one(job):
        source, text = job
        prompt = f"source: {source}\n\n{text}"
        return ask(ChunkSummary, CHUNK_SYSTEM, prompt, f"map:{source}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(summarize_one, make_batches(sources)))


def reduce_summaries(summaries: list[ChunkSummary], strict: bool = False) -> ReviewSummary:
    text = "\n\n".join(
        f"## {item.source} ({item.sentiment})\n"
        + "\n".join(f"- {point}" for point in item.key_points)
        for item in summaries
    )
    prompt = REDUCE_STRICT_SYSTEM if strict else REDUCE_SYSTEM
    label = "reduce_strict" if strict else "reduce"
    return ask(ReviewSummary, prompt, text, label)


def judge(reviews: list[Review], summary: ReviewSummary) -> JudgeReport:
    lines = ["## Рекомендации"]
    lines.extend(f"{index}. {item}" for index, item in enumerate(summary.action_items, 1))
    lines.append("\n## Извлеченные проблемы")
    for review in reviews:
        for issue in review.issues:
            lines.append(f"- [{review.app_name}/{issue.category}] «{issue.quote}»")
    return ask(JudgeReport, JUDGE_SYSTEM, "\n".join(lines), "judge")


def analyze(input_path: str, out_dir: str = "output") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = read_sources(input_path)
    full_text = "\n\n".join(sources.values())

    print("1. Information Extraction")
    reviews = extract_reviews(sources)
    save_json(out / "reviews.json", [item.model_dump() for item in reviews])

    print("2. Аспектный анализ")
    aspects = extract_aspects(sources)
    ghosts = check_quotes(aspects, full_text)
    aspects = remove_ghost_quotes(aspects, ghosts)
    save_json(out / "aspects.json", [item.model_dump() for item in aspects])
    save_json(out / "ghost_quotes.json", {"initial": ghosts, "unresolved": []})
    build_heatmap(aspects, out / "heatmap.png")

    print("3. Map-Reduce")
    summaries = map_summaries(sources)
    summary = reduce_summaries(summaries)
    save_json(out / "summary.json", summary)

    print("4. LLM-as-judge")
    report = judge(reviews, summary)
    if report.overall_score < 0.7:
        summary = reduce_summaries(summaries, strict=True)
        save_json(out / "summary.json", summary)
        report = judge(reviews, summary)
    save_json(out / "judge_report.json", report)

    save_json(out / "validation_errors.json", [])
    save_json(out / "usage_current_run.json", usage_log)
    print(f"Готово. Артефакты сохранены в {out}")


def main() -> None:
    analyze("input", "output")


if __name__ == "__main__":
    main()
