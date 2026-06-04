"""Дополнительные критерии: autodiscovery, multi-doc, caching и hierarchical MR."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import map_summaries, read_sources, reduce_summaries
from prompts import (
    ASPECTS_SYSTEM,
    DISCOVER_SYSTEM,
    GROUP_REDUCE_SYSTEM,
    MULTI_DOC_SYSTEM,
    REDUCE_SYSTEM,
)
from schema import (
    DiscoveredAspects,
    DynamicReviewSentiment,
    GroupSummary,
    MultiDocSummary,
    ReviewSentiment,
    ReviewSummary,
)
from utils import ask, client, load_json, model, save_json, usage_log


def autodiscovery(sources: dict[str, str], fixed: list[ReviewSentiment], out: Path) -> None:
    sample = "\n\n".join(list(sources.values())[:2])
    discovered = ask(DiscoveredAspects, DISCOVER_SYSTEM, sample, "discover_aspects")
    aspect_list = "\n".join(f"- {item.name}: {item.description}" for item in discovered.aspects)

    dynamic = []
    for source, text in sources.items():
        prompt = ASPECTS_SYSTEM + "\nИспользуй эти темы:\n" + aspect_list
        dynamic.extend(ask(list[DynamicReviewSentiment], prompt, text, f"dynamic:{source}"))

    fixed_names = {aspect.aspect for review in fixed for aspect in review.aspects}
    dynamic_names = {aspect.aspect for review in dynamic for aspect in review.aspects}
    comparison = {
        "fixed": sorted(fixed_names),
        "dynamic": sorted(dynamic_names),
        "new_dynamic_topics": sorted(dynamic_names - fixed_names),
        "missing_fixed_topics": sorted(fixed_names - dynamic_names),
    }
    save_json(out / "aspects_discovered.json", [item.model_dump() for item in dynamic])
    save_json(out / "aspect_comparison.json", comparison)


def multi_doc(
    sources: dict[str, str],
    aspects: list[ReviewSentiment],
    summaries,
    out: Path,
) -> None:
    rows = []
    for review in aspects:
        for aspect in review.aspects:
            rows.append({"app_name": review.app_name, "aspect": aspect.aspect})
    frame = pd.DataFrame(rows)
    pd.crosstab(frame["app_name"], frame["aspect"]).to_csv(
        out / "multi_doc_pivot.csv",
        encoding="utf-8-sig",
    )

    source_summaries = []
    for source in sources:
        items = [item for item in summaries if item.source == source]
        source_summaries.append(reduce_summaries(items))

    text = "\n\n".join(
        f"## {source}\n{summary.headline}\n"
        + "\n".join(f"- {finding}" for finding in summary.key_findings)
        for source, summary in zip(sources, source_summaries)
    )
    result = ask(MultiDocSummary, MULTI_DOC_SYSTEM, text, "multi_doc")
    save_json(out / "multi_doc_summary.json", result)


def cache_experiment(sample_text: str, out: Path) -> None:
    runs = []
    for index in range(2):
        result, completion = client.chat.completions.create(
            model=model,
            response_model=list[ReviewSentiment],
            max_retries=3,
            temperature=0.0,
            messages=[
                {"role": "system", "content": ASPECTS_SYSTEM},
                {"role": "user", "content": sample_text},
            ],
            with_completion=True,
        )
        usage = completion.usage
        runs.append(
            {
                "run": index + 1,
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "cache_hit_tokens": int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0),
                "cache_miss_tokens": int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0),
                "objects": len(result),
            }
        )

    prompt_tokens = runs[1]["prompt_tokens"]
    saving_rub = prompt_tokens / 1_000_000 * 21 * 0.9
    report = {
        "runs": runs,
        "provider_exposed_cache_hit": runs[1]["cache_hit_tokens"] > 0,
        "theoretical_assumption": "90% discount on repeated input tokens",
        "theoretical_saving_rub": round(saving_rub, 6),
        "theoretical_saving_usd": round(saving_rub / 90, 6),
    }
    save_json(out / "cache_report.json", report)


def hierarchical_summary(summaries, out: Path, group_size: int = 5) -> None:
    groups = []
    for index in range(0, len(summaries), group_size):
        items = summaries[index : index + group_size]
        text = "\n\n".join(
            f"## {item.source} ({item.sentiment})\n"
            + "\n".join(f"- {point}" for point in item.key_points)
            for item in items
        )
        groups.append(ask(GroupSummary, GROUP_REDUCE_SYSTEM, text, f"group:{index // group_size + 1}"))

    text = "\n\n".join(
        f"## group {index + 1} ({group.overall_sentiment})\n"
        + "\n".join(f"- {theme}" for theme in group.themes)
        for index, group in enumerate(groups)
    )
    result = ask(ReviewSummary, REDUCE_SYSTEM, text, "hierarchical_reduce")
    save_json(out / "summary_hierarchical.json", result)


def run_bonus(input_path: str = "input", out_dir: str = "output") -> None:
    out = Path(out_dir)
    sources = read_sources(input_path)
    aspects = load_json(out / "aspects.json", list[ReviewSentiment])
    summaries = map_summaries(sources)

    print("1. Autodiscovery")
    autodiscovery(sources, aspects, out)
    print("2. Multi-doc")
    multi_doc(sources, aspects, summaries, out)
    print("3. Caching")
    cache_experiment(next(iter(sources.values())), out)
    print("4. Hierarchical Map-Reduce")
    hierarchical_summary(summaries, out)
    save_json(out / "usage_bonus_run.json", usage_log)
    print("Бонусные артефакты сохранены.")


if __name__ == "__main__":
    run_bonus("input", "output")
