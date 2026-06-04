from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


DATASET_NAME = "recmeapp/thumbs-up"
PARQUET_URL = (
    "https://huggingface.co/datasets/recmeapp/thumbs-up/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
N_APPS = 5
REVIEWS_PER_APP = 20


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def select_reviews() -> dict[str, list[dict]]:
    frame = pd.read_parquet(
        PARQUET_URL,
        columns=["app_name", "review", "date", "label", "Words Per Review"],
    )
    top = frame.nlargest(N_APPS * REVIEWS_PER_APP, "Words Per Review").to_dict("records")
    rows = []
    for row in top:
        review = clean(row.get("review"))
        rows.append(
            {
                "app_name": clean(row.get("app_name")),
                "review": review,
                "date": clean(row.get("date")),
                "label": row.get("label"),
                "rating": int(row.get("label", 0)) + 1,
                "word_count": int(row.get("Words Per Review") or len(review.split())),
            }
        )
    return {
        f"source_{index + 1:02d}": rows[index * REVIEWS_PER_APP : (index + 1) * REVIEWS_PER_APP]
        for index in range(N_APPS)
    }


def write_input(selected: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for app_index, (source_name, reviews) in enumerate(selected.items(), 1):
        lines = []
        for review_index, review in enumerate(reviews, 1):
            review_id = f"app{app_index:02d}-review{review_index:03d}"
            lines.extend(
                [
                    f"### REVIEW {review_id}",
                    f"app_name: {review['app_name']}",
                    f"rating: {review['rating'] or review['label']}",
                    f"date: {review['date'] or 'unknown'}",
                    f"text: {review['review']}",
                    "",
                ]
            )
            rows.append({"review_id": review_id, **review})
        (out_dir / f"{source_name}.txt").write_text("\n".join(lines), encoding="utf-8")

    with (out_dir / "source_reviews.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total_words = sum(int(row["word_count"]) for row in rows)
    estimated_tokens = int(total_words * 1.3)
    if len(rows) != N_APPS * REVIEWS_PER_APP:
        raise RuntimeError(f"Expected 100 reviews, got {len(rows)}")
    if estimated_tokens < 50_000:
        raise RuntimeError(f"Estimated token count is below 50k: {estimated_tokens}")
    print(f"Saved {len(rows)} reviews in {len(selected)} source files")
    print(f"Words: {total_words}, estimated tokens: {estimated_tokens}")


def main() -> None:
    write_input(select_reviews(), Path("input"))


if __name__ == "__main__":
    main()
