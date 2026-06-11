import json
from pathlib import Path

from pipeline import DATA_DIR, retrieve


BASE_DIR = Path(__file__).parent
STRATEGIES = ["fixed", "smart"]


def read_gold():
    path = BASE_DIR / "gold.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_corpus_info():
    docs = []
    for path in sorted(DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append(
            {
                "source": path.stem,
                "chars": len(text),
                "words": len(text.split()),
            }
        )

    return {
        "doc_count": len(docs),
        "total_chars": sum(doc["chars"] for doc in docs),
        "docs": docs,
    }


def run_eval(strategy):
    rows = []
    debug_rows = []

    for item in read_gold():
        hits = retrieve(item["question"], strategy=strategy, k=5)
        found_sources = [hit["source"] for hit in hits]
        found_gold = set(found_sources) & set(item["gold_sources"])
        hit_rate = len(found_gold) / len(item["gold_sources"])

        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "gold_sources": item["gold_sources"],
                "retrieved_sources": found_sources,
                "hit_rate_at_5": hit_rate,
            }
        )

        debug_rows.append(
            {
                "id": item["id"],
                "strategy": strategy,
                "hits": [
                    {
                        "chunk_id": hit["id"],
                        "source": hit["source"],
                        "score": hit["score"],
                        "preview": " ".join(hit["text"].split())[:500],
                    }
                    for hit in hits
                ],
            }
        )

    average = sum(row["hit_rate_at_5"] for row in rows) / len(rows)
    return {
        "strategy": strategy,
        "hit_rate_at_5": average,
        "results": rows,
        "debug": debug_rows,
    }


def run_all():
    runs = []
    debug = []
    for strategy in STRATEGIES:
        result = run_eval(strategy)
        runs.append(result)
        debug.extend(result["debug"])
        print(f"{strategy}: hit-rate@5 = {result['hit_rate_at_5']:.2f}")

    output = {"corpus": get_corpus_info(), "runs": runs}
    (BASE_DIR / "eval_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (BASE_DIR / "retrieval_debug.json").write_text(
        json.dumps(debug, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("saved eval_results.json and retrieval_debug.json")


if __name__ == "__main__":
    run_all()
