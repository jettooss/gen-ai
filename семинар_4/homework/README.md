# Домашка семинара 4

Тема: сравнение двух стратегий чанкинга для RAG.

## Команды

```bash
python check_corpus.py
python pipeline.py
python eval.py
```

## Файлы

- `data/` — 8 учебных статей по RAG, поиску и оценке.
- `gold.json` — 10 вопросов для eval.
- `pipeline.py` — простой retrieval pipeline без LLM и без сложного индекса.
- `eval.py` — расчет hit-rate@5.
- `eval_results.json` и `retrieval_debug.json` — результаты прогона.
- `report.md` — краткий отчет по заданию.
