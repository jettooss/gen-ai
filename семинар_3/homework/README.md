# Семинар 3 — домашнее задание

Предметная область: отзывы мобильных приложений.

Источник данных: Hugging Face [`recmeapp/thumbs-up`](https://huggingface.co/datasets/recmeapp/thumbs-up).

`prepare_input.py` выбирает 100 самых длинных отзывов из train parquet и
раскладывает их по пяти исходным файлам. Корпус содержит более 50k оценочных
токенов и подходит для проверки hierarchical Map-Reduce. В фактической выборке
87 приложений: условие "20 отзывов по 5 приложениям" не использовалось, потому
что даже самые длинные отзывы пяти приложений не давали честного корпуса 50k+.

```powershell
python -m pip install -r requirements.txt
python prepare_input.py
python pipeline.py
python bonus.py
```

`pipeline.py` содержит обязательные этапы: IE, аспекты, Map-Reduce и judge.
`bonus.py` содержит autodiscovery, multi-doc, caching и hierarchical Map-Reduce.

Готовые артефакты уже лежат в `output/`, поэтому повторный сетевой прогон для
проверки кода не нужен.
