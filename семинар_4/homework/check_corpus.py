from pathlib import Path


DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    files = sorted(DATA_DIR.glob("*.md"))
    total_chars = 0

    if not 5 <= len(files) <= 15:
        raise SystemExit(f"Ожидалось 5-15 документов, найдено {len(files)}")

    for path in files:
        text = path.read_text(encoding="utf-8")
        words = len(text.split())
        total_chars += len(text)
        if not 500 <= words <= 5000:
            raise SystemExit(f"{path.name}: {words} слов, нужно 500-5000")
        print(f"{path.stem}: {words} слов, {len(text)} символов")

    if total_chars < 30_000:
        raise SystemExit(f"Всего {total_chars} символов, нужно минимум 30000")

    print(f"OK: {len(files)} документов, {total_chars} символов")


if __name__ == "__main__":
    main()
