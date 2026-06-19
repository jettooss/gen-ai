"""
Полный прогон домашки С6 одной командой.

Запуск:
    python run_homework.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))



def main():
    from benchmark_parallel import run_benchmark
    from eval_pwc import run_eval
    from generate_report import main as generate_report
    from measure_critic import run_measurements

    run_benchmark(repeat=1)
    run_measurements(n=10)
    run_eval(n=5)
    generate_report()


if __name__ == "__main__":
    main()
