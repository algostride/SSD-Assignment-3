"""
Parametrized pytest suite to validate and benchmark sorting algorithms
provided by SortFactory across different input shapes and sizes.
"""
import random
import os
import time
import pytest
from Sorting_Package.src.sort_factory import SortFactory

REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_FILE = os.path.join(REPORT_DIR, "performance_report.txt")

RNG_SEED = 12345


def generate_data(case: str, size: int) -> list[int]:
    """Generate test data for the given case and size.

    Supported cases: random, sorted, reversed, almost_sorted, few_unique,
    all_equal, negative_numbers. Always returns a list of integers.
    """
    rnd = random.Random(RNG_SEED)
    result: list[int]

    if case == "random":
        result = [rnd.randint(-10**9, 10**9) for _ in range(size)]
    elif case == "sorted":
        result = sorted([rnd.randint(-10**9, 10**9) for _ in range(size)])
    elif case == "reversed":
        result = sorted([rnd.randint(-10**9, 10**9) for _ in range(size)], reverse=True)
    elif case == "almost_sorted":
        arr = sorted([rnd.randint(-10**9, 10**9) for _ in range(size)])
        swaps = max(1, size // 50)
        for _ in range(swaps):
            if size:
                i = rnd.randrange(size)
                j = rnd.randrange(size)
                arr[i], arr[j] = arr[j], arr[i]
        result = arr
    elif case == "few_unique":
        uniques = max(1, min(10, size // 10 + 1))
        base = [rnd.randint(-10**6, 10**6) for _ in range(uniques)]
        result = [rnd.choice(base) for _ in range(size)]
    elif case == "all_equal":
        val = rnd.randint(-10**6, 10**6)
        result = [val] * size
    elif case == "negative_numbers":
        result = [-abs(rnd.randint(0, 10**9)) for _ in range(size)]
    else:
        result = [rnd.randint(-10**9, 10**9) for _ in range(size)]

    return result


ALGOS = ["bubble", "selection", "quick", "merge", "shell"]
ASCENDING_OPTIONS = [True, False]
SIZES = [0, 1, 10, 100, 1000]
CASES = [
    "random",
    "sorted",
    "reversed",
    "almost_sorted",
    "few_unique",
    "all_equal",
    "negative_numbers",
]


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("ascending", ASCENDING_OPTIONS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("case", CASES)
def test_sort_algorithms_varied(algo: str, ascending: bool, size: int, case: str) -> None:
    """Run the requested sorting algorithm on generated data and verify correctness."""
    factory = SortFactory()
    data = generate_data(case, size)
    expected = sorted(data, reverse=not ascending)
    data_copy = list(data)
    start_time = time.perf_counter()
    result = factory.sort(ascending, algo, len(data_copy), data_copy)
    duration = time.perf_counter() - start_time
    if result is None:
        result = data_copy
    assert result == expected, (
        f"{algo} sort failed for ascending={ascending}, size={size}, case='{case}'. "
        f"Expected first 10: {expected[:10]} got first 10: {result[:10]}"
    )
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} - "
            f"{algo.capitalize():10s} | {'Asc' if ascending else 'Desc':4s} | "
            f"size={size:6d} | case={case:14s} | duration={duration:.6f}s\n"
        )
