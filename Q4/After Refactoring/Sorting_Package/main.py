"""
Command-line entrypoint for running sorting algorithms from SortFactory.
Usage: python main.py <algorithm> <asc/desc> <input_file>
"""
import sys
from src.sort_factory import SortFactory


def read_input(file_path: str) -> list[int]:
    """Read integers from a text file, one per line. Non-integer lines are ignored."""
    values: list[int] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                values.append(int(s))
            except ValueError:
                continue
    return values


def main() -> None:
    """Parse CLI args, run the requested sort, and print results."""
    if len(sys.argv) < 4:
        print("Usage: python main.py <algorithm> <asc/desc> <input_file>")
        sys.exit(1)

    algorithm = sys.argv[1].lower()
    ascending = sys.argv[2].lower() == "asc"
    input_file = sys.argv[3]

    data = read_input(input_file)
    sorter = SortFactory()
    sorted_list = sorter.sort(ascending, algorithm, len(data), data)

    if sorted_list is None:
        sorted_list = data

    print(f"\nAlgorithm: {algorithm.capitalize()}")
    print(f"Order: {'Ascending' if ascending else 'Descending'}")
    print(f"Input: {data}")
    print(f"Sorted Output: {sorted_list}")


if __name__ == "__main__":
    main()
